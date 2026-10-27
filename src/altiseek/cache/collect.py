"""Verify a complete cloud output archive before importing or registering any cache."""

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import WorldCache
from altiseek.registry.manifest import Artifact, load_manifest, register
from altiseek.registry.run import sha256_file, write_metadata


def collect_archive(archive: Path, cfg: CacheConfig, manifest: Path) -> dict:
    """Verify every required file and validation record, then import exact bytes.

    Only expected filenames are read from the ZIP, never arbitrary archive paths.
    Existing different files/registry entries are refused before any import begins.
    GPU checks are verified from their recorded results; no local inference runs.
    """
    worlds = {name: load_worlds(cfg, name) for name in cfg.world_set_ids}
    checkpoint_sha = sha256_file(cfg.checkpoint)
    if checkpoint_sha[:16] != cfg.detector_id:
        raise ValueError("local checkpoint identity mismatch")
    cfg.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cache-collect-", dir=cfg.out_dir.parent) as tmp:
        staging = Path(tmp)
        artifacts = []
        files: list[tuple[Path, Path]] = []
        totals = {}
        with ZipFile(archive) as source:
            names = source.namelist()
            if len(set(names)) != len(names):
                raise ValueError("duplicate archive member names")
            summary = json.loads(source.read("production-summary.json"))
            if summary["status"] != "complete" or summary["world_set_ids"] != cfg.world_set_ids:
                raise ValueError("production is incomplete or world-set IDs differ")
            registered = [Artifact(**a) for a in json.loads(source.read("caches/artifacts.json"))]
            by_id = {a.artifact_id: a for a in registered}
            if len(by_id) != len(registered):
                raise ValueError("duplicate artifact IDs")
            observed_ids = set()
            expected_npz = set()
            for name, specs in worlds.items():
                tally = {"worlds": 0, "views": 0, "live_checks": 0, "bytes": 0, "generation_s": 0.0}
                for spec in specs:
                    prefix = f"caches/{name}/{spec.world_id}"
                    path = staging / name / f"{spec.world_id}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    for suffix in (".npz", ".meta.json", ".validation.json"):
                        staged = path.with_suffix(suffix)
                        staged.write_bytes(source.read(prefix + suffix))
                        files.append((staged, cfg.out_dir / name / staged.name))
                    expected_npz.add(prefix + ".npz")
                    cache = WorldCache.load(path, expected=header_for(spec, cfg))
                    meta = resume_metadata(path, spec, cfg)
                    if meta is None:
                        raise ValueError("missing cache metadata")
                    validation = json.loads(path.with_suffix(".validation.json").read_text())
                    for record in (meta, validation):
                        recorded_cfg = CacheConfig.model_validate(record["config"])
                        if header_for(spec, recorded_cfg) != cache.header:
                            raise ValueError("metadata config mismatch")
                        if record["inputs"]["checkpoint"]["sha256"] != checkpoint_sha:
                            raise ValueError("metadata checkpoint mismatch")
                    world_path = cfg.worlds_dir / name / f"{spec.world_id}.json"
                    if meta["inputs"]["world"]["sha256"] != sha256_file(world_path):
                        raise ValueError("metadata world checksum mismatch")
                    seed = 20261026 + spec.seed
                    indices = sorted(
                        int(i)
                        for i in np.random.default_rng(seed).choice(
                            len(cache.views), min(20, len(cache.views)), replace=False
                        )
                    )
                    digest = sha256_file(path)
                    if (
                        validation["passed"] is not True
                        or validation["stage"] != "validate-cache"
                        or validation["cache_id"] != cache.header.cache_id
                        or validation["inputs"]["cache"]["sha256"] != digest
                        or validation["seed"] != seed
                        or validation["view_indices"] != indices
                        or validation["score_atol"] != 1e-4
                        or validation["box_atol_px"] != 0.05
                    ):
                        raise ValueError("invalid live-validation record")
                    if (
                        meta["stage"] != "build-cache"
                        or meta["views"] != len(cache.views)
                        or meta["detections"] != len(cache.detections)
                        or meta["bytes"] != path.stat().st_size
                    ):
                        raise ValueError("invalid build counts")
                    artifact = by_id[cache.header.cache_id]
                    parents = {
                        "world_id": spec.world_id,
                        "detector_id": cfg.detector_id,
                        "world_set_id": cfg.world_set_ids[name],
                    }
                    if (
                        artifact.kind != "cache"
                        or artifact.sha256 != digest
                        or artifact.parents != parents
                    ):
                        raise ValueError("cloud registry mismatch")
                    observed_ids.add(artifact.artifact_id)
                    target = cfg.out_dir / name / path.name
                    artifacts.append(replace(artifact, location=f"local:{target.as_posix()}"))
                    tally["worlds"] += 1
                    tally["views"] += len(cache.views)
                    tally["live_checks"] += len(indices)
                    tally["bytes"] += path.stat().st_size
                    tally["generation_s"] += meta["timing"]["total_s"]
                expected_counts = {
                    "expected": len(specs),
                    "built": len(specs),
                    "validated": len(specs),
                }
                if summary["sets"][name] != expected_counts:
                    raise ValueError("summary counts disagree with actual files")
                totals[name] = tally
            if (
                observed_ids != set(by_id)
                or {n for n in names if n.endswith(".npz")} != expected_npz
            ):
                raise ValueError("archive/registry contains missing or unexpected caches")
        existing = {(a.kind, a.artifact_id): a for a in load_manifest(manifest)}
        for artifact in artifacts:
            old = existing.get((artifact.kind, artifact.artifact_id))
            if old is not None and old != artifact:
                raise ValueError("existing artifact registered differently")
        for staged, target in files:
            if target.exists() and sha256_file(staged) != sha256_file(target):
                raise ValueError(f"existing local file differs: {target}")
        # All checks pass before any output/registry mutation.
        for staged, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(staged, temp)
            temp.replace(target)
        pending = staging / "manifest.json"
        if manifest.exists():
            shutil.copyfile(manifest, pending)
        for artifact in artifacts:
            register(pending, artifact)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        pending.replace(manifest)
        report = {
            "complete": True,
            "sets": totals,
            "elapsed_cloud_s": summary["elapsed_s"],
            "archive_sha256": sha256_file(archive),
        }
        write_json(cfg.out_dir / "collection-summary.json", report)
        write_metadata(
            cfg.out_dir / "collect.meta.json",
            stage="collect-cache",
            config=cfg,
            inputs={"archive": archive},
            outputs={"manifest": manifest},
            extra=report,
        )
        return report
