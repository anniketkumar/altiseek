"""Archive collection is fail-closed and does not register partial/unverified results."""

import json
from dataclasses import asdict
from zipfile import ZipFile

import numpy as np
import pytest

from altiseek.cache import collect
from altiseek.cache.build import header_for
from altiseek.registry.manifest import Artifact, load_manifest
from altiseek.registry.run import sha256_file
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]


@pytest.fixture
def output(tmp_path, cfg, built, monkeypatch):
    spec, cache = built
    cfg.checkpoint.write_bytes(b"fake checkpoint")
    checkpoint_sha = sha256_file(cfg.checkpoint)
    cfg = cfg.model_copy(
        update={
            "detector_id": checkpoint_sha[:16],
            "world_set_ids": {"sensor-fit": "set"},
            "worlds_dir": tmp_path / "worlds",
            "out_dir": tmp_path / "collected",
        }
    )
    world_path = cfg.worlds_dir / "sensor-fit" / f"{spec.world_id}.json"
    world_path.parent.mkdir(parents=True)
    spec.save(world_path)
    cache.header = header_for(spec, cfg)
    path = tmp_path / "cache.npz"
    cache.save(path)
    digest = sha256_file(path)
    monkeypatch.setattr(collect, "load_worlds", lambda *a: [spec])
    parents = {"world_id": spec.world_id, "detector_id": cfg.detector_id, "world_set_id": "set"}
    artifact = Artifact(
        "cache",
        cache.header.cache_id,
        digest,
        "local:/kaggle/working/cache.npz",
        parents,
        "2026-10-27T00:00:00+00:00",
        {"views": len(cache.views)},
    )
    meta = {
        "stage": "build-cache",
        "config": cfg.model_dump(mode="json"),
        **parents,
        "cache_id": cache.header.cache_id,
        "inputs": {
            "world": {"sha256": sha256_file(world_path)},
            "checkpoint": {"sha256": checkpoint_sha},
        },
        "outputs": {"cache": {"sha256": digest}},
        "views": len(cache.views),
        "detections": len(cache.detections),
        "bytes": path.stat().st_size,
        "timing": {"total_s": 1.0},
    }
    seed = 20261026 + spec.seed
    validation = {
        "stage": "validate-cache",
        "config": cfg.model_dump(mode="json"),
        "cache_id": cache.header.cache_id,
        "passed": True,
        "seed": seed,
        "score_atol": 1e-4,
        "box_atol_px": 0.05,
        "view_indices": sorted(
            int(i) for i in np.random.default_rng(seed).choice(len(cache.views), 20, replace=False)
        ),
        "inputs": {"cache": {"sha256": digest}, "checkpoint": {"sha256": checkpoint_sha}},
    }
    prefix = f"caches/sensor-fit/{spec.world_id}"
    files = {
        prefix + ".npz": path.read_bytes(),
        prefix + ".meta.json": json.dumps(meta).encode(),
        prefix + ".validation.json": json.dumps(validation).encode(),
        "caches/artifacts.json": json.dumps([asdict(artifact)]).encode(),
        "production-summary.json": json.dumps(
            {
                "status": "complete",
                "elapsed_s": 2.0,
                "world_set_ids": cfg.world_set_ids,
                "sets": {"sensor-fit": {"expected": 1, "built": 1, "validated": 1}},
            }
        ).encode(),
    }
    return cfg, files, prefix


def zip_files(path, files):
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_collect_verified_bytes_and_idempotent_registry(tmp_path, output):
    cfg, files, prefix = output
    archive = tmp_path / "output.zip"
    zip_files(archive, files)
    manifest = tmp_path / "manifest.json"
    report = collect.collect_archive(archive, cfg, manifest)
    assert report["complete"]
    assert report["sets"]["sensor-fit"]["live_checks"] == 20
    (entry,) = load_manifest(manifest)
    assert entry.location.startswith(f"local:{cfg.out_dir.as_posix()}")
    local = cfg.out_dir / prefix.removeprefix("caches/")
    assert local.with_suffix(".npz").read_bytes() == files[prefix + ".npz"]
    collect.collect_archive(archive, cfg, manifest)
    assert len(load_manifest(manifest)) == 1


@pytest.mark.parametrize("damage", ["seed", "checksum", "missing"])
def test_collection_failure_does_not_install_or_register(tmp_path, output, damage):
    cfg, files, prefix = output
    record = json.loads(files[prefix + ".validation.json"])
    if damage == "seed":
        record["seed"] += 1
    elif damage == "checksum":
        record["inputs"]["cache"]["sha256"] = "bad"
    else:
        del files[prefix + ".npz"]
    files[prefix + ".validation.json"] = json.dumps(record).encode()
    archive = tmp_path / "bad.zip"
    zip_files(archive, files)
    manifest = tmp_path / "manifest.json"
    with pytest.raises((ValueError, KeyError)):
        collect.collect_archive(archive, cfg, manifest)
    assert not manifest.exists()
    assert not cfg.out_dir.exists()
