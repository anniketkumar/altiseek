"""Shared stage inputs and resumable manifests for cache jobs."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from altiseek.cache.build import header_for
from altiseek.cache.config import CacheConfig
from altiseek.cache.storage import WorldCache
from altiseek.detector.infer import Detector, detector_id_of
from altiseek.registry.run import sha256_file
from altiseek.world.library import Library
from altiseek.world.spec import WorldSpec


def load_worlds(cfg: CacheConfig, name: str) -> list[WorldSpec]:
    root = cfg.worlds_dir / name
    meta = json.loads((root / "world_set.meta.json").read_text(encoding="utf-8"))
    expected = cfg.world_set_ids[name]
    if meta["world_set_id"] != expected or sha256_file(root / "worlds.csv")[:16] != expected:
        raise ValueError(f"world_set_id mismatch for {name}")
    with (root / "worlds.csv").open(encoding="utf-8", newline="") as fh:
        worlds = [WorldSpec.load(root / f"{r['world_id']}.json") for r in csv.DictReader(fh)]
    if len(worlds) != meta["worlds"] or len({w.world_id for w in worlds}) != len(worlds):
        raise ValueError("world set count/uniqueness mismatch")
    for w in worlds:
        if (
            w.world_set != name
            or w.split != meta["split"]
            or w.library_id != cfg.library_ids[w.split]
            or w.library_id != meta["library_id"]
        ):
            raise ValueError("world set/library/split mismatch")
    return worlds


def load_libraries(cfg: CacheConfig) -> dict[str, Library]:
    libraries = {split: Library.load(path) for split, path in cfg.libraries.items()}
    for split, lib in libraries.items():
        if lib.library_id != cfg.library_ids[split] or lib.split != split:
            raise ValueError(f"library_id/split mismatch for {split}; stop, never mix artifacts")
        digest = hashlib.sha256()
        for name in ("backgrounds.csv", "objects.csv", "conditions.json"):
            digest.update((lib.path / name).read_bytes())
        if digest.hexdigest()[:16] != lib.library_id:
            raise ValueError(f"library index checksum mismatch for {split}")
        for row in lib.backgrounds:
            if sha256_file(lib.patch_path(row.patch_id)) != row.sha256:
                raise ValueError(f"library patch checksum mismatch: {row.patch_id}")
        for row in lib.objects:
            if sha256_file(lib.crop_path(row.crop_id)) != row.sha256:
                raise ValueError(f"library crop checksum mismatch: {row.crop_id}")
    return libraries


def resume_metadata(path: Path, spec: WorldSpec, cfg: CacheConfig) -> dict[str, Any] | None:
    """Verify existing bytes, including an orphan file left before metadata was saved.

    An orphan with a matching header can be regenerated. A mismatched artifact must
    never be overwritten, even if it has no metadata/completion entry.
    """
    if not path.exists():
        return None
    header = header_for(spec, cfg)
    WorldCache.load(path, expected=header)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["outputs"]["cache"]["sha256"] != sha256_file(path):
        raise ValueError(f"cache file checksum mismatch: {path}")
    if (
        meta["cache_id"] != header.cache_id
        or meta["world_id"] != spec.world_id
        or meta["detector_id"] != cfg.detector_id
        or meta["world_set_id"] != cfg.world_set_ids[spec.world_set]
    ):
        raise ValueError("cache metadata identity mismatch")
    return meta


def gpu_detector(cfg: CacheConfig, device: str) -> Detector:
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        raise ValueError("cache inference requires a cloud CUDA GPU (CLAUDE.md section 2)")
    if detector_id_of(cfg.checkpoint) != cfg.detector_id:
        raise ValueError("checkpoint detector_id mismatch")
    return Detector(
        cfg.checkpoint,
        device=device,
        score_min=cfg.tau_min,
        iou=cfg.inference.iou,
        max_det=cfg.inference.max_det,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
