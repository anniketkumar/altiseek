"""Frozen D46 cache contracts; retain original worlds and output namespaces."""

from pathlib import Path

from altiseek.cache.config import CacheConfig

STRESS_IDS = {"stress-low": "2ad630d766f25e9b", "stress-high": "dcd091267fc6eb58"}
CASES = ("detector", "camera", "stress")


def case_config(original: CacheConfig, case: str, root: Path) -> CacheConfig:
    if case not in CASES:
        raise ValueError(f"unknown robustness case: {case}")
    changes = {
        "out_dir": root / case / "caches",
        "world_set_ids": {s: original.world_set_ids[s] for s in ("sensor-fit", "test")},
    }
    if case == "detector":
        changes.update(
            detector_id="78aba879cc50e025",
            checkpoint=Path("data/kaggle/altiseek-detector-v2/detector/train/weights/best.pt"),
        )
    elif case == "camera":
        camera = original.render.camera.model_copy(update={"resolution_px": 320})
        changes["render"] = original.render.model_copy(update={"camera": camera})
    else:
        changes.update(world_set_ids=STRESS_IDS, worlds_dir=Path("data/m9-stress/worlds"))
    return CacheConfig.model_validate(original.model_dump() | changes)
