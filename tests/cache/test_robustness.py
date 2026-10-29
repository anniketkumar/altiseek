from pathlib import Path

import pytest

from altiseek.cache.config import CacheConfig
from altiseek.cache.robustness import STRESS_IDS, case_config
from altiseek.config import load_config


def test_robustness_contracts_preserve_physical_geometry():
    original = load_config("configs/cache/default.yaml", CacheConfig)
    detector = case_config(original, "detector", Path("out"))
    camera = case_config(original, "camera", Path("out"))
    stress = case_config(original, "stress", Path("out"))
    assert detector.detector_id == "78aba879cc50e025"
    assert detector.render == original.render
    assert camera.render.camera.resolution_px == 320
    assert camera.render.levels == original.render.levels
    assert camera.render.camera.hfov_deg == original.render.camera.hfov_deg
    assert camera.detector_id == original.detector_id
    assert set(camera.world_set_ids) == {"sensor-fit", "test"}
    assert stress.world_set_ids == STRESS_IDS
    assert stress.render == original.render
    assert original.render.camera.resolution_px == 640
    assert len({c.out_dir for c in (detector, camera, stress)}) == 3
    with pytest.raises(ValueError, match="unknown"):
        case_config(original, "oops", Path("out"))
