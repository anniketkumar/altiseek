import numpy as np
import pytest

from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.calibration.config import SensorConfig
from altiseek.deploy.calibration import load_view, outcomes, save_view
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]


def test_live_view_resume_rejects_corruption_and_wrong_contract(tmp_path):
    evidence = np.zeros(0, dtype=DETECTION_DTYPE)
    mask = np.ones((2, 2), dtype=bool)
    save_view(tmp_path, (0, 1, 2), "contract", evidence, mask, {"world_id": "world"})
    saved, detections, visible = load_view(tmp_path, (0, 1, 2), "contract")
    assert saved["world_id"] == "world"
    assert len(detections) == 0
    np.testing.assert_array_equal(visible, mask)
    with pytest.raises(ValueError, match="identity"):
        load_view(tmp_path, (0, 1, 2), "other")
    with pytest.raises(ValueError, match="overwrite"):
        save_view(tmp_path, (0, 1, 2), "contract", evidence, mask, {})
    (tmp_path / "0-1-2.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_view(tmp_path, (0, 1, 2), "contract")


def test_live_outcomes_exclude_partial_cells_and_keep_raw_scores(built):
    original, _ = built
    spec = original.model_copy(update={"world_set": "sensor-fit"})
    grid = round(spec.area_m / spec.cell_m)
    target = spec.targets[0]
    cell = (int(target.x_m // spec.cell_m), int(target.y_m // spec.cell_m))
    mask = np.zeros((grid, grid), dtype=bool)
    mask[cell] = True
    detections = np.zeros(1, dtype=DETECTION_DTYPE)
    detections["ix"], detections["iy"], detections["score"] = *cell, 0.9
    metadata = {"key": [0, 0, 0], "world_id": spec.world_id}
    result = outcomes(spec, [(metadata, detections, mask)], SensorConfig(), 1)
    assert result.counts[0, 0].tolist() == [1, 0, 1, 0]
    assert result.scores.tolist() == pytest.approx([0.9])
    mask[:] = False
    result = outcomes(spec, [(metadata, detections, mask)], SensorConfig(), 1)
    assert result.counts.sum() == 0
    assert len(result.scores) == 0
    with pytest.raises(ValueError, match="sensor-fit B"):
        outcomes(spec.model_copy(update={"split": "T"}), [], SensorConfig(), 1)
