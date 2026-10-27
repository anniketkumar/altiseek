import numpy as np
import pytest

from altiseek.calibration.config import SensorConfig
from altiseek.calibration.real import real_outcomes
from altiseek.detector.infer import Detection
from altiseek.detector.views import Label
from altiseek.geometry.rectify import ViewWindow


def test_ground_grid_y_flip_boundary_and_fill():
    window = ViewWindow(0, 0, 1, 10)
    valid = np.ones((10, 10), dtype=bool)
    truth = [Label(0, 1, 8, 2, 9)]
    detections = [
        Detection(0, 0.8, 1, 8, 2, 9),
        Detection(0, 0.6, 1, 8, 2, 9),
        Detection(0, 0.7, 2.4, 8, 2.8, 9),
        Detection(1, 1, 5, 1, 6, 2),
    ]
    result = real_outcomes(window, valid, truth, detections, 2.5, SensorConfig())
    assert result["counts"] == [[1, 15, 1, 1], [1, 15, 1, 1]]
    assert result["scores"] == [0.8, 0.7]
    assert result["labels"] == [True, False]
    valid[8, 1] = False  # target cell's centre is in fill
    masked = real_outcomes(window, valid, truth, detections, 2.5, SensorConfig())
    assert masked["counts"] == [[0, 15, 0, 1], [0, 15, 0, 1]]
    with pytest.raises(ValueError, match="mask"):
        real_outcomes(window, valid[:1], truth, detections, 2.5, SensorConfig())


def test_grid_origin_is_not_view_corner():
    # Same physical target/detection across shifted overlapping windows stays in one cell.
    first = real_outcomes(
        ViewWindow(0, 0, 1, 10),
        np.ones((10, 10), dtype=bool),
        [Label(0, 2.3, 8, 2.5, 9)],
        [Detection(0, 0.8, 2.5, 8, 2.7, 9)],
        2.5,
        SensorConfig(),
    )
    second = real_outcomes(
        ViewWindow(1, 0, 1, 10),
        np.ones((10, 10), dtype=bool),
        [Label(0, 1.3, 8, 1.5, 9)],
        [Detection(0, 0.8, 1.5, 8, 1.7, 9)],
        2.5,
        SensorConfig(),
    )
    assert first["labels"] == second["labels"] == [False]
