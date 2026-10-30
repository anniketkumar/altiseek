import numpy as np
import pytest

from altiseek.belief.state import Belief
from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.deploy.perception import project_detections
from altiseek.detector.infer import Detection
from tests.deploy.test_projection import nadir
from tests.env.test_search import built, cfg, library, sensor

__all__ = ["built", "cfg", "library", "sensor"]


def test_boxes_use_pixel_boundaries_and_raw_scores():
    detection = Detection(class_id=0, score=0.7, x_min=310, y_min=310, x_max=330, y_max=330)
    rows = project_detections([detection], nadir(), 2.5, 0.05)
    assert rows["x_m"][0] == pytest.approx(150)
    assert rows["y_m"][0] == pytest.approx(150)
    assert (rows["ix"][0], rows["iy"][0]) == (60, 60)
    assert rows["score"][0] == pytest.approx(0.7)
    assert len(project_detections([detection], nadir(), 2.5, 0.8)) == 0


@pytest.mark.parametrize("relook", ["first", "distinct_variant"])
def test_partial_cells_never_update_or_consume_first_look(sensor, relook):
    belief = Belief(np.full((2, 2), 0.25), 1, 1, sensor, relook)
    initial = belief.log_odds.copy()
    det = np.zeros(1, dtype=DETECTION_DTYPE)
    det["score"], det["ix"], det["iy"] = 0.9, 1, 1
    visible = np.array([[True, False], [False, True]])
    args = (det, (0, 0, 2, 2), 0, 0, "V2", np.empty((0, 2)), 1)
    assert belief.update(*args, visible_cells=visible) == 2
    np.testing.assert_array_equal(belief.log_odds[~visible], initial[~visible])
    np.testing.assert_array_equal(belief.observed[0], visible)
    np.testing.assert_array_equal(belief.used_variants[0], visible.astype(np.uint32))
    assert belief.log_odds[1, 1] > initial[1, 1]
    assert belief.log_odds[0, 0] < initial[0, 0]
    assert belief.update(*args, visible_cells=visible) == 0
    assert belief.update(*args, visible_cells=~visible) == 2
    assert belief.observed.all()


def test_all_visible_matches_cached_update(sensor):
    first = Belief(np.full((2, 2), 0.25), 1, 1, sensor)
    second = Belief(np.full((2, 2), 0.25), 1, 1, sensor)
    args = (np.zeros(0, dtype=DETECTION_DTYPE), (0, 0, 2, 2), 0, 0, "V3", np.empty((0, 2)), 1)
    assert first.update(*args) == second.update(*args, visible_cells=np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(first.log_odds, second.log_odds)
    np.testing.assert_array_equal(first.observed, second.observed)
