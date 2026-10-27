import json

import numpy as np
import pytest

from altiseek.cache.storage import DETECTION_DTYPE
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.fit import measurements
from altiseek.calibration.model import BinaryRates, LevelModel, SensorModel
from altiseek.calibration.outcomes import WorldOutcomes, cell_outcomes, collect_world
from altiseek.calibration.temperature import calibrated, fit_temperature, reliability
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]


@pytest.fixture
def model():
    rates = BinaryRates(recall=0.8, false_alarm=0.1)
    level = LevelModel(
        name="L1",
        binary=rates,
        soft=rates,
        temperature=1,
        occupied_hist=(0.2, 0.8),
        empty_hist=(0.8, 0.2),
    )
    result = SensorModel(
        detector_id="detector",
        fit_world_set_id="fit",
        cache_contract={},
        fit_cache_ids=("cache",),
        fit_config=SensorConfig(histogram_bins=2),
        global_binary=rates,
        levels=(level,),
    )
    return result.model_copy(update={"sensor_model_id": result.content_id()})


def test_hand_computed_posteriors(model):
    assert model.posterior(0.2, 0, 0.8, "V0") == pytest.approx(0.8)
    assert model.posterior(0.2, 0, None, "V0") == pytest.approx(0.02 / 0.82)
    for variant in ("V1", "V2"):
        assert model.posterior(0.2, 0, 0.8, variant) == pytest.approx(2 / 3)
        assert model.posterior(0.2, 0, None, variant) == pytest.approx(1 / 19)
    assert model.posterior(0.2, 0, 0.8, "V3") == pytest.approx(8 / 9)
    # A low retained score is negative in V2, but scored evidence in V3.
    assert model.posterior(0.2, 0, 0.2, "V3") == pytest.approx(1 / 3)
    assert model.posterior(0.2, 0, 0.2, "V2") == pytest.approx(1 / 19)


@pytest.mark.parametrize("variant", ["V0", "V1", "V2", "V3"])
def test_clipping_and_vectorisation(model, variant):
    p = model.posterior(np.array([0, 0.2, 1]), 0, np.array([1, 0, 0]), variant)
    assert np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    with pytest.raises(ValueError, match="finite"):
        model.posterior(float("nan"), 0, None, variant)


def test_identity_and_loader(tmp_path, model):
    path = tmp_path / "sensor.json"
    model.save(path)
    assert SensorModel.load(path, detector_id="detector", fit_world_set_id="fit") == model
    for kwargs in (
        {"detector_id": "bad"},
        {"detector_id": "detector", "fit_world_set_id": "T"},
        {"detector_id": "detector", "cache_contract": {"changed": True}},
    ):
        with pytest.raises(ValueError, match="mismatch"):
            SensorModel.load(path, **kwargs)
    body = json.loads(path.read_text())
    body["levels"][0]["temperature"] = 2
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="content"):
        SensorModel.load(path, detector_id="detector")


def test_histograms_reject_bad_probabilities(model):
    body = model.model_dump(exclude={"sensor_model_id"})
    body["levels"][0]["empty_hist"] = [0, 1]
    with pytest.raises(ValueError, match="histogram"):
        SensorModel.model_validate(body)


def test_calibration_improves_bce_and_preserves_order():
    s = np.tile([0.9, 0.1], 100)
    y = np.tile([True, False, False, True], 50)
    t = fit_temperature(s, y, 0.05, 20)
    assert t == pytest.approx(20)
    assert reliability(calibrated(s, t), y, 15)["bce"] < reliability(s, y, 15)["bce"]
    assert np.all(np.diff(calibrated(np.linspace(0, 1, 50), t)) > 0)
    assert reliability(np.array([]), np.array([], dtype=bool), 15)["ece"] is None


def test_cell_maximum_classes_and_bounds():
    rows = np.zeros(5, dtype=DETECTION_DTYPE)
    rows["ix"], rows["iy"] = [0, 0, 1, 2, -1], [0, 0, 0, 0, 0]
    rows["score"], rows["class_id"] = [0.2, 0.8, 0.9, 0.7, 1], [0, 0, 1, 0, 0]
    assert cell_outcomes(rows, (0, 0, 2, 2)) == {(0, 0): pytest.approx(0.8)}


def test_count_denominators_and_pairs(built):
    spec, cache = built
    result = collect_world(spec, cache, SensorConfig())
    for li, level in enumerate(cache.header.render.levels.levels):
        view_count = int((cache.views[:, 0] == li).sum())
        assert (result.counts[li, :, :2].sum(axis=1) == view_count * level.footprint_cells**2).all()
        assert (result.counts[li, :, 2:] <= result.counts[li, :, :2]).all()
        assert (result.relook[li, :, 0] <= result.relook[li, :, 1]).all()


def test_bootstrap_world_clusters_and_zero_support():
    worlds = []
    for i, hits in enumerate((0, 10)):
        c = np.array([[[10, 100, hits, 0], [10, 100, hits, 0]]])
        worlds.append(
            WorldOutcomes(
                str(i),
                c,
                np.zeros((1, 2, 2), dtype=int),
                np.array([0]),
                np.array([0.9]),
                np.array([True]),
            )
        )
    cfg = SensorConfig(bootstrap_samples=200)
    a = measurements(worlds, cfg)
    assert a == measurements(worlds, cfg)
    assert a["levels"][0]["binary"]["recall"]["ci95"] == [0, 1]
    assert a["levels"][0]["binary"]["false_alarm"]["smoothed"] > 0
    assert a["levels"][0]["binary"]["conditional_recall"]["raw"] is None
