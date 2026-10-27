"""World-cluster bootstrap, B-only fitting and fixed-model evaluation."""

from typing import Any

import numpy as np

from altiseek.cache.config import CacheConfig
from altiseek.calibration.config import SensorConfig
from altiseek.calibration.model import BinaryRates, LevelModel, SensorModel
from altiseek.calibration.outcomes import WorldOutcomes
from altiseek.calibration.temperature import calibrated, fit_temperature, reliability


def ratio_summary(
    numerators: np.ndarray, denominators: np.ndarray, picks: np.ndarray, alpha: float
) -> dict[str, Any]:
    n, d = int(numerators.sum()), int(denominators.sum())
    ns, ds = numerators[picks].sum(axis=1), denominators[picks].sum(axis=1)
    ratios = ns[ds > 0] / ds[ds > 0]
    return {
        "numerator": n,
        "denominator": d,
        "raw": n / d if d else None,
        "smoothed": (n + alpha) / (d + 2 * alpha),
        "ci95": np.quantile(ratios, [0.025, 0.975]).tolist() if len(ratios) else None,
        "bootstrap_valid": len(ratios),
    }


def measurements(worlds: list[WorldOutcomes], cfg: SensorConfig) -> dict[str, Any]:
    if not worlds:
        raise ValueError("no worlds")
    counts, relook = np.stack([w.counts for w in worlds]), np.stack([w.relook for w in worlds])
    picks = np.random.default_rng(cfg.seed).integers(
        len(worlds), size=(cfg.bootstrap_samples, len(worlds))
    )

    def summarize(c: np.ndarray) -> dict[str, Any]:
        return {
            "recall": ratio_summary(c[:, 2], c[:, 0], picks, cfg.pseudocount),
            "false_alarm": ratio_summary(c[:, 3], c[:, 1], picks, cfg.pseudocount),
        }

    levels = []
    for li in range(counts.shape[1]):
        row = {}
        for ti, name in enumerate(("binary", "soft")):
            row[name] = summarize(counts[:, li, ti])
            row[name]["conditional_recall"] = ratio_summary(
                relook[:, li, ti, 0], relook[:, li, ti, 1], picks, cfg.pseudocount
            )
        levels.append(row)
    return {
        "worlds": len(worlds),
        "levels": levels,
        "global_binary": summarize(counts[:, :, 0].sum(axis=1)),
    }


def joined(worlds: list[WorldOutcomes], level: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([w.scores[w.levels == level] for w in worlds]),
        np.concatenate([w.labels[w.levels == level] for w in worlds]),
    )


def calibration_report(worlds: list[WorldOutcomes], model: SensorModel) -> list[dict[str, Any]]:
    report = []
    for li, level in enumerate(model.levels):
        s, y = joined(worlds, li)
        report.append(
            {
                "level": level.name,
                "before": reliability(s, y, model.fit_config.reliability_bins),
                "after": reliability(
                    calibrated(s, level.temperature), y, model.fit_config.reliability_bins
                ),
            }
        )
    return report


def fit(
    worlds: list[WorldOutcomes],
    cfg: SensorConfig,
    cache: CacheConfig,
    cache_ids: list[str],
    contract: dict[str, Any],
) -> tuple[SensorModel, dict[str, Any]]:
    """Caller verifies the complete sensor-fit set and all cache checksums."""
    stats = measurements(worlds, cfg)

    def rates(values: dict[str, Any]) -> BinaryRates:
        return BinaryRates(**{k: values[k]["smoothed"] for k in ("recall", "false_alarm")})

    levels = []
    for li, level in enumerate(cache.render.levels.levels):
        s, y = joined(worlds, li)
        temperature = fit_temperature(s, y, cfg.temperature_min, cfg.temperature_max)
        calibrated_scores = calibrated(s, temperature)
        histograms = []
        for mask in (y, ~y):
            h = np.histogram(calibrated_scores[mask], bins=cfg.histogram_bins, range=(0, 1))[0]
            h = h.astype(float) + cfg.pseudocount
            histograms.append(tuple(h / h.sum()))
        levels.append(
            LevelModel(
                name=level.name,
                binary=rates(stats["levels"][li]["binary"]),
                soft=rates(stats["levels"][li]["soft"]),
                temperature=temperature,
                occupied_hist=histograms[0],
                empty_hist=histograms[1],
            )
        )
    model = SensorModel(
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
        fit_config=cfg,
        cache_contract=contract,
        fit_cache_ids=tuple(sorted(cache_ids)),
        global_binary=rates(stats["global_binary"]),
        levels=tuple(levels),
    )
    model = model.model_copy(update={"sensor_model_id": model.content_id()})
    stats["calibration"] = calibration_report(worlds, model)
    return model, stats
