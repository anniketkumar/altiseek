"""Versioned V0-V3 sensor models with a shared scalar/vector posterior interface."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from altiseek.calibration.config import SensorConfig
from altiseek.calibration.temperature import calibrated
from altiseek.config import Config


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]


class BinaryRates(Config):
    recall: float = Field(gt=0, lt=1)
    false_alarm: float = Field(gt=0, lt=1)


class LevelModel(Config):
    name: str
    binary: BinaryRates
    soft: BinaryRates
    temperature: float = Field(gt=0, allow_inf_nan=False)
    occupied_hist: tuple[float, ...]
    empty_hist: tuple[float, ...]

    @model_validator(mode="after")
    def histograms(self) -> "LevelModel":
        for h in (self.occupied_hist, self.empty_hist):
            if not h or not np.isfinite(h).all() or min(h) <= 0 or not np.isclose(sum(h), 1):
                raise ValueError("histogram probabilities must be positive and sum to one")
        return self


class SensorModel(Config):
    schema_version: Literal[1] = 1
    sensor_model_id: str = ""
    detector_id: str
    fit_world_set_id: str
    fit_config: SensorConfig
    # Geometry, inference and renderer identity without per-world identity.
    cache_contract: dict[str, Any]
    fit_cache_ids: tuple[str, ...]
    global_binary: BinaryRates
    levels: tuple[LevelModel, ...]

    def content_id(self) -> str:
        return digest(self.model_dump(mode="json", exclude={"sensor_model_id"}))

    @model_validator(mode="after")
    def check_model(self) -> "SensorModel":
        if not self.levels or len({level.name for level in self.levels}) != len(self.levels):
            raise ValueError("model needs uniquely named levels")
        if any(
            len(h) != self.fit_config.histogram_bins
            for level in self.levels
            for h in (level.occupied_hist, level.empty_hist)
        ):
            raise ValueError("histogram dimensions disagree with fit config")
        if self.sensor_model_id and self.sensor_model_id != self.content_id():
            raise ValueError("sensor_model_id does not match content")
        return self

    def save(self, path: Path) -> None:
        if self.sensor_model_id != self.content_id():
            raise ValueError("model must be identified before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        detector_id: str,
        fit_world_set_id: str | None = None,
        cache_contract: dict[str, Any] | None = None,
    ) -> "SensorModel":
        model = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            not model.sensor_model_id
            or model.detector_id != detector_id
            or (fit_world_set_id is not None and model.fit_world_set_id != fit_world_set_id)
            or (cache_contract is not None and model.cache_contract != cache_contract)
        ):
            raise ValueError("sensor model identity/parent mismatch")
        return model

    def posterior(
        self,
        prior: float | np.ndarray,
        level: int,
        score: float | np.ndarray | None,
        variant: Literal["V0", "V1", "V2", "V3"],
    ) -> np.ndarray:
        """Update cell probabilities; None or zero means no retained swimmer detection.

        V0-V2 threshold raw scores at tau; V3 uses every retained score at tau_min.
        This function does not enforce observation history: M7 must gate re-looks.
        """
        if not 0 <= level < len(self.levels):
            raise ValueError("unknown level")
        p, s = np.broadcast_arrays(
            np.asarray(prior, dtype=float), np.asarray(0 if score is None else score, dtype=float)
        )
        if (
            not np.isfinite(p).all()
            or not np.isfinite(s).all()
            or ((p < 0) | (p > 1) | (s < 0) | (s > 1)).any()
        ):
            raise ValueError("prior and score must be finite probabilities")
        cfg, lm = self.fit_config, self.levels[level]
        limit = cfg.log_odds_limit
        floor = 1 / (1 + np.exp(limit))
        p = np.clip(p, floor, 1 - floor)
        odds = np.log(p) - np.log1p(-p)
        if variant == "V0":
            positive = np.maximum(p, s)
            negative = np.exp(-np.logaddexp(0, -(odds + np.log1p(-cfg.naive_recall))))
            return np.clip(np.where(s >= cfg.tau, positive, negative), floor, 1 - floor)
        if variant in ("V1", "V2"):
            rates = self.global_binary if variant == "V1" else lm.binary
            lr = np.where(
                s >= cfg.tau,
                rates.recall / rates.false_alarm,
                (1 - rates.recall) / (1 - rates.false_alarm),
            )
        elif variant == "V3":
            rates = lm.soft
            bins = np.minimum(
                (calibrated(s, lm.temperature) * cfg.histogram_bins).astype(int),
                cfg.histogram_bins - 1,
            )
            lr = np.where(
                s >= cfg.tau_min,
                rates.recall
                * np.asarray(lm.occupied_hist)[bins]
                / (rates.false_alarm * np.asarray(lm.empty_hist)[bins]),
                (1 - rates.recall) / (1 - rates.false_alarm),
            )
        else:
            raise ValueError("unknown sensor variant")
        updated = np.clip(odds + np.log(lr), -limit, limit)
        return np.exp(-np.logaddexp(0, -updated))
