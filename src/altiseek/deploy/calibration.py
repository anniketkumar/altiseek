"""Verified per-view live calibration storage and shared sensor-fit outcomes."""

import json
from pathlib import Path

import numpy as np

from altiseek.calibration.config import SensorConfig
from altiseek.calibration.outcomes import WorldOutcomes, cell_outcomes
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


def load_view(root: Path, key: tuple[int, int, int], contract_id: str) -> tuple | None:
    stem = "-".join(map(str, key))
    metadata = root / f"{stem}.json"
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text())
    data = root / f"{stem}.npz"
    if (
        saved["contract_id"] != contract_id
        or saved["key"] != list(key)
        or sha256_file(data) != saved["sha256"]
    ):
        raise ValueError("Live calibration view identity/checksum mismatch")
    with np.load(data, allow_pickle=False) as arrays:
        return saved, arrays["detections"].copy(), arrays["visible"].copy()


def save_view(
    root: Path,
    key: tuple[int, int, int],
    contract_id: str,
    detections: np.ndarray,
    visible: np.ndarray,
    metadata: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if load_view(root, key, contract_id) is not None:
        raise ValueError("Refuse to overwrite a completed live calibration view")
    stem = "-".join(map(str, key))
    temporary = root / f"{stem}.tmp.npz"
    np.savez_compressed(temporary, detections=detections, visible=visible)
    data = root / f"{stem}.npz"
    temporary.replace(data)
    report = {**metadata, "contract_id": contract_id, "key": list(key), "sha256": sha256_file(data)}
    temp_json = root / f"{stem}.tmp.json"
    temp_json.write_text(json.dumps(report, indent=2) + "\n")
    temp_json.replace(root / f"{stem}.json")


def outcomes(
    spec: WorldSpec, views: list[tuple], cfg: SensorConfig, level_count: int
) -> WorldOutcomes:
    """Every fully observed cell is a trial; raw centre scores match shared fitting."""
    if spec.split != "B" or spec.world_set != "sensor-fit":
        raise ValueError("Gazebo sensor fitting requires sensor-fit B worlds")
    grid = round(spec.area_m / spec.cell_m)
    occupied = {(int(t.x_m // spec.cell_m), int(t.y_m // spec.cell_m)) for t in spec.targets}
    counts = np.zeros((level_count, 2, 4), dtype=np.int64)
    levels, scores, labels = [], [], []
    for metadata, detections, visible in views:
        li = metadata["key"][0]
        if visible.shape != (grid, grid) or visible.dtype != np.bool_:
            raise ValueError("Invalid live visibility grid")
        truth = {cell for cell in occupied if visible[cell]}
        values = {
            cell: score
            for cell, score in cell_outcomes(detections, (0, 0, grid, grid)).items()
            if visible[cell]
        }
        for cell, score in values.items():
            levels.append(li)
            scores.append(score)
            labels.append(cell in truth)
        for ti, threshold in enumerate((cfg.tau, cfg.tau_min)):
            positive = {cell for cell, score in values.items() if score >= threshold}
            counts[li, ti] += (
                len(truth),
                int(visible.sum()) - len(truth),
                len(positive & truth),
                len(positive - truth),
            )
    return WorldOutcomes(
        spec.world_id,
        counts,
        np.zeros((level_count, 2, 2), dtype=np.int64),
        np.asarray(levels, dtype=np.int32),
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
    )
