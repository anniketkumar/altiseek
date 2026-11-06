"""D63 B-only timing analysis; no flight, inference, policy or schema changes.

Positions are commanded ENU metres; primary durations are simulator seconds
between post-inference meter snapshots. Calibration uses capture timestamps and
is deliberately a separate diagnostic. See the predeclared M12 energy protocol.
"""

import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from altiseek.calibration.model import digest
from altiseek.env.energy import EnergyModel
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

NATIVE_ROOT = PurePosixPath("/home/aniket/altiseek-m10")
ACQUISITION = "cf148ad98784906d"
CONTRACT = "0e81d39c16e288b9"
TRANSFER = "5560516d08907341"
BOOTSTRAP_SEED = 20261106
BOOTSTRAP_DRAWS = 10_000
B_SETS = {"sensor-fit": "5dc45d0e1ea5afbd", "rl-val": "40f07c646a97efba"}


@dataclass(frozen=True)
class FlightSource:
    flight: str
    world: str
    method: str
    seed: int
    manifest: str
    sha256: str


SOURCES = (
    FlightSource(
        "20261031T043133Z",
        "003b0c54f581c65b",
        "B5",
        0,
        f"continuations/{ACQUISITION}/full-budget-B5.json",
        "9b0fbdc0aca9f856480c449071f99e6993c08017ec65f6d0489e35385b9fe9ca",
    ),
    FlightSource(
        "20261102T090615Z",
        "48a9325328a6d2c4",
        "B5",
        0,
        f"transfers/{TRANSFER}/preflight/571e79a898ec1412.json",
        "45e44581ef38524c8bd187d26570bac6f79679da8754a5264a75b617e625eb69",
    ),
    FlightSource(
        "20261102T090914Z",
        "48a9325328a6d2c4",
        "PPO",
        17,
        f"transfers/{TRANSFER}/preflight/9bb74a29561c4edc.json",
        "fff15dbf6ccce7d166aa8260fbbdb507797739cbe859a335c88aaaa1bc6c5b0c",
    ),
    FlightSource(
        "20261102T091144Z",
        "48a9325328a6d2c4",
        "PPO",
        29,
        f"transfers/{TRANSFER}/preflight/8655d0f0800235ea.json",
        "07615968f3c1e1e2a7d197af36578f17a3c939f5c7b6fa0798485683e17ce97c",
    ),
    FlightSource(
        "20261102T091411Z",
        "48a9325328a6d2c4",
        "PPO",
        43,
        f"transfers/{TRANSFER}/preflight/c5550a88e5516884.json",
        "bc33fcde522c9fa5d9cdc57ef1ef0580f50c4f1decf42a2acd8aeed29d35d188",
    ),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def native_path(root: Path, recorded: str) -> Path:
    """Map a recorded native path without allowing traversal or another source root."""
    path = PurePosixPath(recorded)
    if ".." in path.parts:
        raise ValueError("Native path traversal")
    return root.joinpath(*path.relative_to(NATIVE_ROOT).parts)


def b_worlds(repo: Path, name: str) -> list[WorldSpec]:
    """Only the two predeclared B sets can be opened, never a test-set directory."""
    if name not in B_SETS:
        raise ValueError("Only declared B sets are allowed")
    root = repo / "data/worlds" / name
    meta = read_json(root / "world_set.meta.json")
    if (
        meta["split"] != "B"
        or meta["world_set_id"] != B_SETS[name]
        or sha256_file(root / "worlds.csv")[:16] != B_SETS[name]
    ):
        raise ValueError("B world-set identity mismatch")
    with (root / "worlds.csv").open(encoding="utf-8", newline="") as stream:
        ids = [r["world_id"] for r in csv.DictReader(stream)]
    if len(ids) != meta["worlds"] or len(set(ids)) != len(ids):
        raise ValueError("World-set count mismatch")
    worlds = []
    for world_id in ids:
        if len(world_id) != 16 or any(c not in "0123456789abcdef" for c in world_id):
            raise ValueError("Invalid world ID")
        world = WorldSpec.load(root / f"{world_id}.json")
        if (
            world.world_id != world_id
            or world.split != "B"
            or world.world_set != name
            or world.library_id != "6b541bb0ff9d062a"
        ):
            raise ValueError("Expected an original B world")
        worlds.append(world)
    return worlds


def verified_search(root: Path, source: FlightSource) -> tuple[dict, dict]:
    """Validate a pinned B report and its completion/cleanup records before analysis."""
    report_path = root / "camera-flights" / source.flight / "search-summary.json"
    manifest = read_json(root / source.manifest)
    recorded = manifest.get("report", manifest.get("report_path", ""))
    if (
        native_path(root, recorded) != report_path
        or manifest["sha256"] != source.sha256
        or sha256_file(report_path) != source.sha256
    ):
        raise ValueError("Pinned search report checksum/path mismatch")
    if source.flight == SOURCES[0].flight:
        if manifest["run_id"] != ACQUISITION:
            raise ValueError("Integration acquisition mismatch")
    else:
        expected = {
            "world_id": source.world,
            "method": source.method,
            "training_seed": source.seed,
            "sensor": "refit",
            "smoke": True,
        }
        if manifest["protocol_id"] != TRANSFER or manifest["key"] != expected:
            raise ValueError("B preflight identity mismatch")
    report = read_json(report_path)
    if (
        not report["passed"]
        or report["world_id"] != source.world
        or report["method"] != source.method
        or (report["training_seed"] or 0) != source.seed
        or not report["measured_energy"]
        or not report["normal_edge_launch"]
        or report["battery_s"] != 500
        or report["view_cap"] != 0
        or report["error"]
        or not report["search"][-1]["ended"]
    ):
        raise ValueError("Invalid completed B measured-search report")
    if source.flight != SOURCES[0].flight and (
        report["protocol_id"] != TRANSFER or report["sensor_case"] != "refit"
    ):
        raise ValueError("Unexpected preflight protocol/sensor")
    cleanup_path = report_path.with_name("status.json")
    cleanup = read_json(cleanup_path)
    if cleanup["state"] != "passed" or not cleanup["owned_processes_stopped"]:
        raise ValueError("Search cleanup unconfirmed")
    provenance = {
        "flight": source.flight,
        "world_id": source.world,
        "method": source.method,
        "training_seed": source.seed,
        "report_sha256": source.sha256,
        "manifest_sha256": sha256_file(root / source.manifest),
        "cleanup_sha256": sha256_file(cleanup_path),
    }
    return report, provenance


def finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite timing/position")
    return number


def residual(previous: dict, current: dict, energy: EnergyModel, *, measured: bool) -> dict:
    """One completed interval in seconds; never use wall time as simulator time."""
    before = np.asarray(previous["goal_enu_m"], dtype=float)
    after = np.asarray(current["goal_enu_m"], dtype=float)
    if before.shape != (3,) or after.shape != (3,) or not np.isfinite([before, after]).all():
        raise ValueError("Invalid ENU goal")
    dx, dy, dz = (float(v) for v in after - before)
    nominal_time, nominal_energy = energy.move(dx, dy, dz)
    dwell_time, dwell_energy = energy.observe()
    if measured:
        for view in (previous, current):
            if not view["measured"]["started"] or view["measured"]["exhausted"]:
                raise ValueError("Accepted view has invalid meter state")
        elapsed = finite(current["measured"]["elapsed_s"]) - finite(
            previous["measured"]["elapsed_s"]
        )
    else:
        if previous["flight_run"] != current["flight_run"]:
            raise ValueError("Cannot bridge calibration flights")
        elapsed = finite(current["stamp_s"]) - finite(previous["stamp_s"])
    if elapsed <= 0:
        raise ValueError("Non-monotone simulator time")
    row = {
        "from_level": int(previous["level"]),
        "level": int(current["level"]),
        "horizontal_m": math.hypot(dx, dy),
        "vertical_m": dz,
        "motion": "climb" if dz > 0 else "descend" if dz < 0 else "same-level",
        "elapsed_s": elapsed,
        "nominal_s": nominal_time + dwell_time,
        "overhead_s": elapsed - nominal_time - dwell_time,
    }
    if measured:
        used = finite(current["measured"]["energy_s"]) - finite(previous["measured"]["energy_s"])
        if used < 0:
            raise ValueError("Non-monotone measured energy")
        row["energy_residual_s"] = used - nominal_energy - dwell_energy
    return row


def median_fit(flights: list[list[float]]) -> dict:
    """Equal-flight median and the predeclared whole-flight percentile bootstrap."""
    if len(flights) != 5 or any(not values for values in flights):
        raise ValueError("D63 requires five nonempty flights")
    medians = np.array([np.median([finite(v) for v in values]) for values in flights])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(5, size=(BOOTSTRAP_DRAWS, 5))
    return {
        "overhead_s": float(np.median(medians)),
        "ci95_s": np.quantile(np.median(medians[draws], axis=1), [0.025, 0.975]).tolist(),
        "flight_medians_s": medians.tolist(),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "flight",
    }


def describe(rows: list[dict]) -> dict:
    values = [r["overhead_s"] for r in rows]
    return {
        "transitions": len(rows),
        "median_overhead_s": float(np.median(values)),
        "min_overhead_s": min(values),
        "max_overhead_s": max(values),
    }


def strata(rows: list[dict]) -> list[dict]:
    groups: defaultdict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[(row["level"], row["motion"], round(row["horizontal_m"], 6))].append(row)
    return [
        {"level": level, "motion": motion, "horizontal_m": distance, **describe(group)}
        for (level, motion, distance), group in sorted(groups.items())
    ]


def calibration_records(root: Path, repo: Path, keys: list[tuple]) -> tuple[list[dict], dict]:
    """Verify accepted B timing metadata; hash NPZ bytes without decoding detections."""
    acquisition = root / "continuations" / ACQUISITION
    definition = read_json(acquisition / "definition.json")
    contract = read_json(acquisition / "contract.json")
    if (
        digest(definition) != ACQUISITION
        or digest(contract) != CONTRACT
        or contract["acquisition_id"] != ACQUISITION
    ):
        raise ValueError("Calibration identity mismatch")
    worlds = b_worlds(repo, "sensor-fit")
    if [w.world_id for w in worlds] != definition["world_ids"] or len(worlds) != 40:
        raise ValueError("Calibration B world order mismatch")
    review_path = repo / "docs/m10-deployment/m10-calibration-review.json"
    review = read_json(review_path)
    if review["run_id"] != ACQUISITION or review["contract_id"] != CONTRACT:
        raise ValueError("Wrong calibration review")
    records, hashes = [], []
    per_flight: defaultdict[str, int] = defaultdict(int)
    for world in worlds:
        directory = acquisition / "views" / world.world_id
        for key in keys:
            path = directory / ("-".join(map(str, key)) + ".json")
            row = read_json(path)
            if (
                row["world_id"] != world.world_id
                or row["key"] != list(key)
                or row["level"] != key[0]
                or row["contract_id"] != CONTRACT
                or row["sha256"] != sha256_file(path.with_suffix(".npz"))
                or not 0 <= finite(row["pose_delta_s"]) <= 0.04
                or not 0 <= finite(row["goal_error_m"]) <= 1
            ):
                raise ValueError("Calibration view identity/checksum/capture mismatch")
            flight = row["flight_run"]
            if (
                flight not in review["flights"]
                or world.world_id not in review["flights"][flight]["worlds"]
            ):
                raise ValueError("Unknown B calibration flight")
            records.append(row)
            hashes.append([world.world_id, list(key), sha256_file(path), row["sha256"]])
            per_flight[flight] += 1
    if len(records) != 8800 or set(per_flight) != set(review["flights"]):
        raise ValueError("Calibration view/flight count mismatch")
    flights = []
    for flight, count in sorted(per_flight.items()):
        if count != review["flights"][flight]["views"]:
            raise ValueError("Calibration flight view count mismatch")
        directory = native_path(root, flight)
        status = read_json(directory / "status.json")
        if not status["owned_processes_stopped"]:
            raise ValueError("Calibration cleanup unconfirmed")
        flights.append(
            {
                "flight": directory.name,
                "views": count,
                "controller": review["flights"][flight]["controller"],
                "failed": flight in review["failed_flights"],
                "cleanup_sha256": sha256_file(directory / "status.json"),
            }
        )
    return records, {
        "acquisition_id": ACQUISITION,
        "contract_id": CONTRACT,
        "definition_sha256": sha256_file(acquisition / "definition.json"),
        "contract_sha256": sha256_file(acquisition / "contract.json"),
        "review_sha256": sha256_file(review_path),
        "view_records_id": digest(hashes),
        "worlds": len(worlds),
        "views": len(records),
        "flights": flights,
    }


def calibration_intervals(records: list[dict], energy: EnergyModel) -> list[dict]:
    grouped: defaultdict[str, list] = defaultdict(list)
    for record in records:
        finite(record["stamp_s"])
        grouped[record["flight_run"]].append(record)
    rows = []
    for flight, views in sorted(grouped.items()):
        views.sort(key=lambda r: r["stamp_s"])
        for previous, current in itertools.pairwise(views):
            if current["world_id"] != previous["world_id"]:
                raise ValueError("Calibration flight mixes worlds")
            rows.append(
                {
                    "flight": PurePosixPath(flight).name,
                    "world_id": current["world_id"],
                    "from_key": previous["key"],
                    "to_key": current["key"],
                    **residual(previous, current, energy, measured=False),
                }
            )
    return rows
