"""Frozen transfer identities, paired seeds and deferred case scheduling (D56).

No function here reads outcomes to select worlds, policies or settings.
"""

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from altiseek.calibration.model import digest
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec

METHODS = (("B5", 0), ("PPO", 17), ("PPO", 29), ("PPO", 43))
SENSOR_CASES = ("original", "refit")
# Controller errors (repr) that are isolated operational capture/flight failures.
OPERATIONAL_PREFIXES = (
    "ValueError('Actual camera missed policy goal by ",
    "ValueError('Image/pose mismatch ",
    "TimeoutError(",
)
# A stale-telemetry landing surfaces as either error, depending on whether the
# policy thread is waiting for a capture or starting a new move (v3).
STALE_TELEMETRY_ERRORS = (
    "RuntimeError('Flight ended before policy observation')",
    "RuntimeError('Flight is already landing')",
)


class OperationalFailure(RuntimeError):
    """A failed live case with verified cleanup and a recognised operational cause."""


def retry_record(attempt: int, reason: str, goal: Iterable, level: int, meter: dict | None) -> str:
    """One JSON line for a rejected capture; meter snapshots hold NumPy arrays (v4)."""
    record = {
        "attempt": attempt,
        "reason": reason,
        "goal": [float(v) for v in goal],
        "level": int(level),
        "meter": meter,
    }
    return json.dumps(record, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))


def paired_seed(world_id: str, purpose: str, training_seed: int = 0) -> int:
    return int(digest(["m10-v1", world_id, purpose, training_seed])[:8], 16) % (2**31)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if protocol.get("status") != "frozen" or protocol.get("protocol_id") != digest(body):
        raise ValueError("Transfer protocol is not frozen or its identity changed")
    for name, checksum in protocol["inputs"].items():
        if sha256_file(name) != checksum:
            raise ValueError(f"Frozen transfer input changed: {name}")
    return protocol


def authorize_world(protocol: dict, world: WorldSpec) -> None:
    if world.world_id != world.content_id():
        raise ValueError("Invalid world identity")
    if world.split == "B":
        if world.world_id != protocol["b_preflight_world_id"] or world.world_set != "rl-val":
            raise ValueError("World is outside frozen B preflight scope")
    elif world.split == "T":
        if world.world_id not in protocol["world_ids"] or world.world_set != "test":
            raise ValueError("World is outside frozen held-out transfer scope")
    else:
        raise ValueError("Unsupported transfer world split")


def sensor_path(protocol: dict, case: str) -> Path:
    if case not in SENSOR_CASES:
        raise ValueError("Unknown transfer sensor case")
    item = protocol["sensors"][case]
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Frozen transfer sensor checksum mismatch")
    return path


def live_cases(world_ids: Iterable[str]) -> list[dict]:
    """Every world x method x sensor case, in a fixed identity-shuffled order."""
    cases = [
        {"world_id": w, "method": method, "training_seed": seed, "sensor": sensor}
        for w in world_ids
        for method, seed in METHODS
        for sensor in SENSOR_CASES
    ]
    return sorted(cases, key=lambda c: digest(["m10-order-v1", *c.values()]))


def operational_failure(summary: dict, cleanup: dict, flight_summary: dict) -> bool:
    """Only recognised capture/flight failures with confirmed cleanup may be deferred."""
    reason = summary.get("error") or ""
    stale = (
        reason in STALE_TELEMETRY_ERRORS
        and flight_summary.get("reason") == "Telemetry became stale or invalid"
    )
    return bool(
        cleanup.get("owned_processes_stopped") is True
        and summary.get("passed") is False
        and (stale or reason.startswith(OPERATIONAL_PREFIXES))
    )


def run_deferred(
    cases: list[dict],
    attempt: Callable[[dict], bool],
    record_failure: Callable[[dict, OperationalFailure], None],
    failed_attempts: Counter,
    *,
    passes: int = 3,
    consecutive_limit: int = 3,
) -> None:
    """Run cases, deferring operational failures to the back of the queue.

    ``attempt`` returns True when it flew the case and False when a verified
    completion was reused. ``failed_attempts`` persists across invocations, so a
    relaunch cannot reset a case's attempt budget. Other errors propagate.
    """
    pending, consecutive = list(cases), 0
    for _ in range(passes):
        deferred = []
        for case in pending:
            key = digest(case)
            if failed_attempts[key] >= passes:
                raise RuntimeError(f"Case exhausted {passes} attempts; review required: {case}")
            try:
                flew = attempt(case)
            except OperationalFailure as error:
                failed_attempts[key] += 1
                record_failure(case, error)
                deferred.append(case)
                consecutive += 1
                if consecutive >= consecutive_limit:
                    raise RuntimeError(
                        f"{consecutive_limit} consecutive live failures; review required"
                    ) from None
                continue
            if flew:
                consecutive = 0
        pending = deferred
        if not pending:
            return
    raise RuntimeError("Retry passes exhausted; failed cases retained for review")
