"""D54 calibration adapter: retry rejected captures at the same waypoint."""

import json
from pathlib import Path

import calibration_flight
import numpy as np
from retry_m10_capture import retry_capture

from altiseek.deploy.search import LiveView
from altiseek.registry.run import sha256_file


class RecoveryCalibration(calibration_flight.CalibrationFlight):
    def sense(self, goal: np.ndarray, level: int) -> LiveView:
        capture = super().sense

        def rejected(attempt: int, reason: str) -> None:
            record = {"attempt": attempt, "reason": reason, "goal": goal.tolist(), "level": level}
            with (self.output / "capture-retries.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(f"Recapturing same waypoint after rejection: {reason}", flush=True)

        # The original sense recalculates PX4's target from current measured
        # world pose, clears settling, then applies the unchanged pose checks.
        return retry_capture(lambda: capture(goal, level), rejected)


def main() -> None:
    import sys

    repo = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1])
    files = ["calibration_recovery.py", "retry_m10_capture.py", "run_m10_capture.py"]
    (output / "capture-recovery.json").write_text(
        json.dumps(
            {
                "decision": "D54",
                "max_capture_attempts": 3,
                "acceptance_checks": "unchanged D52",
                "sources_sha256": {
                    name: sha256_file(repo / "scripts/deploy" / name) for name in files
                },
            },
            indent=2,
        )
        + "\n"
    )
    calibration_flight.CalibrationFlight = RecoveryCalibration
    calibration_flight.main()


if __name__ == "__main__":
    main()
