"""D56 transfer adapter: retry rejected search captures at the same waypoint.

Rejected frames never update belief. The measured battery keeps running while
the vehicle re-settles, so every recapture is charged to the episode.
"""

import json
import sys
from pathlib import Path

import numpy as np
import search_flight_pilot
from retry_m10_capture import retry_capture

from altiseek.deploy.search import LiveView
from altiseek.deploy.transfer import retry_record
from altiseek.registry.run import sha256_file

FILES = ("search_recovery.py", "retry_m10_capture.py", "run_m10_transfer_flight.py")


class RecoverySearch(search_flight_pilot.SearchFlight):
    def sense(self, goal: np.ndarray, level: int) -> LiveView:
        capture = super().sense

        def rejected(attempt: int, reason: str) -> None:
            meter = self.flight_meter.snapshot() if self.flight_meter is not None else None
            with (self.output / "capture-retries.jsonl").open("a") as log:
                log.write(retry_record(attempt, reason, goal, level, meter) + "\n")
            print(f"Recapturing same waypoint after rejection: {reason}", flush=True)

        # The original sense recomputes PX4's target from the latest measured
        # world pose, clears settling, then applies the unchanged D52 checks.
        return retry_capture(lambda: capture(goal, level), rejected)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1])
    (output / "capture-recovery.json").write_text(
        json.dumps(
            {
                "decision": "D56",
                "max_capture_attempts": 3,
                "acceptance_checks": "unchanged D52",
                "sources_sha256": {
                    name: sha256_file(repo / "scripts/deploy" / name) for name in FILES
                },
            },
            indent=2,
        )
        + "\n"
    )
    search_flight_pilot.SearchFlight = RecoverySearch
    search_flight_pilot.main()


if __name__ == "__main__":
    main()
