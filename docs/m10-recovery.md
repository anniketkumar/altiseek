# M10 calibration recovery (D53)

The owner authorized skipping isolated failed flights and retrying unfinished
worlds after other work, on 2026-11-01. Acquisition settings, capture checks,
saved views and the D52 acquisition identity remain unchanged.

`scripts/continue_m10_resilient.py` is a separate scheduler. It waits on the
existing continuation lock, allowing the current scheduler and flight to finish
without interruption. If D52 completes, it exits without rerunning anything.
If D52 stops on a recognized capture-position or timeout error with confirmed
simulator cleanup, it moves that world to the back of the queue and continues.
Unknown failures and integrity failures require review.

Within the recovery scheduler, isolated flight failures are appended to
`failures.jsonl` and deferred. Each invocation allows three passes, stopping
early after three consecutive flight failures. Saved views are checked and
reused. No sensor fit is produced until all 40 worlds pass and all 8,800 views
are verified. Fit inputs are restored to their original frozen world order.

The D52 source files remain frozen. The new scheduler records its own source
hash and retry limits in `scheduler.json`, separately from acquisition identity.
Its detached launcher uses an additional recovery lock to prevent duplicate
waiting schedulers. Logs/PID are under `data/m10/continuation/`.

This change handles scheduling, not the cause of the observed 2.011 m camera
position error. That rejected capture remains excluded. No position threshold
is relaxed. Status and failure logs are local; no internet is required.
