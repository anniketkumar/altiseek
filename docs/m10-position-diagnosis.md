# Position rejection diagnosis and bounded correction (D54)

On 2026-11-02 the owner requested diagnosis and continuation after three
consecutive failed calibration flights. The failures were 1.189, 1.578 and
1.152 m actual camera-to-goal errors. The last flight's PX4 estimated-position
hold error stayed below 0.291 m. Thus satisfying the estimated-position settle
condition does not ensure the camera reaches the requested world position.

The original controller converts a world goal into a PX4 local setpoint once,
using the latest PX4 and Gazebo positions at the start of each move. Those two
streams are not synchronized for this conversion, and the estimated-to-world
offset can change during movement. The saved accepted views show vertical
offsets around 0.57–0.67 m in the affected flights. This identifies a weakness
in control feedback; the logs do not uniquely establish whether estimator
drift, stream timing or both caused each rejected capture. No inference about
target appearance or detector performance is used to change the controller.

`calibration_recovery.py` calls the original capture function again at the same
goal after a position or image/pose timing rejection. That function recomputes
the local setpoint from the latest pose and resets settling before capture.
There are at most three attempts per waypoint; exhausted attempts still fail
the flight and are deferred by the D53 scheduler. Unknown/integrity errors are
not retried. This is bounded correction, not a relaxed acceptance threshold.

All existing valid views remain unchanged. The 1 m camera-goal bound, 0.04 s
image/pose bound, visibility calculation, rendering, detector and sensor-fit
world set remain unchanged. Rejected captures never enter the calibration
collection. Original acquisition sources stay frozen; the separate adapter and
its source hashes are recorded in each flight's `capture-recovery.json`, and
attempts in `capture-retries.jsonl`. The original run identity describes the
unchanged observation contract; D54 separately records altered execution.

`run_m10_capture.py` reuses the original simulator lifecycle and cleanup. Its
only dispatch change is selecting the calibration adapter for calibration
flights. Completed stages are not re-flown. All 40 completed worlds and 8,800
validated views are still required for refitting. Report D54 execution alongside
D52/D53 provenance; do not claim every view used the same controller revision.
