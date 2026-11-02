# M10 transfer protocol (D56) — fixed before any new T outcome

This protocol was written, implemented and frozen before any T world was flown or
any T cached reference was computed for M10. `scripts/freeze_m10_transfer.py`
hashes this document, every source and config file, both sensor models, the
three PPO checkpoints, the detector, the selected T world specs and caches, and
the D55 calibration review into `configs/eval/m10_transfer_v1.json`. Every runner
command re-verifies those hashes; any change stops the study.

The report and demo scripts do not affect any episode. They are committed and
tested before T execution but kept out of that execution hash, so a defect can
be fixed without invalidating flights. Each output records its script hash.
Any change to them after T outcomes exist needs a Decision Log entry and a
diff review, and must not alter the comparisons fixed below.

## Scope and selection

- **Worlds.** 24 existing T test worlds (`22750c80717314c3`): three from each of
  the eight original families, ranked within family by the hash of
  `["m10-transfer-selection-v1", world_id]`. Selection reads world metadata
  only. T stays distractor-free; the low/high distractor labels are not
  density contrasts (D43).
- **Policies.** V3 B5 with its frozen B-selected family parameters, and all three
  frozen D41 V3 PPO checkpoints (seeds 17/29/43), sampling actions as in M9. No
  retraining, tuning or checkpoint selection.
- **Sensors.** Original cached-fit `fcf8cdae58ca3947` and Gazebo B refit
  `caba50dfc568562d` (D55), each used unchanged with V3 updates.
- **Episodes.** 24 worlds × 4 policies × 2 sensors = **192 live episodes**, plus
  **96 cached references** with the original sensor (the cached detections were
  fitted by it). One paired environment seed per world and one action seed per
  policy and world, shared by live and cached runs. Live cases run in a fixed
  identity-shuffled order to spread time-of-run effects across conditions.

## Live execution

D52 applies unchanged: measured 500 s energy, normal seeded L5 edge launch,
whole-cell visibility, Gazebo ground-truth pose, the same settling and capture
limits (0.5 m PX4 hold, 0.3 m/s, 1°, 2°, 2 s dwell; image/pose ≤ 0.04 s; camera
goal ≤ 1 m). Episodes run to termination, battery exhaustion or the step cap.

**Same-waypoint recapture.** A capture rejected for camera-goal error or
image/pose mismatch is retried at the same waypoint, at most three attempts. Each
retry recomputes the PX4 setpoint from the latest pose and settles again, as
D54 does for calibration (`scripts/search_recovery.py`). The rejected frame
never updates belief. The measured battery keeps running throughout, so the
retry's time is charged to the episode. Every recapture is logged per flight
and reported.

**Deferred episodes.** An episode that still fails, for those reasons or a
timeout or telemetry-confirmed stale telemetry, with confirmed process
cleanup, is logged to `failures.jsonl`. Its flight directory is kept, the
case moves to the back of the queue, and it is later flown again from the
start with the same seeds. A case may be attempted three times. Three
consecutive failures, an exhausted case, an integrity or identity mismatch or
any unrecognised error stop the study for review. Relaunching does not reset
the per-case count. The reported result is the first complete valid attempt;
failed attempts are listed in the report. No case is dropped.

## Preflight before T

The full regression suite must pass. Then one complete B episode per policy
(B5 and each PPO seed) is flown with the refit sensor on rl-val world
`003b0c54f581c65b`, under the same rules. The runner stops there for review
(`--stop-after-preflight`), and the measured runtimes set the ETA. The original
all-seed pilots and the full-budget B check stay prerequisites. Preflight is B
integration, not a held-out result.

## Reporting

`scripts/report_m10_transfer.py` refuses to run unless all 192 live and 96 cached
results verify. It writes `docs/m10-transfer-results/`:

- AUC (primary), found fraction, success, first-find time (censored at 500 s),
  false declarations, energy, simulator time and path, per policy and condition.
  Live wall time and recaptures are reported separately.
- Paired whole-world differences, 2,000 bootstrap samples, seed 20261102:
  live-original − cached (H3a), live-refit − cached, live-refit − live-original
  (H3b), and PPO − B5 in each condition. PPO seeds are averaged within world;
  every seed is also reported alone.
- The fraction of the AUC gap recovered by the refit. It is reported as
  undefined when the gap is not positive or its interval includes zero.
- Per-family means, descriptive only (three worlds each).
- Level-wise detector outcomes on executed live views against cached variant 0
  at the same lattice node. Cells must be fully visible live and inside the
  nominal footprint. Repeated nodes are clustered by world, not independent.

The aggregate gap does not isolate rendering, pose, visibility or timing
effects. Negative results are reported as found.

## Demo

The demo case is fixed in the protocol: the first selected world, B5, original
sensor, regardless of its result. `scripts/demo_m10_transfer.py` writes a local
MP4. It shows the top-down exported world with the belief overlay, the path
coloured by level, the current footprint and the true targets, plus the live
camera inset with swimmer detections. It contains dataset imagery and stays in
`data/` (gitignored); only a numeric manifest, `docs/m10-demo.json`, is
committed.

## Acceptance

M10 is complete only after every case is complete, all integrity checks pass,
the report and demo are produced and a final validity review is recorded.
