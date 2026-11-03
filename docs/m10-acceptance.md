# M10 final validity review and disposition (D60, 2026-11-03)

This covers the held-out PX4/Gazebo transfer study under protocol v4
`5560516d08907341` (`configs/eval/m10_transfer_v4.json`,
`docs/m10-transfer-protocol.md`). All numbers come from the generated
`docs/m10-transfer-results/` (`results.md`, `results.json`, `episodes.csv`).
Intervals are 95% whole-world bootstrap intervals over 24 T worlds, with PPO
seeds averaged within each world. This is simulator-to-simulator transfer only.

## Integrity: passed

- **Completeness.** All 192 live episodes (24 worlds × B5 and PPO 17/29/43 ×
  original and refit sensor) and all 96 paired cached references are complete.
  The report runs only in strict mode, which verifies every completion's key,
  protocol ID and report checksum.
- **Frozen inputs.** Every runner command re-verified all 187 frozen inputs:
  sources, configs, both sensor models, three PPO checkpoints, detector, T world
  specs and caches, calibration review. A recheck after the run also passed.
- **Live checks.** Each live result passed identity and completion checks:
  protocol, world, method, seed, sensor case, paired seeds, measured 500 s
  energy, normal L5 edge launch, an ended episode, and confirmed simulator
  cleanup. No simulator process remains.
- **Failures.** Five attempts failed on telemetry-confirmed stale-telemetry
  landings and were deferred. Each was a distinct case, and each completed on
  its next attempt. All five are listed in the run's `failures.jsonl` and in
  the report. None was dropped.
- **Recaptures.** The camera-goal check rejected captures 3 times, in 3
  episodes, and every same-waypoint recapture was then accepted. Rejected
  frames never updated belief, and the retry time was charged to the battery.
- **B preflight.** It passed for all four policies with the refit sensor before
  any v4 T flight (`docs/m10-transfer-preflight.md`).
- **Superseded protocol versions.** Their records are kept, and none is reported
  as results:
  - v1: the preflight never exercised the policies.
  - v2: a failure-classifier gap stopped the run after 13 episodes (D57).
  - v3: a recapture-logging bug stopped it after 30 (D58).

  v2 outcomes were not examined. For v3, only the ending types were inspected.
  Each fix changed only operational code, and the worlds, policies, sensors,
  seeds and demo case stayed fixed across every version.
- **Analysis after outcomes.** The report script is outside the execution hash
  and did not change after the freeze. The demo script's belief overlay was
  made visible after T outcomes existed (D59); the change is cosmetic. The
  per-step timing analysis below is post-hoc and labelled as such.

## Findings

**H3a: deployment gap.**

| Policy | Cached AUC | Live AUC, original sensor | Live − cached [95% CI] |
|---|---|---|---|
| B5 | 0.899 | 0.891 | −0.008 [−0.069, +0.078] |
| PPO (3 seeds) | 0.731 | 0.597 | −0.135 [−0.199, −0.073] |
| PPO 17 / 29 / 43 | 0.713 / 0.748 / 0.734 | 0.574 / 0.615 / 0.601 | each interval excludes 0 |

- B5 transfers with no resolved AUC loss and found every target in all 24
  worlds live. PPO loses AUC under live execution for every seed, and its live
  first finds come about 62 s later.
- Two plausible contributors are present, but they are not isolated:
  1. **Detector shift on executed views.** Compared with cached variant 0 on
     the same lattice nodes, L5 recall is lower live: 0.360 (320/890) against
     0.528, difference interval [−0.283, −0.041]. L3/L4 are also lower, but
     their intervals include zero. This agrees with the D55 calibration
     finding.
  2. **Measured time (post-hoc).** A live step costs about 1.6–1.7× the cached
     nominal energy, because settling, inference and decisions run on the
     measured clock: B5 17.6 against 10.2 s per step, PPO 16.0 against 9.6 s.
     B5 needs a median of 4 steps and PPO 11–12, so the overhead accumulates
     more in PPO episodes.

**H3b: refit sensor.** Re-fitting on Gazebo B views did not recover the PPO
gap.
- PPO refit − original AUC is +0.027 [−0.036, +0.096]; the recovered fraction
  is 0.20 [−0.38, 0.70].
- For B5 the refit was slightly worse: −0.036 [−0.073, −0.010], with first
  finds 16.8 s later.
- The refit reduced PPO false declarations: −0.083 [−0.167, −0.014].
- PPO 17 found more targets with the refit (found fraction +0.163
  [0.042, 0.312]), but its AUC gain is not resolved.
- **H3b is not supported on the primary metric.** This is consistent with the
  D55 refit's known weaknesses: poor calibration, saturated L2/L3 temperatures,
  and a single view per location.

**RQ2 under deployment.** PPO stays below B5 in every condition: cached
−0.168, live original −0.295, live refit −0.232. All three intervals exclude 0.
The negative M8/M9 PPO findings carry through to deployment and widen.

## Limitations (carried into M11)

- **Sample size.** 24 worlds with one live action draw per policy, world and
  sensor; per-family results (3 worlds each) are descriptive only. Many paired
  comparisons were made, none corrected for multiplicity; AUC is the declared
  primary metric.
- **Simulation setup.**
  - Gazebo ground-truth pose, so no localisation is tested.
  - CPU llvmpipe rendering.
  - Detector, belief, policy and supervisor are co-located rather than run as
    separate ROS nodes (D52).
  - Stop-and-shoot with continuous measured accounting.
- **Scene coverage.**
  - T worlds contain no distractors.
  - No policy used L1 live, so there is no L1 detector comparison.
- **Inherited caveats.** The calibration limitations in
  `docs/m10-calibration-review.md` apply. Stale-telemetry landings (5 in about
  200 flights) show the simulator stack is not fully stable. Their cause is
  probably brief CPU-only simulator stalls, but it is unproven.
- **Demo.** It shows the predetermined case only (B5, 3 views, 38 s). It
  contains dataset-derived imagery and stays local:
  `data/m10/demo/5560516d08907341/30988c2109126434/demo.mp4`; its numeric
  manifest is `docs/m10-demo.json`.

## Disposition (D60)

M10 meets its §21 criteria:
- matched T scenarios ran end-to-end in PX4/Gazebo;
- the RQ3 comparison and the H3b refit are done;
- the demo video is recorded locally.

All integrity checks passed. **M10 is complete.** The results above, negative
ones included, are final for this protocol and are not to be tuned on T. Any
follow-up, such as more draws, a stall-tolerant controller or a separate
time-accounting ablation, needs a new protocol and B-only design choices. M11
(write-up) is next.
