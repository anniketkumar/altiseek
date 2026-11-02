# M10 B calibration review (D55, 2026-11-02)

Acquisition `cf148ad98784906d` (D52 contract `0e81d39c16e288b9`) finished at
08:33 Singapore time on 2 November 2026 and refitted B sensor model
**`caba50dfc568562d`** (`data/sensor_models/caba50dfc568562d.json`, SHA-256
`71c0046cd984ece448946420bc84145a0263f7d0d2bee657ed3adb18e4ec21f6`). This review
covers B data only. No T world, cache or outcome was read. Calibration
completion is not M10 completion.

`scripts/review_m10_calibration.py` repeats every check below from the saved
files and writes `docs/m10-calibration-review.json`. The run was made at commit
`dc707a1`, whose acquisition sources match the frozen D52 definition. Later
commits change those sources (the transfer work), so a rerun after them reports
the changes in `frozen_sources_changed`.

## Integrity: passed

- **Identity.** `definition.json` re-hashes to the run ID. The 40 sensor-fit B
  worlds (`5dc45d0e1ea5afbd`) keep their frozen order, and the detector
  checkpoint hash is unchanged. The contract belongs to this acquisition.
- **Views.** All 8,800 views load with matching key, contract and NPZ checksum:
  121/49/25/16/9 per world across L1–L5. Each view's world ID and level match
  its key.
- **Capture limits.** Every accepted view has image/pose delta 0.000 s
  (limit 0.04 s). Camera-to-goal error has mean 0.166 m, p95 0.304 m and
  max 0.966 m (limit 1 m).
- **Stages.** All 42 stage records verify: full-budget B5, the ten-view pilot
  and the 40 worlds. Each report checksum matches and passes, with
  `owned_processes_stopped` confirmed for its flight.
- **Refit.** An independent refit from the verified views reproduces model ID
  `caba50dfc568562d` and `fit.json` exactly, and the saved model loads under the
  B fit-set and deployment-contract checks.
- **Cleanup.** No Gazebo, PX4, DDS, scheduler or controller process remains.
  The final recovery log line is the completion record.

## Provenance and failure history (preserved)

- 48 flights contributed views. 6,944 views came from the original D52
  controller and 1,856 from the D54 adapter (9 flights). Views from different
  controller revisions are pooled in the fit.
- **The D54 same-waypoint correction was never exercised.** None of the
  adapter flights wrote a `capture-retries.jsonl`. Every world completed after
  D54 did so on first captures. The earlier position rejections therefore
  cannot be credited to D54 or shown to be fixed by it. The cause stays
  unresolved, as in `docs/m10-position-diagnosis.md`.
- `failures.jsonl` holds seven records: six failed flights, plus a second
  deferral record for one of them. The failures were one 0.060 s image/pose
  mismatch, camera-goal misses of 2.037, 1.189, 1.578 and 1.152 m, and one
  telemetry-confirmed early flight end. Every failed flight has confirmed
  cleanup. None of their rejected captures were saved. The 233 views those
  flights accepted before failing were retained under the atomic-view design
  (109, 14, 79, 6, 5 and 20 views).

## Quality: usable, with documented limitations

Cell-level rates at τ, Gazebo refit versus the original cached M6 fit on the
same 40 worlds. f is per 10⁴ empty cells. The trial sets differ: cached trials
use four render variants and the cached footprint; Gazebo trials use one live
view and whole visible cells. Treat the comparison as descriptive.

| Level | Views | Occupied trials | Gazebo r(τ) [95% CI] | Cached r(τ) | Gazebo f | Cached f | Gazebo soft r | Cached soft r | Temperature, Gazebo / cached | ECE after, Gazebo / cached |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 4,840 | 280 | 0.775 [0.691, 0.856] | 0.794 | 3.17 | 2.98 | 0.900 | 0.921 | 0.87 / 0.74 | 0.182 / 0.162 |
| L2 | 1,960 | 284 | 0.775 [0.720, 0.829] | 0.797 | 3.93 | 3.82 | 0.866 | 0.904 | 20.00 / 19.86 | 0.272 / 0.263 |
| L3 | 1,000 | 266 | 0.711 [0.637, 0.779] | 0.780 | 3.90 | 3.79 | 0.865 | 0.869 | 20.00 / 5.28 | 0.259 / 0.259 |
| L4 | 640 | 264 | 0.572 [0.477, 0.663] | 0.664 | 4.04 | 3.86 | 0.780 | 0.834 | 7.54 / 3.16 | 0.268 / 0.246 |
| L5 | 360 | 224 | 0.263 [0.192, 0.342] | 0.446 | 4.04 | 3.87 | 0.737 | 0.762 | 3.76 / 2.46 | 0.316 / 0.285 |

Pooled binary recall: Gazebo 0.634 [0.583, 0.685] against cached 0.709.

- **Coarse-level recall loss.** Gazebo recall falls with altitude faster than
  cached recall. At L5 the cached value lies far outside the Gazebo interval;
  at L3 and L4 it lies just outside. This is the expected sim-to-sim gap:
  Gazebo texture filtering (mipmapping) and the cached area-averaging renderer
  differ most at coarse GSD (§17). The cause is not isolated. False-alarm rates
  are similar.
- **Calibration stays poor.** After temperature scaling, ECE is 0.18–0.32 per
  level. At L2 and L3 the temperature sits at the configured upper bound of
  20.0, so the calibrator saturates, as the cached L2 fit nearly did.
  Thresholds and hyperparameters were not changed.
- **Not measured.** Conditional (second-look) recall has zero trials: D52
  collects one view per lattice location. The model keeps the 0.5 pseudocount
  value, which must not be reported as a measurement. The default first-look
  rule makes the model's update independent of it.
- Pose is Gazebo ground truth, and all views are B sensor-fit worlds.

## Disposition (D55)

Integrity is accepted. `caba50dfc568562d` is the frozen **refit** sensor for
the H3b arm of the transfer study, and `fcf8cdae58ca3947` stays the original
arm. Both are used unchanged. The quality limitations above are carried into
the transfer report and may limit how much of any gap the refit recovers. A
weak or negative H3b result will be reported as such. No threshold, level or
policy is adjusted in response to this review.
