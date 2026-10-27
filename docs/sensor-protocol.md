# M6 sensor fitting and evaluation protocol

Frozen before M6 test outcomes are read (2026-10-27). This is the sensor evaluation,
not the M9 policy protocol. All choices below are fixed using the specification,
owner decisions and sensor-fit B only. Do not select models or settings on T.

- Fit all 40 sensor-fit worlds (`5dc45d0e1ea5afbd`) with the accepted detector,
  geometry, cache settings and original operating threshold. Verify cache headers
  and SHA-256 metadata before use. Retain all views and four variants.
- Centre-cell rule, chosen by the owner: occupancy means at least one world target
  centre in the cell. A detection belongs only to its centre cell. An observed cell
  has its centre in the actual shifted footprint. Full world truth determines
  occupancy even when a boundary target is outside the shifted view. Multiple
  targets in one cell count once; multiple swimmer detections reduce to their
  maximum score. Other predicted classes are ignored.
- Calibrate the **maximum-score detected-cell outcome**, with cell occupancy as its
  TP/FP label, at raw score >= 0.05. This is the random variable consumed by V3;
  it is not IoU precision or per-box one-to-one matching. Temperature is fitted
  independently per level by BCE, constrained to [0.05, 20]. Report a bound hit.
- Estimate binary r/f at tau=0.37650254368782043 and soft r/f at tau_min=0.05.
  Point estimates and 95% percentile intervals use raw counts; bootstrap whole
  worlds (2,000 draws, seed 20261027). Runtime likelihoods use Jeffreys smoothing
  (0.5 successes and 0.5 failures), including the global, exposure-pooled V1.
- V3 conditional score masses use ten equal-width bins over calibrated [0,1],
  separately for occupied and empty detected cells, with 0.5 added to each bin.
  Scores are conditional on raw score >= tau_min; do not threshold calibrated
  scores again. Fifteen equal-width bins define ECE/reliability (detected cells
  only). Empty evaluation bins are null, not zero accuracy.
- Measure second-look recall for every ordered distinct variant pair at the same
  (world, level, lattice view), using occupied cells observed in both looks. Report
  at both thresholds with whole-world intervals. Keep first observation per
  (cell,level) as the default until this B-only evidence is reviewed and D6 settled.
  Pair measurements do not establish independence across overlapping viewpoints.
- Freeze the fitted model before evaluating all 80 test worlds
  (`22750c80717314c3`). Report per-level rates and calibration before/after; do not
  re-fit on test. These worlds have no distractors and only ten worlds per family.
- Real T: repeat the existing M4 level-evaluation view selection, using only its
  split/image/GSD/window columns (never outcome counts). Reconstruct exact windows
  from tile geometry; the saved coordinates are rounded. Run the frozen detector
  on CPU at tau_min with the same NMS settings. Use a 2.5 m grid anchored at (0,0)
  in each source camera's ground frame, not at the view corner. Only cells whose
  centres fall on valid source pixels are observed. Match target/detection centres
  by cell and reduce scores as above. Report reliability/ECE and raw cell r/f;
  bootstrap frames. Save per-view outcomes atomically so this CPU job resumes.
- Re-check D7 using B cell recall, false alarms per 10,000 empty observations,
  and the existing coverage-energy calculation. A centre-cell metric is stricter
  than D24's 1 m object match; do not silently equate them or redesign from T.
- Model identity covers detector, fit world set, full fit config, cache contract,
  fit cache IDs and fitted numerical parameters. Model loaders validate content
  and refuse detector, fit-set or supplied cache-contract mismatch.

No imagery or caches are published. Numeric reports and plots may be committed.
