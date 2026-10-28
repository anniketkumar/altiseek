# Interim M8 AUC sanity audit

This records the first two completed D41 runs, not acceptance of M8 or a change
to its protocol. V3 seed 43 and the V0 runs are still pending at this review.
Training continues under the frozen six-run protocol; M9 remains blocked.

## Saved results

| V3 seed | Run ID | Selected step | Full p AUC | Zero-p AUC | Zero-p minus B2, 95% CI | Full p minus zero-p, 95% CI |
|---|---|---:|---:|---:|---|---|
| 17 | `985e4600240c36a8` | 65,536 | 0.713 | 0.658 | -0.021 [-0.109, +0.068] | +0.055 [+0.016, +0.096] |
| 29 | `24c1acc2a1dafcaf` | 98,304 | 0.719 | 0.704 | +0.025 [-0.089, +0.143] | +0.015 [-0.036, +0.074] |

Each checkpoint was selected by mean family-normalised AUC, not raw mean AUC.
Three sampled draws are averaged within each of the 40 validation worlds before
paired bootstrap (2,000 resamples, seed 20261028). These are B selection results;
intervals do not adjust for checkpoint selection, shared imagery or multiple checks.
Both runs have finite optimizer losses. The last three sampled AUCs span 0.007
and 0.006 respectively, but that alone does not prove convergence.

The frozen B2 V3 mean is 0.679. Seed 29 therefore triggers the existing point-mean
sanity gate, even though its interval crosses zero. **The gate remains failed for
that run.** Statistical uncertainty is not used to silently replace the rule.
Seed 17 shows a positive paired p-channel benefit under this evaluation; seed 29's
benefit is uncertain. Neither result establishes generalization to T.

## Input and implementation audit

- `SearchFeatures.forward` zeros only p, after log preprocessing. It leaves q,
  all five coverage masks, resolved cells, footprint, position and coordinates.
  The vector still contains level, time, battery, confirmed found fraction and K.
- Resolved cells and confirmed count are allowed feedback under D36. Removing p
  does not remove every source of sensor/verification feedback. Coverage is marked
  over the observed region regardless of detections; it is geometric history.
- B2 follows a selected fixed-level prior sweep. It is a useful reference, not a
  mathematical upper bound on every policy with the remaining public inputs.
  Consequently, beating B2 with p zeroed is not by itself proof of hidden-state leakage.
- Normal training/evaluation constructs `SearchEnv` with `oracle=False`. The policy
  receives only the observation dictionary and geometric action mask. Truth-bearing
  episode information is used for logging/termination, not passed into the network.
- The observation regression now changes both hidden target coordinates and cell
  indices while holding public state and K fixed. Observations and action masks
  remain identical. The CNN test confirms p removal and retained q sensitivity.
  All 14 targeted environment/CNN/style tests passed. This is a bounded audit of
  these access paths, not proof against every possible form of leakage or overfitting.

This explains why the diagnostic needs careful interpretation; it does not prove
which remaining channel caused seed 29's excess. Do not change the reward, model,
selected checkpoints or acceptance criterion in response to this interim result.
Review all six results and resolve the sanity gate explicitly before accepting M8.

## Reproduction

`uv run python scripts/report_ppo.py runs/ppo/m8-auc --config configs/policy/ppo_auc.yaml --partial`

This reads the preserved `complete.json`, selected `eval-*.json`, `zero-belief.json`
and original baseline episodes, checks identities/hashes, and regenerates paired
intervals in `runs/ppo/m8-auc/progress-report.md`/`.json`. The final automatic report
will contain the same paired comparisons for all six runs. This document remains
an explicitly interim audit rather than silently changing as training finishes.
