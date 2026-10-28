# M8 acceptance review

## Current verdict

All six D41 training runs, checkpoint selection and reporting are complete.
M8 is **not yet accepted under the unchanged criteria**: the zero-p comparison
fails for V3 seed 29, and stable learning across V0 seeds is not demonstrated.
This is distinct from PPO losing to strong baselines, which is a valid result.
No further training is currently queued. M9 has not started.

## Evidence against the criteria

| Requirement | Evidence | Review |
|---|---|---|
| At least three seeds per configuration | V0/V3, seeds 17/29/43, 131,072 transitions each | Complete |
| Select checkpoints on rl-val | Predeclared sampled-action family-normalised AUC; report verifies selection and checkpoint hashes | Complete |
| Reproducible results and behavior analysis | Numeric results, all-seed curves, altitude use, first finds, repeated visits, no-information views and comparison with B3 | Complete |
| Numerical training health | Optimizer losses finite for all six runs; no logged invalid-action failures | Passed, but finiteness does not imply learning |
| Stable learning curves | V3 last-three AUC ranges 0.007/0.006/0.019; V0 0.060/0.072/0.125 | V3 supports practical stability within this budget; V0 does not establish stable convergence |
| Zero-p policy does no better than B2 | Five runs pass; V3 seed 29 exceeds B2 by 0.025 AUC | Literal gate fails; investigated, not waived |

The complete report is [ppo-auc-results.md](ppo-auc-results.md), with numeric
results in [ppo-auc-results.json](ppo-auc-results.json) and the regenerated plot
in [m8_auc_learning.png](figures/m8_auc_learning.png). Full trajectories and
checkpoints remain under `runs/ppo/m8-auc/`, outside git. The completed initial
suite and D40 diagnosis remain separate and unchanged.

| Variant | Selected AUC, mean ± training-seed SD | B3 | B5 |
|---|---:|---:|---:|
| V0 | 0.212 ± 0.066 | 0.457 | 0.551 |
| V3 | 0.720 ± 0.007 | 0.868 | 0.888 |

These are validation selection scores, not held-out test results. V3 improved
over the initial configuration, but the D41 reward, discount and action-rule
changes were bundled; this does not isolate their individual causal effects.
V0 remains a poor, variable learned policy under this budget. Do not infer that
every possible V0 policy must fail, or that the sensor variant alone explains
the learned-policy gap. Frozen baselines outperform PPO on mean validation AUC.

## Zero-p issue

The seed-29 zero-p excess is +0.025, paired whole-world 95% CI [-0.089, +0.143].
Crossing zero does not erase the recorded point-mean gate failure. As detailed
in [the interim audit](ppo-auc-audit.md), p removal retains prior, coverage,
resolved cells, confirmed found count and navigation state. B2 is a fixed-level
heuristic, not an upper bound for all policies with those public inputs.

Source inspection and the strengthened observation/mask regression found no
direct unfound-location path into policy inputs. They do not prove absence of
all leakage, memorization or other implementation errors. The exact contribution
of each remaining channel to seed 29's score has not been isolated.

Full-p minus zero-p AUC for V3 is +0.055/+0.015/+0.083; respective paired CIs are
[+0.016, +0.096], [-0.036, +0.074], [+0.023, +0.145]. Thus the p channel helps two
seeds under this evaluation, while seed 29's benefit is uncertain. These intervals
are selection-set diagnostics, not confirmatory hypothesis tests.

## Proposed disposition — awaiting owner decision

Recommend closing M8 as a **completed, budget-limited experiment with a negative
PPO-versus-baseline result**, using a separately recorded D42 amendment:

1. Preserve every score, checkpoint and failed diagnostic. Freeze the current
   selected policies; do not train until a preferred conclusion appears.
2. Explicitly accept V0's failure to establish stable convergence within the
   declared budget as a limitation. Do not label all configurations stable.
3. Replace the inference that zero-p superiority over B2 necessarily proves
   leakage with a documented input-access audit and paired diagnostic comparison.
   A detected prohibited input path or artifact/split mismatch remains blocking.
   Keep the original gate and its failed outcome in the historical report.
4. Leave the D41 training/evaluation protocol and results untouched. Amend only
   the milestone disposition and future interpretation of the sanity check,
   explicitly acknowledging that this decision follows inspection of B results.
5. Resolve the separate test-world distractor decision and freeze M9's evaluation
   protocol before any main test evaluation. No test results informed this review.

This proposal is **not applied** by this document. The alternative is to keep M8
open and predeclare a further B-only optimization experiment to address V0
stability. That would require another training budget and would not guarantee
better results. Blindly extending the current runs is not recommended.

## Verification record

The revision previously passed the full 280-test suite; the subsequently added
sampled-evaluation replay/RNG test and style checks passed separately. The later
input-isolation audit passed all 14 targeted environment/CNN/style checks. The
1,024-step real-cache smoke run completed, followed by all six production runs.
Final reporting verified consistent sources/inputs, checkpoint hashes, selection
rules and paired validation-world/repeat identities. The learning figure was
visually reviewed.

A fresh `uv run pytest` at final review did not start: Windows Application Control
blocked the pytest executable (OS error 4551). No new test result is claimed for
that attempt, and no alternative execution path was used to bypass the policy.
The final review changes only documentation and generated numeric results/plots.
