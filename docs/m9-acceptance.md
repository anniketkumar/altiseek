# M9 existing-cache phase: accepted

Accepted on 2026-10-29 under the owner's instruction to accept the results unless
a serious validity problem is found. The review found no blocking validity issue.
This accepts the completed D44 phase, not experiments that have not been run.

Run `341aadf1366d8f05` used clean committed code `39903c4`. All **25,920 episodes**
completed in **4,091.6 s (68.2 min)**. The automatic report verified every expected
episode's identity and checksum before generating tables and figures. All six
selected checkpoints and frozen baseline settings were preserved. Nothing was
retrained, reselected or tuned using T.

Preflight passed **290 tests** and **59/59 B smoke episodes**. Acceptance review
checked all 900 aggregate rows: metric ranges, four actual T family sizes, all
three PPO seeds, energy budgets, monotone found curves and agreement between
curve endpoints and found-fraction metrics. The nominal oracle exceeds random
for every reported family/variant. Both generated figures were visually reviewed.

## Main findings

| Nominal setting | PPO AUC (seed SD) | B5 AUC | Paired PPO minus B5, 95% CI |
|---|---:|---:|---|
| V0 | 0.242 (0.055) | 0.677 | -0.434 [-0.489, -0.377] |
| V3 | 0.744 (0.014) | 0.903 | -0.160 [-0.198, -0.121] |

V3 PPO finds 89.0% of targets versus B5's 97.5%, averaged across worlds and
action draws. The learned policy remains weaker than the strong heuristic.
That is an accepted negative finding, not a reason to tune on T or discard seeds.

For B5, V3 improves nominal AUC over V0 by **0.227**, paired 95% CI
**[0.162, 0.293]**. This supports the value of the richer sensing pipeline in
this comparison; it does not establish that each intermediate model is superior
or that the imperfect calibrated scores are reliable probabilities.

| V3 scenario | PPO AUC | B5 AUC |
|---|---:|---:|
| 250 s battery | 0.628 | 0.832 |
| 500 s battery | 0.744 | 0.903 |
| 750 s battery | 0.799 | 0.927 |
| Prior shifted 50 m east | 0.616 | 0.871 |
| Prior rotated 90 degrees | 0.710 | 0.870 |
| Prior spread doubled | 0.684 | 0.898 |

B5 outperforms PPO in all six pooled V3 scenarios. The shifted-prior case hurts
PPO particularly strongly. The full report retains family results, every training
seed, descriptive paired intervals and all other policy/sensor comparisons.

## Limits and preserved qualifications

- This is simulation-only evaluation on 80 static composited T worlds, not a
  claim of real search-and-rescue performance.
- T contains no distractors. Original low/high tags are pooled into four actual
  prior/target families of 20 worlds; distractor robustness remains untested.
- D42's V0 convergence limitation and original zero-belief diagnostic failure
  remain unchanged. This acceptance does not turn them into passing findings.
- World bootstrap intervals are conditional on the three trained policies;
  training-seed spread is reported separately. Family comparisons have limited
  power and are descriptive.
- The oracle is an L1 perfect-verification reference, not a mathematical upper
  bound. B5 exceeding its AUC and normalised scores exceeding one are permitted
  by the frozen protocol, not evidence of a broken metric.
- Detector/camera swaps, other appearance conditions and the remaining ablations
  were not run. Full M9 remains broader than this accepted existing-cache phase.

## Saved outputs and reproduction

- [Generated tables](m9-results/results.md) and [full numeric report](m9-results/results.json).
- [V0 figure](figures/m9_summary_V0.png) and [V3 figure](figures/m9_summary_V3.png).
- Episodes, metadata and original reports remain under
  `runs/eval/m9-v1/341aadf1366d8f05/`; do not restart the completed evaluator.
- Protocol: [m9-protocol.md](m9-protocol.md).

To regenerate tables and figures without rerunning any episodes:

```bash
uv run python scripts/report_evaluation.py runs/eval/m9-v1/341aadf1366d8f05 --out runs/eval/m9-v1/341aadf1366d8f05/report
```

Use the WSL prefix in [env.md](env.md). Published numeric files are byte-identical
copies of the generated report; figures contain only numerical plots.
