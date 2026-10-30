# M9 complete: extension acceptance (D48)

Accepted on 2026-10-30 under the owner's delegation to accept results unless a
serious validity problem is found. The review found no blocking validity issue.
Together with the [accepted existing-cache phase](m9-acceptance.md), this completes
the agreed M9 scope: E1/E2, battery and prior perturbations, detector and camera
swaps, a separate B distractor stress test, appearance holdout and all five
training ablations. M10 deployment has not started. Completion accepts the
experiments and their negative findings; it does not establish PPO superiority
or operational robustness.

## Evidence and verification

The [D47 protocol](m9-extension-protocol.md) and artifact manifest were frozen
before extended evaluation. Run `1c19752957aabaa0` used committed source
`5395f29` and completed all **18,720 new episodes**. Its report reuses **4,320
original nominal episodes**, checking their hashes against the published v1
report. No completed episode or training run was restarted for this review.
Across both M9 phases there are 44,640 unique evaluation episodes.

All 24 D46 training runs completed; all three seeds (17, 29, 43) remain reported.
The [training report](m9-training-results.md) includes learning curves and
diagnostics. Checkpoints were selected using B raw AUC, before T evaluation.
The 320 additional GPU caches were verified and registered, and replacement
sensors were fitted using B only. Neither policies nor baseline parameters were
tuned using the new T or stress results.

Preflight passed **322 tests**, including style, and **288 B-only smoke episodes**.
The final publication check also passed all 322 tests in 59.03 s, including lint
and formatting, after adding the review audit script.
The automatic report verified episode identities and checksums. The subsequent
[numeric audit](m9-extension-results/review-audit.json) passed all 1,106 aggregate
rows and checked 805 paired-row counts, finite metrics and intervals, budgets,
monotone found curves, seed/world counts and agreement with the original nominal
report. Oracle exceeds random in every reported family/condition/variant group.
The declaration trajectory audit checked 42,535 paid actions and 92 terminal
attempts with insufficient battery: an unaffordable action consumes the remaining
budget without performing verification. The summary figure was visually reviewed.

Published outputs are byte-preserved copies of the generated
[tables](m9-extension-results/results.md), [numeric report](m9-extension-results/results.json)
and [summary figure](figures/m9_extension_summary.png). The JSON report SHA-256 is
`79148c958a0c6031bb38d970b97cd4b4cc60b724073d13e1499ff56f1ab5b745`.

## Ablations

These are V3 PPO results on 80 unchanged T worlds, averaging action repeats and
then all three training seeds within each world. Differences use paired
whole-world bootstrap 95% intervals. The appearance comparison uses its
matched-size mixed-condition control; other differences use the fresh D46 control.

| Configuration | AUC | AUC difference from its control [95% CI] |
|---|---:|---|
| Fresh control | 0.739 | Reference |
| Distinct-variant re-look | 0.824 | +0.085 [0.058, 0.113] |
| Learned DECLARE | 0.617 | -0.122 [-0.170, -0.073] |
| Unknown K | 0.729 | -0.010 [-0.036, 0.015] |
| Potential shaping | 0.730 | -0.009 [-0.028, 0.012] |
| Egocentric map | 0.761 | +0.022 [-0.006, 0.048] |
| Appearance holdout | 0.730 | -0.011 [-0.036, 0.013] |
| Appearance matched control | 0.741 | Reference |

Re-look improves AUC but remains below the original B5 reference (0.903): paired
gap -0.079 [-0.115, -0.044]. Its independent-evidence assumption across distinct
variants is optimistic, and it changes the sensing contract. This does not
justify promoting it to the default after seeing T outcomes.

Learned DECLARE produces **57.47 false declarations per episode**, versus 0.13
for control. Verification costs 2 s when affordable; the saved trajectories
confirm that cost. The AUC objective has no false-declaration penalty, so this
is a valid failure of the prescribed task/objective, not evidence of useful
declaration behavior. Training curves fluctuate; fixed-budget completion does
not establish stable convergence.

Unknown K, shaping and egocentric maps have pooled paired intervals crossing
zero. The appearance-held-out condition has only **11 T worlds**: AUC 0.630
versus matched control 0.680, difference -0.049 [-0.119, 0.015]. This is weak,
imprecise evidence. The holdout applies to policy training only; the detector
and sensor model saw both appearance conditions.

## Frozen-policy robustness

These comparisons use the original selected PPO policies and frozen M7 baseline
settings. Detector and camera cases use their prescribed B-refitted sensors.

| Setting | V3 PPO AUC | V3 B5 AUC |
|---|---:|---:|
| Original nominal T | 0.744 | 0.903 |
| Detector swap, T | 0.589 | 0.881 |
| Camera resolution 320 px, T | 0.166 | 0.686 |
| Low distractors, B stress | 0.705 | 0.840 |
| High distractors, B stress | 0.705 | 0.861 |

The detector swap lowers PPO AUC by 0.155 [0.107, 0.209]; the camera change
lowers it by 0.577 [0.529, 0.629]. Frozen PPO therefore does not survive these
changes well, even with B sensor refits. Preserve this negative result rather
than selecting new checkpoints using T.

The paired B stress AUC difference is +0.001 [-0.008, 0.009], but PPO false
declarations rise from **0.65 to 8.26 per episode**, difference +7.61
[5.39, 10.01]. Stable AUC does not establish operational distractor robustness:
the default auto-declaration contract verifies false alarms without additional
time cost, and AUC does not penalize them. Baseline low/high comparisons also
include their pre-existing family-dependent parameter choices; PPO weights are
identical across the paired densities.

## Interpretation and remaining limits

The original main result remains PPO below B5. These experiments do not rescue
H2 or change the qualified M8 acceptance (D42). All seeds, negative results,
historical gate failures and convergence caveats remain visible. The intervals
are descriptive, without adjustment for multiple comparisons.

T still has no distractors; its original density labels are not density
contrasts. The separate B stress test completes the owner-approved substitute,
but cannot establish held-out T distractor generalization. Glint remains
excluded. Worlds are flat-sea composites with static targets and a nadir camera;
source imagery includes rectified oblique frames. Only six capture days underlie
the coarse splits, and volunteers may recur across days. Belief independence
is approximate and calibration remains imperfect. The finest-level oracle is
a privileged reference, not an upper bound; ablation normalization uses original
contextual references rather than matched task-specific oracles.

M9 is complete within these documented scope decisions. Any future policy
revision requires B-only selection and a separate protocol; do not tune on these
T outcomes. PX4/Gazebo transfer, deployment demonstration and the final paper
remain M10/M11 work.

## Reproduction from preserved local artifacts

Run from the repository root in the documented WSL environment:

```bash
uv run python scripts/report_evaluation.py runs/eval/m9-v1/341aadf1366d8f05
uv run python scripts/report_m9_training.py
uv run python scripts/report_m9_extension.py runs/eval/m9-v2/1c19752957aabaa0
uv run python scripts/audit_m9_extension.py runs/eval/m9-v2/1c19752957aabaa0
```

These commands regenerate reports and review checks from saved files, without
training or evaluation. Raw episodes, imagery, caches and checkpoints stay in
gitignored local storage. The wrapper's `awaiting_review` marker is historical;
this document and D48 record the final acceptance.
