# M9 extension evaluation (D47)

This implements the remaining D46 experiments after all 24 training runs and
320 additional GPU caches completed. The existing v1 results remain accepted
and unchanged. No T or stress outcome selects a checkpoint or setting.

## Frozen inputs and comparisons

`configs/eval/protocol_v2.yaml` freezes episode and bootstrap seeds, three action
draws, the 500 s battery and the case list. `artifacts_v2.json` freezes all
24 raw-AUC B-selected checkpoints, the six historical PPO checkpoints, world
IDs, cache/sensor configurations and 1,169 input checksums. Selection is the
largest trained-checkpoint mean B AUC, earliest step on ties. The verification
script checks every training checkpoint and all eight validation reports per
run, including B world membership, draw counts and recomputed means.

| Comparison | Worlds | Policies | New episodes |
|---|---:|---|---:|
| Eight D46 configurations | 80 T each | V3 PPO, all three seeds, three draws | 5,760 |
| Detector swap | 80 unchanged T | random/oracle/B1-B5 on V0-V3; six historical PPO policies | 4,320 |
| Camera resolution swap | 80 unchanged T | same frozen policy set | 4,320 |
| Paired B distractor stress | 40 low + 40 high | same frozen policy set | 4,320 |
| Total | | | 18,720 |

The 4,320 original nominal episodes are reused from
`runs/eval/m9-v1/341aadf1366d8f05`. Their checksums must match the published v1
report. They are never rerun or reselected. New results use a separate run ID.

Compare each ablation with the fresh D46 control; compare the appearance holdout
primarily with its matched-size mixed-condition control. Appearance training
and validation counts are 54/13; T condition counts are 69/11. This is only a
policy-training holdout: original detector/sensor fitting saw both conditions.

Detector/camera comparisons retain original PPO weights and M7 baseline
parameters, adapting only to the prescribed B-refitted sensor contract. Sensor
IDs are detector `87cfb6fc190fa842` and camera `01807b9dcc55ee7f`. No policy or
baseline is tuned to a swapped sensor or test outcome.

Stress uses original policies and sensor `fcf8cdae58ca3947`. Pair by world seed:
background, targets, appearance and priors match; distractor counts differ.
Baseline family settings use the previously selected low/high family setting
respectively. This measures the frozen family-conditioned baseline system;
its low/high difference includes those pre-existing parameter differences.
It is B imagery stress evidence, not held-out T generalisation. T still has no
distractors. Distinct-variant re-look is intentionally optimistic about evidence
independence; no result establishes calibrated posterior probabilities.

## Metrics and uncertainty

Report AUC, found fraction, success, false declarations, censored first-find
time, energy/path use, false-clear diagnostics, altitude use, found curves and
descent after coarse detection. Keep reward out of cross-objective comparisons.
Unknown-K initial false-clear diagnostics use the actual assumed-count prior.

Average action draws within each world/training seed, then average all three
training seeds within world. Bootstrap 2,000 whole-world resamples with fixed
seed 20261029, using paired differences on identical worlds; stress pairs use
their shared seed. Report per-seed metrics and seed SD separately. These are
descriptive 95% intervals, not multiplicity-adjusted significance tests.

Group by all worlds, four actual prior/target families, and appearance condition.
Show undefined normalisation when oracle-minus-random is nonpositive. For
ablations use the original nominal random/oracle references, explicitly labelled;
they are contextual references, not matched ablation-specific upper bounds.
The oracle remains a finest-level perfect-verification reference, not a bound.

## Preflight, execution and reporting

The full suite passed 322 tests. B-only smoke uses one original rl-val world
for each trained mode and historical-policy set, and one sensor-fit B world
for each replacement sensor: 288 episodes. No stress or T world is executed
during this smoke. Tests cover pairing, all seeds, earliest-tie selection,
checkpoint corruption, seven-action and egocentric adapters, unknown K,
deterministic replay and averaging draws before seeds.

Run `uv run python scripts/continue_m9_extension.py` as a hidden detached job.
It resumes verified episode files, rejects concurrent writers, writes status
under `runs/eval/m9-v2`, regenerates the report, then stops for review. The
runner's `--limit N` supports a clean episode-boundary pause. No training is
launched by this continuation.

Rebuild all extension numeric tables and figures with:

```
uv run python scripts/report_m9_extension.py runs/eval/m9-v2/<run-id>
```

Scientific review checks integrity/budgets, every training seed, paired
intervals, appearance sample limitations and whether negative findings remain
faithfully reported. Do not label full M9 complete before that review.
