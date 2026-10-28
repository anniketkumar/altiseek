# M8 AUC revision (D41)

Approved by the owner on 2026-10-28 after the D40 diagnosis. Preserve the six
initial runs and their reports. This protocol is recorded before revision training.

## Objective and controls

Train on B rl-train only. Each newly found target at elapsed time t contributes
`max(0, 1 - t / 500) / K`; gamma is 1. The sum of these rewards plus the initial
observation's contribution equals the logged found-fraction AUC. The initial
observation precedes actions and is not credited to an action. Battery remains a
hard constraint. False declarations and energy remain reported outcomes, with no
additional training penalty. There is no shaping. Known K is required.

Keep the CNN, observations, sensor model, worlds, seeds, optimizer and 131,072-step
budget from D39. `configs/policy/ppo_auc.yaml` changes only objective, discount and
evaluation action rule. Core M7 rewards remain available for historical comparison;
the training wrapper supplies the revised reward. This is a bundled objective and
evaluation correction, not an isolated causal estimate for each component.

## Selection and evaluation

Primary: sample masked actions, three independent draws per each of 40 rl-val
worlds, at step zero and every 16,384 steps. Action RNG seed is
`20261028 + repeat * 1000000 + world.seed`; environment seeds retain D39's rule.
Local action RNGs do not consume training RNG state. Secondary: one argmax episode
per world at every checkpoint. Never select between action rules after seeing scores.

Select the trained checkpoint with highest primary mean family-normalised AUC;
ties choose the earliest. Repeat the primary evaluation with its p input zeroed.
Average action repeats within each world before paired comparisons or bootstrap;
repeats are not independent worlds. Report every seed, both action rules, false
declarations, altitude use, coarse-detection descents, repeated visits and views with
no informative cells. Preserve the section 19 zero-p check against B2; investigate
any failure rather than silently relaxing it. No test-world access or tuning.

## Pilot and continuation

V3 seed 17 pauses after 65,536 steps using `--stop-after 65536`. This is a pause
within its full run identity, not a separate run to discard. Proceed to all six runs
if its selected primary mean AUC is at least the frozen V3 random reference
(0.659093), with finite optimizer metrics and no invalid actions. Otherwise inspect
the recorded curves and behavior before spending the remaining budget. This pilot
threshold is a resource decision, not a significance test or held-out result.

If it passes, resume the same directory to 131,072 steps, then run the remaining
V3 seeds and all three V0 seeds. Outputs live under `runs/ppo/m8-auc`; reports use
`docs/ppo-auc-results.*`. Never overwrite `m8-initial` or its reports. M8 completion
requires the six runs, reviewed learning curves, behavior analysis and sanity checks;
launching training alone does not complete the milestone.

Run `uv run python -u scripts/continue_ppo_auc.py` as a hidden detached process.
It resumes the pilot, records `pilot-gate.json`, expands only after a pass, and
generates the separate report. Check `continuation-status.json` and the per-run
logs. A failed gate stops expansion; a completed report still requires scientific
review before changing the milestone status.
