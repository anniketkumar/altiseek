# M8 failure diagnosis (D40)

The six initial runs and their selected checkpoints remain unchanged. Their full
report is `docs/ppo-results.md`; this investigation uses only B training and
validation worlds. It is diagnostic work, not a replacement evaluation protocol.

The original full selected-result JSON was archived byte-for-byte at
`runs/ppo/m8-initial/selected-results.json`; its SHA-256 matches the diagnostic
plan's original input. `docs/ppo-results.json` publishes per-world metrics without
the step trajectories. Historical metadata refers to the original document path;
use the archived file when checking that recorded input hash.

## Frozen-checkpoint checks

`scripts/diagnose_ppo.py` runs each of the six selected checkpoints on all 40
rl-val worlds with the M7/M8 environment seeds. Deterministic replay must match
the original AUC, rewards, found fractions and step counts. Sampled-action
evaluation adds three explicitly seeded draws per world from the same masked
action distribution. No checkpoint is reselected using those draws.

At every tenth deterministic decision, record the action distribution and
intervene on public inputs while holding the state and action mask fixed:
rotate p, zero p after preprocessing, rotate q, zero coverage masks, rotate the
drone marker, or zero the map. Report total variation distance and argmax
changes. These are sensitivity measurements; altered inputs may be off the
training distribution, so sensitivity alone cannot establish an architectural
root cause. No hidden target coordinates enter these interventions.

Count repeated view positions and steps receiving no informative cell update.
Compute undiscounted action return and gamma=0.99 discounted action return from
saved rewards, excluding the initial observation that precedes policy actions.
Compare with saved baseline trajectories under the same rules. The initial
observations are paired; excluding them matches what PPO actually optimizes.

## Controlled learning check

`scripts/check_ppo_overfit.py` selects the smallest world ID in each of four
rl-train families: sharp-single-low, diffuse-single-low, sharp-group-low and
diffuse-group-low. Selection does not depend on achieved search outcomes.
For each world, choose the first seed in 20261028..20261127 whose initial
observation has not already ended the episode, then fix that launch/variant
sequence on every reset. Each of four workers receives one of those worlds.

Train a fresh V3 network with the original architecture, reward and optimizer,
seed 17, for 32,768 transitions. Evaluate at 0/8192/16384/24576/32768 with both
argmax and three sampled action draws on those exact four training episodes.
Run B5 on the same episodes as a feasibility reference. This intentionally
measures memorization/optimization, not generalization or validation performance.
All outputs and resumable checkpoints go to a separate content-addressed
directory under `data/ppo-diagnosis/`.

## Interpretation

Distinguish demonstrated causes from hypotheses. In particular, compare the
actual training objective before interpreting low AUC as a failure to optimize
that objective. Sampling improvements identify a train/evaluation action-rule
gap, but do not prove the policy uses belief. Successful controlled learning
rules out some gross implementation failures, not all bugs or representation
limitations. Preserve negative results and avoid claiming a single universal
root cause if the sensor variants behave differently.
