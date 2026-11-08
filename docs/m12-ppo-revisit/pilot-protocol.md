# M12 Phase 2: B5 warm start and V3 CPU pilot (D66)

Declared 2026-11-08 before demonstration collection, imitation training or the
pilot. The owner explicitly authorized Phase 2 after reviewing Phase 1. Stop
after the pilot for the full-run go/no-go and 1M/2M budget decision. No T access,
test generation, new flight or full seed suite is authorized here.

## Fixed inputs and demonstrations

Use the existing 160 rl-train B worlds, V3 sensor `fcf8cdae58ca3947`,
`configs/env/search_live.yaml`, the exact D63 dwell and 500 s battery, and
`configs/policy/selected_live.yaml` from D65. B5 receives only its normal public
observation/mask plus the shared navigation and sensor model; no target locations.

Declare four reset draws per world as an episode budget, with environment seeds
`20261108 + repeat * 1000000 + world.seed` for repeats 0–3, and B5 seed one higher.
Run repeat 0 on all 160 worlds first as an inventory; save it and measure the
transition count, bytes and throughput before expansion. Resume the same
collection to repeats 1–3, reusing every completed episode. There is no target
transition count, rejection sampling or favorable-trajectory filtering. Report
the actual count before pretraining; initial-completion episodes contribute zero
action transitions and remain recorded. If storage/runtime is infeasible, stop
and declare any budget amendment before collecting its additional data.

Each atomic, checksummed episode stores only the PPO observation (`map`, `vector`),
mask, B5 action, D41 action reward and backward Monte-Carlo return (gamma 1).
Metadata identifies B world, cache, seeds and input/source hashes. No imagery,
hidden target fields, detector truth or privileged expert state enters the arrays.
The initial observation's action-independent contribution is not assigned to an
action. Verify action reward sum plus the initial contribution against saved AUC.
All arrays, checkpoints and episode outputs stay in ignored data/ and runs/.

## World split and supervised training

Hold out exactly two of the 20 worlds per family (16 total), ranked by
`digest(["m12-bc-holdout-v1", world_id])`, using only rl-train IDs/family tags.
The other 144 worlds supply training transitions. Every reset of a world remains
in its partition. No rl-val world is used for gradient updates or early stopping.

Seed 17, CPU with two torch threads, existing SearchFeatures CNN, actor and critic
architecture. Adam at 3e-4, batch 32, maximum 40 epochs, no augmentation, no class
reweighting. Clip gradient norm at 0.5, as in the existing PPO optimizer.
Each epoch visits every training transition once: seeded shuffle of
episode shards, then seeded shuffle of transitions within each shard; a short
last batch is retained. Keep one compressed shard in memory at a time.

Loss = masked action cross-entropy + 0.5 times mean squared error between value
predictions and Monte-Carlo returns. Invalid actions have zero probability and
cannot be labels. Holdout loss uses the same expression, averaged over transitions.
Select the smallest holdout loss, earliest epoch on ties; stop after five epochs
without an improvement of at least 1e-4. Save the actual minimum-loss weights
even when an improvement is too small to reset patience. Report losses, masked
accuracy and critic MSE on both partitions. Pretraining resumes at complete epoch
boundaries with optimizer, shuffle generator and torch RNG preserved.

Warm start imports only actor/CNN/critic weights into a fresh PPO optimizer. It
does not import the imitation optimizer's moments. Verify observation/action
spaces, architecture and checksums. Standard rollout checkpoints must resume the
warm-started PPO exactly. No field is added to PPOConfig or any existing schema;
the new imitation configuration is separate. Prove existing config dumps and
saved historical plan IDs are unchanged; source-hashed prospective training runs
correctly receive new identities when implementation sources change.

## Pilot and one fixed fallback

V3, seed 17; D41 hyperparameters, objective and sampled evaluation (three action
draws for every one of 40 rl-val B worlds). Use corrected-cost baseline references
from `docs/m12-ppo-revisit/baseline-results.json`. Primary checkpoint selection
remains D41 mean family-normalised AUC, ties earliest. Also report raw AUC against
the D65 V3 B5 reference (0.817165973214872) and the step-zero clone.

Budget 1,048,576 PPO steps; evaluation/checkpoint every 65,536 steps, preserving
512-transition rollouts. Step zero evaluates the selected clone before any PPO
update. Argmax is a secondary diagnostic. Checkpoints and jobs are detached and
resumable; run one training job at a time on the laptop.

Quantify the handoff's collapse fallback prospectively: if **two consecutive**
primary evaluations among 65,536/131,072/196,608/262,144 have raw mean AUC at
least **0.10 below step zero**, stop the primary run at 262,144 and run exactly
one new pilot from the same cloned weights with learning rate 1e-4. Preserve the
partial primary run and report both. Otherwise continue the same run to 1M.
The fallback alone changes learning rate; no further tuning is authorized.

A clone deficit of more than 0.15 raw AUC relative to B5 is flagged as substantial
for owner review, without changing the pilot budget or training recipe. Covariate
shift is a possible cause, not established by the gap alone. DAgger is only a
proposal for the owner; do not add it silently. A bad curve or negative comparison
is retained. Non-finite optimization, mask violations or identity/split failures
stop for diagnosis rather than changing scientific settings.

## Outputs and owner stop

Publish demonstration inventory/counts, world split and input hashes, imitation
curves, step-zero versus B5, every pilot evaluation, selected checkpoint, any
fallback trigger, and learning curve under `docs/m12-ppo-revisit/`. Numeric plots
may go in docs/figures/. Retain D63's original coverage failure, D65's explicit
30% relaxation, every historical negative finding and all new negative outcomes.

Run `uv run python -m pytest` (Windows), commit and push tested increments with
explicit file lists and no attribution trailers. Stop with the Phase 2 results
for the owner's full-run go/no-go and budget decision. D64 remains reserved for
the later evaluation freeze; next new decision D67.
