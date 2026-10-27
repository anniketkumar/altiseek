# M7 implementation and validation protocol

M7 uses B imagery and the `rl-val` worlds only for baseline selection. The sensor
model `fcf8cdae58ca3947` is frozen. This is development/selection evidence, not a
held-out test result or the M9 policy evaluation protocol.

## Main settings (D36)

The owner selected known K and an allocentric north-up map with coordinate
channels after reviewing alternatives. Unknown K and egocentric maps remain
ablations. Unknown-count support, when enabled, uses a configured assumed count
and excludes actual K from observations; main runs use known K.

- Six actions: N/E/S/W, ascend, descend. Each level change snaps to the nearest
  new-level footprint centre; ties use south/west lattice order. Charge both the
  horizontal snap and altitude change, plus dwell, through the shared energy model.
- Start at L5, at a seeded random edge launch location snapped to its lattice.
  The initial observation costs 2 s and can make declarations. Initial finds,
  false declarations and energy enter episode metrics/reward accounting.
  If everyone is found at reset, a Gymnasium caller's first step returns terminal
  without another move, observation or reward. Episode logging already includes
  the initial event and need not take this extra API transition.
- Auto-declare at posterior > 0.5, one fixed rule across V0–V3 and normal policies.
  This is a decision threshold for the ablation, not a claim of reliable calibration.
  Resolve the centre cell once, finding every remaining target in it or recording
  one false declaration. Suppress later detections within 1 m of found positions.
- A cell updates only on its first observation at each level (D35). Full world
  truth is confined to declaration verification and diagnostic logging. Policies
  receive no target cells, cache truth, true prior, family label or world identity.
- Geometry masks apply in training and evaluation. Invalid masked actions are
  zero-cost no-ops that consume the 300-step cap; baselines must never use them.
  An action that cannot finish within remaining energy ends the episode without
  moving/observing; remaining energy is charged as hover. There is no free last view.
- Map channels: p, q, one observed mask per level, resolved, current footprint,
  drone position, x and y. Tensor rows run north to south and columns west to east;
  belief storage remains [ix,iy]. Vector: level one-hot, battery fraction, elapsed
  time / battery, found / K, K. Returned arrays are distinct each step.
- Reward: found count − 0.5 × false declarations − energy / initial budget.
  Optional shaping uses gamma * next potential − previous potential, with
  potential = −belief entropy and terminal potential zero. Main runs disable it.
- The oracle is an explicit privileged bound: it knows target cells, follows
  shortest energy paths to L1 views covering them, and verifies them without
  detector misses once covered at L1. It pays the same move and dwell costs.
  Its perfect verification distinguishes this bound from detector-limited policies.

## Policy and metric contracts

Normal policies implement observation + mask → action, using only public map/vector
state and the frozen sensor model. B4's objective is expected entropy reduction per
energy, with expected true positives as tie-breaker. V0 lacks a probabilistic score
model, so its planner uses its stated binary r0=0.9/f=0 surrogate; belief updates
remain the exact V0 rule. V3 integrates conditional histogram probabilities.

Found-vs-time AUC integrates the stepwise found fraction to the 500 s horizon and
divides by 500. After episode termination the last found fraction is carried to
the horizon. Initial observations are included. Report energy and actual elapsed
time separately; climbing energy can exceed wall-clock flight time. Normalised
scores use paired random/oracle AUCs and are null if the denominator is nonpositive.

Episode logs and hidden-truth diagnostics belong to a logging wrapper, not policy
observations. Record world/cache/model/config IDs, seed, actions, event times,
energy, found/false counts and altitude usage. Resume only matching run identities.

## Baseline selection grid (D37)

Run all 40 rl-val worlds with paired launch/variant seeds `20261027 + world.seed`
under every V0–V3 model. The grid in `configs/policy/baselines.yaml` was chosen
before full rl-val outcomes: B1/B2 fixed level L1–L5; B3 coarse L4/L5, confirm
L1/L2, candidate threshold 0.01/0.05. B2/B3 choose the nearest view among the
top quarter of remaining views ranked by prior mass. That quarter is fixed, not
claimed tuned. Choose each baseline's settings per family and sensor variant by
mean validation AUC; ties prefer fewer false declarations, then canonical config
order. These selected validation scores are optimistically biased, not test scores.

B1 chooses the nearer end of its boustrophedon route. Travel between goals at the
same level stays at that level, even if an altitude shortcut would cost less.
The initial validation run `2b3af75b6628cf6d` was stopped during review and is
superseded by implementation `m7-v2-fixed-level-routing`; do not combine them.
B3 interrupts the coarse
sweep on a candidate, finishes its confirmation, then resumes the unvisited
coarse views. A completed fixed-level sweep stays at that level. B4 considers
one-step actions; B5 considers up to two moves with no free travel, correcting
overlapping footprints for first-look gating. Across different levels it sums
the expected second information over the first outcome, assuming conditional
independence across levels. A 2,049-point log-odds table interpolates that sum.
This is open-loop depth-two planning, not an adaptive branch-contingent policy.

The finest-level oracle is a reference, not a per-episode mathematical upper
bound: a lucky coarse observation can find a person sooner. Do not clamp negative
normalisation denominators or scores. Report undefined ratios and use paired
family-mean oracle/random AUCs when displaying family normalised scores.

## Dependency

Gymnasium is pinned to 1.4.0. Its installed checker validates reset/step seeding,
spaces and fresh returned arrays. Sources: [release notes](https://github.com/Farama-Foundation/Gymnasium/releases)
and [environment API](https://gymnasium.farama.org/api/env/).
