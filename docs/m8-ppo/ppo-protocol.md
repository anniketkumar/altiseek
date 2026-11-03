# M8 CPU PPO protocol

The owner selected MaskablePPO with a custom CNN and local TensorBoard on
2026-10-28 (D39). This protocol selects on B validation only. M9 test evaluation
requires a separately frozen protocol; no M8 script loads test worlds.

## Data and model

- Train on all 160 rl-train worlds, sampled uniformly at reset. The eight
  families have equal world counts. Validate on all 40 rl-val worlds with the
  same launch/variant seeds as M7 (`20261027 + world.seed`). Keep sensor model
  `fcf8cdae58ca3947`, all cache identities and M7 selected baselines frozen.
- Main settings retain known K, north-up coordinates, five levels, first-look
  belief updates and 500 s battery. No reward shaping. No privileged information
  reaches the policy or value network. Family labels only group evaluation.
- Train separate V0 and V3 policies, each with seeds 17, 29 and 43. This is the
  compute-limited RQ1 option allowed by CLAUDE.md section 15; the existing
  baselines cover all four sensor variants. Report every training seed.
- Pin sb3-contrib and stable-baselines3 to 2.9.2, TensorBoard to 2.21.0. Existing
  Gymnasium 1.4.0 and CPU torch 2.11.0 satisfy their published requirements.

## Network and optimizer

The custom extractor processes the float32 map with convolutions
12→16 (5x5, stride 3), 16→32 (3x3, stride 2), 32→32 (3x3, stride 2), ReLU
after each. Adaptive average pooling retains a 4x4 spatial grid. Concatenate the
flattened features with the vector and use a 128-unit ReLU layer; separate
64-unit policy/value MLPs feed the masked six-action distribution and value.
For other level/coordinate settings, input channels follow the observation space.

The sparse probability/prior channels use the fixed transform
`1 + log(clamp(value, exp(-20), 1)) / 20`. This exposes their dynamic range without
fitting statistics or adding information. Other channels are unchanged; K is
divided by the fixed observation-space maximum. No image /255 normalization.
The zero-belief diagnostic sets channel p to exactly zero after preprocessing,
while keeping prior, coverage, resolved cells, coordinates and vector inputs.

Initial settings in `configs/policy/ppo.yaml`: learning rate 0.0003, discount 0.99,
GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, four epochs, minibatch 64,
target KL 0.03. Each rollout has 512 transitions (four environments × 128 steps).
Each seed gets 131,072 transitions, with validation/checkpointing every 16,384.
This is an initial fixed budget, not a promise of convergence. Extend or change
settings only through a new logged configuration, retaining all earlier results.

CPU pilot (1,024 transitions, two optimizer epochs, seed 17/V3): four in-process
environments achieved 86–91 steps/s with one torch thread and 112–125 with two;
four subprocess environments with two threads achieved 148–185. Production
therefore uses four subprocess environments and two torch threads. These short
pilot rates include rollout/update work, exclude setup/evaluation/checkpointing,
and do not predict four-epoch production throughput exactly. Pilot artifacts are
under `runs/ppo/b2b97a515cfbc586`, `42188db0bd5df40e` and `6fb132f6deb4d91d`.

The core's battery/cap truncations end the finite task. `FiniteSearch` marks them
terminal for SB3, preventing its normal timeout value bootstrap. Evaluation uses
the original M7 Gymnasium API. Initial observations precede policy actions: their
rewards appear in evaluation metrics, not the training transitions. Training and
evaluation both supply masks; an invalid selected action raises immediately.

## Checkpoints, selection and diagnostics

- Save policy and optimizer plus worker world/state, belief, RNG, last observation
  and episode-start flags after complete optimizer updates. Preserve Python,
  NumPy and torch RNGs. A completion marker with SHA-256 hashes is written last;
  interrupted incomplete saves are ignored. Load only trusted local checkpoints.
  An integration test compares resumed and uninterrupted weights/observations.
- The run identity hashes the resolved settings, implementation sources, lockfile,
  sensor, frozen baseline references and verified training/validation caches.
  A different configuration or implementation creates a different run directory.
- Checkpoint selection maximizes the equally weighted mean of family-normalized
  validation AUC, using frozen M7 random/oracle family means. Nonpositive reference
  denominators are errors. Ties prefer the earliest trained checkpoint. An
  untrained step-zero evaluation is logged but cannot be selected.
- Log optimizer diagnostics to local CSV and TensorBoard. Save per-checkpoint
  validation JSON with all world metrics and trajectories. Report raw AUC, found
  fraction, false declarations, first-find times and time per level, alongside
  normalized scores. Compare every seed with frozen B3 and the strongest baseline.
- At each selected checkpoint, evaluate again with the belief channel zeroed.
  Compare paired results with the frozen prior-only B2 heuristic. A better score
  triggers the specified leakage investigation; do not silently redefine the
  gate or assert that it passed. This is an input-removal diagnostic, not a
  separately trained prior-only policy.
- Behaviour analysis includes altitude time and immediate descent after an L4/L5
  cached swimmer detection above tau. This last diagnostic uses raw cached views,
  including detections that may later be suppressed/resolved; it is not a claim
  that every such detection remained informative. Detection data never enter
  policy inputs through this diagnostic.

## Sources checked

- [MaskablePPO API](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html):
  dictionary policies, evaluation masks and in-environment `action_masks` for workers.
- [Custom feature extractors](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).
- Package metadata from PyPI for [sb3-contrib](https://pypi.org/project/sb3-contrib/2.9.2/),
  [stable-baselines3](https://pypi.org/project/stable-baselines3/2.9.2/) and
  [TensorBoard](https://pypi.org/project/tensorboard/2.21.0/).
- Installed `ppo_mask.py` inspected for truncation bootstrap, mask collection,
  optimizer checkpointing and vector-environment semantics before integration.
