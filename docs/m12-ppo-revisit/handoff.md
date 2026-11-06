# M12 hand-off: PPO revisit and evaluation power

Written 2026-11-06 for a coding agent picking this up cold. Read `CLAUDE.md` in full and
`AGENTS.md` first; they override anything here. This file is a work plan, not a frozen protocol.

## 1. Goal

M0–M11 are complete. The headline RQ2 result is negative: V3 PPO trails B5 everywhere
(rl-val 0.720 vs 0.888 AUC; test 0.744 vs 0.903, D45), and live transfer costs PPO
−0.135 AUC (D60). This milestone tests whether that gap is a property of learned search or
an artifact of the setup. It has three parts:

1. **Energy model correction (B only).** Live steps cost more time than the cached model
   charges, because settling, capture, inference and decisions consume simulator time.
   Fit that overhead on B, retune the baselines, and train under the corrected model.
2. **B5 warm start for PPO.** Behaviour-clone B5 into the existing network, then fine-tune
   it with the D41 AUC objective for longer (about 1M steps instead of 131k).
3. **More test worlds.** A fresh T test set of 100 worlds per family (800 worlds) replaces
   10 per family as the primary evaluation. The original 80 worlds are reported as secondary.

The outcome may still be negative. That is fine. A negative result under a warm start from
the strongest heuristic is a stronger finding than the current one. Report it as it comes.

## 2. Rules specific to this work

General rules are in CLAUDE.md §23 and AGENTS.md. On top of those:

- **The 1.65× overhead figure is from T. Do not use it.** `docs/m10-deployment/m10-acceptance.md`
  reports live steps at 17.6 s (B5) and 16.0 s (PPO) against 10.2/9.6 s nominal. Those are
  post-hoc measurements on the T transfer flights. Using them, or anything else from
  `docs/m10-deployment/m10-transfer-results/`, to set a parameter is test-set tuning.
  Re-estimate the overhead from B live data only (§4).
- **Never edit frozen files.** `configs/eval/*`, `configs/policy/selected.yaml`,
  `configs/env/energy.yaml` and existing world-set entries stay byte-identical. New settings go
  in new files.
- **Do not change existing artifact identities.** If you add a field to `EnergyModel`,
  `PPOConfig` or any other pydantic config, check every place that hashes a resolved config
  (cache IDs, run IDs, protocol hashes) and prove historical IDs are unchanged. Prefer new
  config *values* over new schema fields. Verify historical artifacts at tag `results-final`.
- **All design choices are made on B** (rl-train, rl-val, sensor-fit, B live flights). The
  new test set is generated before the protocol is frozen, but nothing is evaluated on it until
  the protocol is frozen.
- **PPO trains and evaluates on CPU** (CLAUDE.md §2). Do not spend Kaggle GPU quota on PPO.
- **Decision Log:** next free ID is D63. Log the M12 start and the energy estimator (D63), the
  protocol freeze (D64), and anything else that deviates. Add an M12 row to CLAUDE.md §0 and §21.
- **Tests:** on this laptop Windows Application Control blocks `pytest.exe`; run
  `uv run python -m pytest`. Measured 2026-11-04: 378 passed, 2 skipped (they need `fcntl`).
  AGENTS.md says 391 at D62; check which count is current before relying on either.

## 3. Where things are

| What | Where |
|---|---|
| Energy model (shared simulator/baselines/Gazebo) | `src/altiseek/env/energy.py`, config `configs/env/energy.yaml` |
| Environment | `src/altiseek/env/search.py`, `configs/env/search.yaml` |
| Baselines incl. B5 | `src/altiseek/policies/baselines.py`; grid `configs/policy/baselines.yaml`; frozen choice `configs/policy/selected.yaml` |
| Baseline tuning run | `scripts/search/run_baselines.py`, report `scripts/search/report_baselines.py` |
| PPO config/network/wrapper | `src/altiseek/policies/ppo.py` (`PPOConfig`, `SearchFeatures`, `FiniteSearch`) |
| PPO training helpers | `src/altiseek/policies/training.py` (`make_model`, `save_checkpoint`, `load_checkpoint`, `evaluate`) |
| PPO entry point | `scripts/ppo/train_ppo.py` (`--config`, `--variant`, `--seed`, `--n-envs`, `--threads`, `--vectorization`); D41 config `configs/policy/ppo_auc.yaml` |
| PPO protocols | `docs/m8-ppo/ppo-protocol.md`, `docs/m8-ppo/ppo-auc-protocol.md` |
| World sets | `configs/world/default.yaml` (`world_sets`), `scripts/world/generate_worlds.py` |
| Cache build on Kaggle | `scripts/cache/` (`prepare_cache_inputs.py`, `cache_production.py`, `collect_cache.py`, `validate_cache.py`); `docs/data/cache.md`; Kaggle push via `scripts/detector/kaggle_job.py` |
| Frozen evaluation pattern to copy | `scripts/eval/freeze_m9_extension.py`, `configs/eval/protocol_v2.yaml`, `artifacts_v2.json`, `docs/m9-evaluation/m9-extension-protocol.md` |
| M10 live accounting | `docs/m10-deployment/m10-live-protocol.md`, `src/altiseek/deploy/meter.py` |
| B live data | B calibration acquisition `cf148ad98784906d` (40 sensor-fit worlds, 8,800 views, D55, `docs/m10-deployment/m10-calibration-review.md`); B transfer preflight on rl-val world `48a9325328a6d2c4` (`docs/m10-deployment/m10-transfer-preflight.md`); B integration `docs/m10-deployment/m10-measured-budget.json`; raw logs under `data/m10/` |

Caches total about 30 MB (`data/caches/`), so a private Kaggle dataset of the B caches and
sensor model is cheap if you use Kaggle CPU (§7).

## 4. Phase 1: energy correction (B only)

1. **Declare the estimator before computing it** (D63): which B live records are used, what
   "overhead" means (measured elapsed per view minus nominal `move` + `observe` time for the
   same displacement), and the summary statistic (for example the median with a bootstrap CI,
   by level if it differs). Calibration sweeps fly different paths from search episodes;
   say whether settling scales with move length and model it only if B shows it does.
2. Compute it from B data only. Write `docs/m12-ppo-revisit/energy-correction.md`.
3. Implement it as a new config, `configs/env/energy_live.yaml`, by preference raising
   `dwell_s` (no schema change). Add a field only if the B data require a distance-dependent
   term, and then follow the identity rule in §2.
4. Check the §8 design constraints still hold under the new costs: full L1 coverage must
   exceed the 500 s battery and full L5 coverage stay cheap. If not, stop and ask the owner
   before changing the battery.
5. Retune B1–B3 on rl-val with the existing `configs/policy/baselines.yaml` grid under the new
   energy config. Write the result to `configs/policy/selected_live.yaml`; leave
   `selected.yaml` untouched. B4/B5 have no tuned parameters, but rerun them on rl-val for the
   reference numbers.

**Stop point for the owner:** the overhead value, its CI, and whether it changes B5's rl-val AUC.

## 5. Phase 2: B5 warm start and V3 pilot

**Implementation.**
- `src/altiseek/policies/imitation.py`: demonstration collection and pre-training. Use only
  the observation and action mask the PPO policy would see; the expert (B5) may use the belief
  and sensor model as it normally does, never ground truth.
- `scripts/ppo/collect_demos.py`: run B5 (V3 sensor, `selected_live.yaml`, new energy config)
  on rl-train worlds. Use several reset seeds per world so launch points and variant draws vary.
  Record per step: observation, mask, B5's action, and the D41 AUC reward. Measure how many
  transitions you get before fixing the count; B5 needs few steps per episode.
- Pre-training: masked cross-entropy on the actor logits against B5's actions, plus value-head
  regression on the Monte-Carlo D41 return (gamma 1). Without value pre-training, early PPO
  updates against a random critic tend to wreck the cloned policy. Split by world (about 10%
  of rl-train worlds held out) for early stopping. Never use rl-val for pre-training.
- `scripts/ppo/train_ppo.py`: add an option to start from the pre-trained weights. Step 0's
  evaluation then measures the cloned policy.

**Tests:** imitation loss respects masks (masked actions get zero probability), demonstrations
contain no ground-truth fields, a warm-started run resumes exactly from a checkpoint
(the existing continuation test pattern), and old configs still resolve to identical IDs.

**Pilot (about 5–6 h on the laptop).** V3, seed 17, D41 hyperparameters
(`ppo_auc.yaml`), new energy config, `total_steps` 1,048,576, `eval_every` 65,536 (both divide
the 512-transition rollout). Fix one fallback before starting: if the curve collapses well
below the step-0 value within the first 262,144 steps, rerun once with learning rate 1e-4.
Do not tune anything else.

Report on rl-val (sampled actions, three repeats per world, as D41):
- the cloned policy's AUC at step 0 against B5 under the same energy config;
- the learning curve;
- the best checkpoint against B5.

If the clone is far below B5, covariate shift is the likely cause. DAgger, which relabels the
clone's own states with B5's actions, is the fallback. Propose it to the owner; don't add it
silently.

**Stop point for the owner:** go/no-go for the full runs, and their step count (1M or 2M).

## 6. Phase 3: test-v2 worlds and protocol v3

**New test set.** Add a new world-set entry (for example `test-v2`, split T, `per_family: 100`,
a new `seed_base` such as 5,000,000, disjoint from 1M–4M). Do not edit existing entries.
Generate only the new set and check that every existing world ID is unchanged. The T library
is unchanged, so T stays distractor-free (D43) and the original low/high distractor tags stay
meaningless on T.

**Caveat for the write-up:** every T world is built from the same small library (83 swimmer
crops, 567 patches, three videos from one day). More worlds narrows the uncertainty from
scenario layout, not from imagery. Say so wherever the new intervals are reported.

**Cache build (Kaggle GPU, about 17 h).** 800 worlds × 70 s measured (D33) ≈ 15.6 h, plus live
validation, over two 12-hour sessions. It resumes per world. Use the D32 byte-preserved input
path and verify every ID on collection. The owner has 30 GPU hours this week. The remainder is
reserve, for example more B rl-train worlds if the pilot shows overfitting. Ask before using it.

**Protocol v3** (`configs/eval/protocol_v3.yaml`, `artifacts_v3.json`, a freeze script modelled
on `freeze_m9_extension.py`, and `docs/m12-ppo-revisit/protocol.md`), frozen before any test-v2
evaluation:
- energy config `energy_live.yaml`; baselines from `selected_live.yaml`;
- methods: random, oracle, B1–B5 on V0–V3; warm-started PPO, all seeds, three sampled repeats;
  the original M8 V0/V3 PPO checkpoints as a reference (trained under the old costs);
- the from-scratch control, if the owner approves it (§7);
- primary comparison: warm-started V3 PPO against V3 B5, normalised AUC, paired by world,
  bootstrapped over worlds after averaging repeats and seeds (as D47);
- secondary: per family, the battery sweep, and the original 80 test worlds;
- every seed reported; no checkpoint reselection after test-v2 results exist.

**Stop point for the owner:** approve the protocol before it is frozen.

## 7. Phase 4: full PPO training

- Minimum: warm-started V0 and V3 × seeds 17/29/43 (six runs).
- Recommended addition: from-scratch V3 × three seeds under the new energy config. Without it,
  a gain cannot be attributed to the warm start rather than the energy change. Owner decides.
- Checkpoint selection on rl-val only, as D41.
- Throughput measured in M8/M9: 61–95 steps/s for the standard map (egocentric was 21–29),
  0.6–0.9 h per 131k-step run including validation. Running two jobs at once on the 8-core
  laptop gave little extra total throughput. Run one at a time on the laptop.
- **Kaggle CPU sessions** may not count against the GPU quota. Verify this on the account
  before relying on it. If true, run seeds there in parallel: upload the B caches, B world
  specs and sensor model as a private dataset, and make sure every kernel resumes from its last
  checkpoint (sessions end at 12 h).

## 8. Phase 5: evaluation and write-up

- Run protocol v3 on CPU and verify identities and checksums as in M9.
- Write `docs/m12-ppo-revisit/results.md` and an acceptance review. Add figures through
  `scripts/figures/make_figures.py` from saved results only.
- Update `docs/paper.md` with a new section. Keep every earlier result and qualification as
  they are. Update `README.md`, `docs/README.md`, CLAUDE.md §0/§21/§24 and AGENTS.md.

**Optional, owner decision:** re-fly the M10 transfer with the new PPO and B5 under a new
transfer protocol (24 T worlds, 16–20 h of live wall time measured in M10).

## 9. Compute budget

| Phase | Where | Estimate |
|---|---|---|
| Energy fit + baseline retune | Laptop CPU | 2–3 h |
| Demos + pre-training | Laptop CPU | 2–3 h |
| V3 pilot, 1M steps | Laptop CPU | 5–6 h |
| Six warm-started runs, 1M steps | Laptop or Kaggle CPU | 25–35 h (50–65 h at 2M) |
| From-scratch control, three runs | Laptop or Kaggle CPU | 13–18 h |
| test-v2 caches, 800 worlds | Kaggle GPU | about 17 h |
| Protocol v3 evaluation | Laptop CPU | about 5–10 h (scaled from M9: 25,920 episodes in about 1.6 h) |
| Optional live re-flight | WSL / Gazebo | 16–20 h |

These are scaled from measured runs. Re-measure after the pilot and update this table.

## 10. Done when

- D63/D64 logged; energy correction documented from B only; `selected_live.yaml` frozen.
- Pilot reported and the owner's go/no-go recorded.
- test-v2 caches built, validated and registered; protocol v3 frozen before evaluation.
- All approved PPO runs complete, every seed reported.
- Protocol v3 evaluated; results, acceptance review, figures and paper section committed.
- No frozen file edited, no historical ID changed, no T data used for any choice.
