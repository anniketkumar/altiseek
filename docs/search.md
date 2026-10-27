# M7 environment and baseline reproduction

M7 is complete. The corrected validation run is `8cd76bdb72f965a1`;
the selected settings are in [selected.yaml](../configs/policy/selected.yaml).
[Results](baseline-results.md) and [numeric tables](baseline-results.json) report
all seven methods under V0–V3. The [protocol](search-protocol.md) defines the
observation, reward, movement, selection and information-planning contracts.

## Saved artifacts

All large outputs remain local and gitignored. No detector inference is needed.

| Path under `data/baselines/` | Contents |
|---|---|
| `8cd76bdb72f965a1/` | Plan, 3,520 episode files, completion record, run metadata, replay and selection metadata |
| `sanity/e8fe99ea7d56cb12/` | Plan, 320 coverage episodes and original step-only summary |
| `sanity/e8fe99ea7d56cb12/initial-822759b7fb88637d.json` | Separate initial-observation diagnostic, linked to the original summary by SHA-256 |

The full validation and coverage workers loaded code at `07d0c81`. Fixed-level
travel stays on that level; the previous `2b3af75b6628cf6d` run was stopped and is
superseded. Do not combine runs. Metadata may record a dirty tree because reporting
and performance edits continued while detached workers ran. The optimized
environment subsequently replayed all 320 saved random/oracle episodes and 7,606
steps with matching rewards, counts, time, energy, paths and summary metrics.

The full replay measured core step mean 1.590 ms, median 1.403 ms and p95 2.956 ms.
This excludes reset, logging and policy planning. It does not meet the sub-ms aim;
M8 needs an end-to-end CPU throughput measurement before selecting worker counts.

## Regenerate the report from completed runs

Run from the repository root. This reads saved episodes, checks the complete grid
and identities, and rewrites numeric reports and the deterministic selected config.
It does not repeat validation episodes.

```powershell
uv run python scripts/report_baselines.py data/baselines/8cd76bdb72f965a1 --sanity data/baselines/sanity/e8fe99ea7d56cb12/summary.json --initial-check data/baselines/sanity/e8fe99ea7d56cb12/initial-822759b7fb88637d.json
uv run pytest
```

`selection.meta.json` hashes the episode inputs and resulting reports/config.
Only numeric plots are committed. Selection uses five validation worlds per
family; bootstrap intervals after selection do not remove selection bias or
dependence from shared source imagery. No M7 policy was tuned or evaluated on T.

## Reproduce diagnostics when code changes

The completed results do not need restarting. These commands are for deliberate
revalidation; preserve the existing outputs and metadata when changing behavior.

```powershell
uv run python scripts/benchmark_search.py data/baselines/8cd76bdb72f965a1
uv run python scripts/check_initial_observation.py data/baselines/sanity/e8fe99ea7d56cb12
```

The first command replays saved random/oracle actions against the current core.
The second repeats only the 40 initial V0 coarse observations. The original
step-only `summary.json` remains failed and unchanged: it omitted reset. Combining
it with the initial check gives one crossing from 0.011659996 to 0.001178365,
without changing the p = 0.01 threshold. Of 78 targets, 75 started below that
threshold, so this diagnostic supplies only weak evidence of false clearing.
The report explicitly records both the original and completed checks (D38).

For full reruns, `scripts/run_baselines.py` and `scripts/check_search.py` expose
their arguments with `--help`; they resume per-episode files under matching plan
identities. Run long jobs detached with hidden `Start-Process`, redirect stdout
and stderr, and retain the printed run directory. A changed method or diagnostic
must use a new implementation/run identity, never overwrite accepted episodes.

## M8 integration

- Load `CacheConfig`, the frozen `SensorModel` and `SearchConfig`; construct
  `SearchEnv` over rl-train world specs with a shared `Navigation` and bounded
  `CachePool` per worker. Core loads refuse mismatched cache contracts.
- Keep main settings known K, north-up map, coordinates and 500 s energy budget.
  `reset(seed=...)` returns `(observation, info)`; `step(action)` follows Gymnasium.
  Use `action_masks()` for training and evaluation. Normal policies consume only
  the observation and mask; ground-truth fields are reserved for diagnostics.
- If reset finds everyone, `info['ended']` is true. A Gymnasium training loop's
  next step returns terminal with no additional observation, cost or reward.
  `EpisodeLog` includes the initial observation in episode metrics and can stop
  immediately. Returned observation arrays do not share storage between steps.
- Load `SelectedBaselines` through `load_config` and use
  `settings[family][variant][method]` for baseline comparisons. Family chooses
  frozen hyperparameters outside the policy; it is not a policy input channel.
- Use a custom CNN extractor for the float32 map, concatenate the vector, and
  preserve masks. Choose PPO implementation and tracker before heavy installs.
  CPU training, at least three seeds, rl-val checkpoint selection, and the
  zero-belief-channel sanity check belong to M8. Test-world distractors must be
  settled before M9; the M6 sensor model remains frozen.
