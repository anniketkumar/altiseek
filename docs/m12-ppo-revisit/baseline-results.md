# M12 Phase 1: corrected-cost baseline results

D65 run `634beeb0cc8f4664`; 3,520 CPU episodes on 40 rl-val B worlds, V0-V3, the unchanged M7 candidate grid and seeds. Audit `c915e17ad24d307c`. No T data or new flight was used.

**Phase 1 is complete, pending owner review.** B1-B3 were retuned and B4/B5 rerun under the fixed B-fitted costs. The sensor, 500 s battery, speeds and climb multiplier remain unchanged. PPO training and Phase 2 have not started.

The [D63 fit](energy-correction.md) remains **6.42 s extra per view [3.31, 7.16]**, giving **8.42 s dwell**. Its original 25% L5 gate failed: 135.78 s exceeds 125 s. The owner-authorized [D65 amendment](baseline-protocol.md) allows 30% for M12 only. The same 135.78 s costs 27.156% of 500 s, within the new 150 s allowance; L1 costs 1,318.82 s and remains infeasible. This explicit post-B amendment does not erase the original failure.

## B5 paired comparison

Old-cost B5 episodes are reused from saved run `8cd76bdb72f965a1`, not rerun. Each pair has the same world, sensor variant and reset seed. Differences include changes to B5's energy-aware action choices as well as timing; this is not just a rescaling of a fixed path.

| Sensor | Original AUC | Corrected AUC | Paired change [95% CI] |
|---|---:|---:|---:|
| V0 | 0.551420 | 0.467014 | -0.084406 [-0.118847, -0.053166] |
| V1 | 0.770755 | 0.676683 | -0.094072 [-0.130397, -0.064401] |
| V2 | 0.766314 | 0.671573 | -0.094740 [-0.132321, -0.059843] |
| V3 | 0.887557 | 0.817166 | -0.070391 [-0.099328, -0.046556] |

The V3 reference changes by -0.070391 AUC. The paired interval lies below zero. This measures B5 under a more costly simulator; it does not measure PPO improvement or live transfer. No setting was changed in response to the sign.

## All selected baselines

| Sensor | Method | AUC [95% CI] | Found fraction | False declarations | Normalised AUC |
|---|---|---:|---:|---:|---:|
| V0 | random | 0.1253 [0.0685, 0.1898] | 0.2854 | 3.500 | 0.0000 |
| V0 | oracle | 0.7823 [0.7748, 0.7899] | 1.0000 | 0.000 | 1.0000 |
| V0 | B1 | 0.2001 [0.1364, 0.2693] | 0.3792 | 4.025 | 0.1139 |
| V0 | B2 | 0.3820 [0.2885, 0.4772] | 0.5000 | 2.900 | 0.3908 |
| V0 | B3 | 0.3999 [0.2914, 0.5060] | 0.5229 | 4.025 | 0.4180 |
| V0 | B4 | 0.4257 [0.3311, 0.5172] | 0.6146 | 4.300 | 0.4573 |
| V0 | B5 | 0.4670 [0.3716, 0.5594] | 0.6292 | 4.600 | 0.5201 |
| V1 | random | 0.3544 [0.2514, 0.4594] | 0.5208 | 1.975 | 0.0000 |
| V1 | oracle | 0.7823 [0.7748, 0.7899] | 1.0000 | 0.000 | 1.0000 |
| V1 | B1 | 0.3734 [0.2717, 0.4755] | 0.5458 | 1.300 | 0.0444 |
| V1 | B2 | 0.4550 [0.3373, 0.5737] | 0.5521 | 1.375 | 0.2350 |
| V1 | B3 | 0.6094 [0.5023, 0.7074] | 0.7375 | 2.600 | 0.5960 |
| V1 | B4 | 0.6413 [0.5514, 0.7258] | 0.8479 | 2.850 | 0.6706 |
| V1 | B5 | 0.6767 [0.5865, 0.7591] | 0.8500 | 2.575 | 0.7532 |
| V2 | random | 0.3677 [0.2617, 0.4743] | 0.5500 | 1.825 | 0.0000 |
| V2 | oracle | 0.7823 [0.7748, 0.7899] | 1.0000 | 0.000 | 1.0000 |
| V2 | B1 | 0.3737 [0.2731, 0.4741] | 0.5521 | 1.225 | 0.0145 |
| V2 | B2 | 0.4995 [0.3863, 0.6101] | 0.6396 | 1.575 | 0.3180 |
| V2 | B3 | 0.6203 [0.5133, 0.7187] | 0.7354 | 2.250 | 0.6093 |
| V2 | B4 | 0.6551 [0.5652, 0.7365] | 0.8438 | 2.600 | 0.6932 |
| V2 | B5 | 0.6716 [0.5813, 0.7525] | 0.8604 | 2.675 | 0.7330 |
| V3 | random | 0.5388 [0.4225, 0.6509] | 0.7417 | 1.850 | 0.0000 |
| V3 | oracle | 0.7823 [0.7748, 0.7899] | 1.0000 | 0.000 | 1.0000 |
| V3 | B1 | 0.5801 [0.4658, 0.6885] | 0.6937 | 1.225 | 0.1696 |
| V3 | B2 | 0.6077 [0.4859, 0.7222] | 0.6750 | 1.200 | 0.2830 |
| V3 | B3 | 0.7877 [0.7186, 0.8454] | 0.9417 | 1.875 | 1.0224 |
| V3 | B4 | 0.7919 [0.7124, 0.8590] | 0.9396 | 2.425 | 1.0394 |
| V3 | B5 | 0.8172 [0.7510, 0.8744] | 0.9729 | 2.325 | 1.1433 |

These are **B selection-set results**, not held-out performance. B1-B3 settings maximize mean family/variant AUC, breaking ties by fewer false declarations then canonical parameter JSON. B4/B5 parameters remain empty. The original `configs/policy/selected.yaml` remains unchanged; new choices are in `configs/policy/selected_live.yaml`. Per-family numbers, all B5 pairs and chosen settings are in the JSON report.

Intervals use 10,000 whole-world resamples per variant, seed 20261106, after sorting world IDs. The same draws serve all methods and paired changes. Intervals do not correct baseline-selection bias or shared imagery. Normalisation uses pooled new oracle/random means; the privileged L1 oracle is not a per-episode mathematical upper bound.

## Verification and reproduction

All 3,520 expected new episode keys are present once. World/cache/sensor/config identities, complete-grid metadata, finite metrics, budget limits, valid endings and AUC reconstructed from saved found curves passed. All 160 original B5 report checksums matched their saved selection metadata, and the original raw plan retained its ID. No historical episode was re-executed.

Oracle exceeds random in every family/variant: **True**, minimum family/variant gap 0.049426 AUC. Existing full-coverage detector sweeps were not repeated; sensing is unchanged.

Regenerate this report from the saved episodes:

```sh
uv run python scripts/search/report_live_baselines.py data/m12/baselines/634beeb0cc8f4664
```

The full new grid was launched detached from commit `65d724f`, using `uv run python scripts/search/run_baselines.py --search-config configs/env/search_live.yaml --out data/m12/baselines`. Its existing resume logic skips completed episodes. Raw records remain in ignored data/. Only new configs and reports are published. No config schema, frozen evaluation file, historical result, world-set entry or registry entry changed.

All earlier qualifications remain: M8's qualified acceptance, historical PPO below B5, and unsupported H3b. Phase 1 completion establishes a documented B-only timing model and retuned references, not an affirmative PPO finding. The next step, only after owner review, is the B5 warm start and V3 CPU pilot.
