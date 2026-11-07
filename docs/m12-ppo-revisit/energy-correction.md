# M12 Phase 1: B-fitted energy correction

D63 analysis `5796952a3a671506`, computed 2026-11-07 with code commit `6d33b9a6fdc06dff77b1836f3e714ea8d2e4412b` under the [predeclared protocol](energy-protocol.md).

**Extra elapsed time: 6.42 s per view (95% whole-flight bootstrap interval 3.31 to 7.16 s).** The candidate dwell is **8.42 s**, replacing 2 s only in the new candidate config. Speeds, climb multiplier and the 500 s battery are unchanged.

**Coverage gate: failed; owner decision required.** Full L5 coverage costs 135.78 s (27.156% of the battery), against the existing 25% / 125 s limit. Baseline retuning has not started. No B5 AUC change has been measured; `selected_live.yaml` has not been created. Phase 1 is incomplete.

## Primary estimate

22 complete transitions from five measured-search flights on two B rl-val worlds. The point estimate is the median of flight medians, so the 15-transition PPO flight does not outweigh the shorter flights. 10,000 bootstrap resamples use seed 20261106. The interval is conditional on these few records, not a deployment-wide uncertainty guarantee.

| Flight | Policy | Transitions | Median overhead (s) | Initial measured time (s) | Median energy residual (hover-s) |
|---|---|---:|---:|---:|---:|
| 20261031T043133Z | B5 | 2 | 6.490 | 5.080 | 6.750 |
| 20261102T090615Z | B5 | 1 | 7.160 | 4.640 | 7.430 |
| 20261102T090914Z | PPO 17 | 2 | 3.840 | 2.560 | 3.950 |
| 20261102T091144Z | PPO 29 | 2 | 3.310 | 3.060 | 3.410 |
| 20261102T091411Z | PPO 43 | 15 | 6.420 | 4.520 | 6.660 |

Intervals use post-inference `measured.elapsed_s`, subtracting the unchanged nominal move plus 2 s dwell for the same commanded ENU displacement. Initial observations, takeoff, landing and incomplete moves are excluded. The initial measured times are all below the candidate shared dwell, so applying it at reset is a conservative timing approximation on these five cases. Energy residuals are descriptive only; the estimator fits elapsed time, not measured climb-energy excess.

## Coverage at the unchanged battery

Uses the existing full-lattice lawnmower route, starting at each level's own altitude. The existing `coarse_share: 0.25` in `configs/levels/design.yaml` resolves the handoff's qualitative word 'cheap'. No new threshold was selected.

| Setting | Dwell (s) | L1 coverage (s) | L5 coverage (s) | L5 share | L5 ≤125 s |
|---|---:|---:|---:|---:|---|
| original | 2.000 | 542.000 | 78.000 | 15.600% | yes |
| estimate | 8.420 | 1318.820 | 135.780 | 27.156% | no |
| ci_low | 5.310 | 942.505 | 107.790 | 21.558% | yes |
| ci_high | 9.160 | 1408.360 | 142.440 | 28.488% | no |

L1 remains infeasible at every reported setting. The L5 interval crosses the constraint, but the predeclared point-estimate gate fails. The fit was not lowered to meet the constraint. Keeping the 25% rule would require at least 543.12 s battery at the point estimate; alternatively the owner could retain 500 s and explicitly amend the coarse-coverage rule. Neither change has been made or evaluated.

## Separate calibration diagnostics

Verified 8,800 accepted views on 40 sensor-fit B worlds in acquisition `cf148ad98784906d`. There are 8,752 within-flight capture intervals; 48 first captures are excluded. Accepted views from subsequently failed flights remain included. No interval bridges flights. Every view's NPZ checksum, key, contract, B identity and capture limits were checked; no image was decoded or inference rerun.

Capture intervals contain previous-frame processing and saving, but omit current-frame inference and search-policy decisions. They are not pooled into the primary estimate. These summaries use all transitions arriving at a level, including climbs:

| Destination | Transitions | Median capture-interval overhead (s) |
|---|---:|---:|
| L1 | 4,792 | 6.000 |
| L2 | 1,960 | 6.250 |
| L3 | 1,000 | 6.500 |
| L4 | 640 | 6.000 |
| L5 | 360 | 6.000 |

| Destination | Motion | Horizontal distance (m) | Intervals | Median overhead (s) |
|---|---|---:|---:|---:|
| L1 | same-level | 25.000 | 4,791 | 6.000 |
| L1 | same-level | 353.553 | 1 | 7.145 |
| L2 | climb | 295.011 | 1 | 1.533 |
| L2 | climb | 335.876 | 40 | 0.947 |
| L2 | same-level | 37.500 | 1,918 | 6.250 |
| L2 | same-level | 318.198 | 1 | 5.680 |
| L3 | climb | 246.539 | 1 | 1.381 |
| L3 | climb | 300.520 | 40 | 0.982 |
| L3 | same-level | 50.000 | 958 | 6.500 |
| L3 | same-level | 282.843 | 1 | 5.716 |
| L4 | climb | 206.155 | 1 | 3.412 |
| L4 | climb | 268.701 | 40 | 1.657 |
| L4 | same-level | 60.000 | 598 | 6.500 |
| L4 | same-level | 180.000 | 1 | 5.500 |
| L5 | climb | 165.680 | 40 | -0.027 |
| L5 | climb | 167.705 | 1 | 0.771 |
| L5 | same-level | 75.000 | 318 | 6.000 |
| L5 | same-level | 212.132 | 1 | 5.287 |

Regular same-level strides of 25 to 75 m have median residuals of 6 to 6.5 s, without a monotonic increase with distance. Level and stride length are confounded, and the rare longer moves have little replication. These data do not isolate a distance-dependent settling term. Cross-level residuals are much smaller, sometimes negative; simultaneous horizontal/vertical motion versus the nominal model's summed travel times is a possible contributor, not an isolated causal finding. No distance term or schema field was added. A constant dwell remains an approximation, particularly for level changes; richer modeling needs a new declared protocol.

## Reproduction and preservation

From the repository root in WSL (the first command only reads saved B sources):

```sh
/home/aniket/.venvs/altiseek/bin/python scripts/search/fit_live_energy.py --native-root /home/aniket/altiseek-m10
uv run python scripts/search/report_live_energy.py data/m12/energy/summary.json
```

The JSON report carries source checksums, the analysis ID, per-world numeric diagnostics and flight/controller/failure provenance. Raw timing rows stay in ignored `data/m12/energy/`. No schema or historical implementation changed; no frozen config, world-set entry, registry entry or historical result changed. The candidate config is not referenced by any existing run config. Historical artifacts retain their verification context at `results-final`.

No T source, test world, test seed or M10 transfer-results file selected a parameter. No completed experiment was rerun. The coverage stop is required by handoff Phase 1 step 4 and the approved D63 protocol; owner authorization of a revised constraint or battery is needed before the baseline retune.
