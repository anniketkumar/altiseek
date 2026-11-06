# M12 Phase 1 energy protocol (D63)

Declared 2026-11-06, before computing overhead. The owner approved the proposed
five-flight estimator and Phase 1 scope. This is a new B-only experiment; it does
not replace M7–M10 results. No PPO training, new flight, test-world access, or
Phase 2 work is authorized by this protocol.

## Primary source records

Native root: `/home/aniket/altiseek-m10`. Reports are
`camera-flights/<flight>/search-summary.json`. Verify completion manifests,
checksums, method/seed/world, measured accounting, normal launch, successful
completion and cleanup. Verify the two world specs are split B, rl-val, with
their original content IDs. No source is chosen by overhead or target outcomes.

| Flight | World | Method | Completion manifest relative to native root | Report SHA-256 |
|---|---|---|---|---|
| 20261031T043133Z | 003b0c54f581c65b | B5 | continuations/cf148ad98784906d/full-budget-B5.json | 9b0fbdc0aca9f856480c449071f99e6993c08017ec65f6d0489e35385b9fe9ca |
| 20261102T090615Z | 48a9325328a6d2c4 | B5 | transfers/5560516d08907341/preflight/571e79a898ec1412.json | 45e44581ef38524c8bd187d26570bac6f79679da8754a5264a75b617e625eb69 |
| 20261102T090914Z | 48a9325328a6d2c4 | PPO 17 | transfers/5560516d08907341/preflight/9bb74a29561c4edc.json | fff15dbf6ccce7d166aa8260fbbdb507797739cbe859a335c88aaaa1bc6c5b0c |
| 20261102T091144Z | 48a9325328a6d2c4 | PPO 29 | transfers/5560516d08907341/preflight/8655d0f0800235ea.json | 07615968f3c1e1e2a7d197af36578f17a3c939f5c7b6fa0798485683e17ce97c |
| 20261102T091411Z | 48a9325328a6d2c4 | PPO 43 | transfers/5560516d08907341/preflight/c5550a88e5516884.json | bc33fcde522c9fa5d9cdc57ef1ef0580f50c4f1decf42a2acd8aeed29d35d188 |

The four preflights use the frozen refit sensor; the earlier B5 integration uses
its recorded original sensor. This affects paths, not the timing formula. Exclude
the short 25 s integration report and superseded v1–v3 preflights.

## Estimator and implementation choice

For consecutive accepted views j-1 and j within a search flight, use
`dt = measured.elapsed_s[j] - measured.elapsed_s[j-1]`: cumulative simulator-time
snapshots after inference, including the intervening policy decision. The
per-view top-level `elapsed_s` is wall-clock inference time and must not replace
this field. For commanded ENU goal displacement, use the unchanged energy model:

`overhead_s = dt - (hypot(dx, dy) / 10 + abs(dz) / 3 + 2)`.

Use every complete consecutive transition, including level changes; retain
negative residuals. Exclude initial observations, takeoff/initial positioning,
landing and transitions without accepted views (including budget expiry). Missing,
non-finite or non-monotone required records stop analysis rather than being
silently dropped. Report initial-view measured time minus 2 s separately. Report
energy residuals using `measured.energy_s` and nominal move energy descriptively;
they do not select dwell.

Estimate the median of the five within-flight median overheads, giving each
flight equal weight. NumPy's ordinary median averages the middle pair when even.
The 95% percentile interval resamples five whole flight medians with replacement,
10,000 draws with `default_rng(20261106)` and default linear 0.025/0.975 quantiles.
Do not bootstrap transitions as independent observations.

Candidate dwell is exactly `2 + estimate`, without fit-time rounding, clipping,
multiplicative scaling, speed changes, climb-multiplier changes or battery changes.
A nonpositive correction or dwell stops for review. Computation and evaluation
use CPU. This CI is conditional on five flights on two B worlds, not a calibrated
interval for deployment across new worlds, machines or policies. Repeated worlds
and short B5 flights limit interpretation. Extra time is approximated as hover
energy; no measured-climb model is fitted. Applying the same dwell at reset is
an explicit approximation assessed by the initial-view diagnostic.

## Separate calibration diagnostics

Use accepted metadata from acquisition `cf148ad98784906d`, contract
`0e81d39c16e288b9`, under `continuations/cf148ad98784906d/views/` for all 40
sensor-fit B worlds (`5dc45d0e1ea5afbd`, 8,800 views). Verify acquisition/contract
IDs, world order, B identities, view keys and checksums. No imagery is read and
no detector inference or sensor refit is performed.

Group by `flight_run`, order by capture `stamp_s`, and difference only consecutive
accepted views in that flight, using commanded displacement and the same nominal
formula. Include accepted records from later-failed flights, retaining failure
and controller provenance. Never bridge a restart. Exclude first captures, which
have no preceding interval. Pilot and resumed nonlocal moves remain visible in
separate motion/distance summaries.

Capture intervals include previous-frame processing/saving, not current-frame
inference or search-policy decisions. Do not pool them into the primary estimate.
Report counts and median residuals by destination level, same-level/climb/descent,
and horizontal distance (grouped at 1e-6 m precision), plus per-world medians.
No calibration CI, fitted slope or automatic level-dependent model is declared.
Level and stride length are confounded. If diagnostics require another functional
model, stop and declare an amended estimator before computing that alternative.

## Coverage and baseline retune

Source inspection resolved the preliminary ambiguity about "cheap": the existing
`configs/levels/design.yaml` has `coarse_share: 0.25`, enforced by
`test_chosen_levels_pass_the_coverage_cost_check`. Use that existing rule.
Compute full lattice lawnmower energy starting at each level's own altitude with
`coverage_energy_s`. L1 must exceed 500 s; L5 must cost at most 125 s. Report point
and CI-endpoint costs. The point estimate governs the gate; endpoint crossings
are reported as uncertainty. If it fails, save candidate settings and diagnostics
and stop before retuning. Ask before changing battery or the design constraint;
do not reduce the overhead to pass.

If coverage passes, use new `configs/env/energy_live.yaml` and
`configs/env/search_live.yaml`, original sensor `fcf8cdae58ca3947`, all 40 rl-val
B worlds and V0–V3. Reuse `configs/policy/baselines.yaml` unchanged, including
seeds and grid. Run random/oracle and B1–B5 under new costs. Select B1–B3 per
family/variant by mean AUC, then fewer false declarations, then canonical
parameter JSON as in M7. B4/B5 parameters remain empty. Write only the new
`configs/policy/selected_live.yaml` and M12 reports.

Compare B5 AUC to saved M7 run `8cd76bdb72f965a1`, pairing world, variant and reset
seed; never rerun old-cost episodes. Report each variant's mean AUC and paired
change, with 95% percentile CIs from 10,000 whole-world resamples (seed 20261106,
fresh generator per variant, sorted world IDs). These are B selection-set results.
No AUC result changes the overhead or candidate grid. Stop with overhead, CI,
coverage and B5 AUC change for owner review before Phase 2. D64 is reserved for
the later evaluation freeze.

## Preservation and delivery

No schema change is planned. Keep existing configs, world entries and historical
results byte-identical. Any necessary schema change requires historical resolved
config and artifact-ID regression proof; verify historical artifacts at
`results-final`. Use explicit staging, no attribution trailers or data/cache/image
commits. Run `uv run python -m pytest`; commit and push tested increments.
Baseline on 2026-11-06: 378 passed, 2 skipped (Windows lacks `fcntl`), including
ruff format/lint. AGENTS.md's earlier 391 count is not the current Windows count.
