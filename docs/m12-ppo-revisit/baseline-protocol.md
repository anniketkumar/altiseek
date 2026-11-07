# M12 Phase 1 continuation: coverage amendment and baseline retune (D65)

Declared 2026-11-07 before running or inspecting the corrected-cost baseline grid.
The owner authorized relaxing the coverage rule and delegated the implementation.
This authorizes completion of Phase 1; the Phase 2 owner stop remains in force.

## Coverage amendment

For M12 only, full L5 lattice coverage may cost at most **30% of the unchanged
500 s battery** (150 s). Retain the D63 fitted dwell exactly,
`8.420000000000016` s, and the original speeds and climb multiplier. Full L1
coverage must still exceed 500 s. Settings are in new
`configs/levels/design_live.yaml` and `configs/env/search_live.yaml`.

This is an explicit post-B-analysis design relaxation. The D63 point estimate
costs 27.156% at L5 and its upper overhead-CI endpoint costs 28.488%; 30% is a
round allowance covering both while preserving most of the budget for subsequent
search. The original 25% gate failed and remains failed in the D63 report.
Do not describe D65 as passing the original constraint or as new evidence that a
policy improves. The extra allowance changes no episode dynamics; keeping the
battery fixed isolates the timing correction in the historical comparison.

The owner's desired affirmative outcome does not impose a positive AUC acceptance
criterion. Complete the declared work and report all measured differences,
including decreases, with no outcome-driven changes to dwell, seeds, grid or
selection rule. No T outcomes inform this amendment.

## Fixed evaluation and selection

Use the unchanged `scripts/search/run_baselines.py` with
`--search-config configs/env/search_live.yaml --out data/m12/baselines`.
The existing resumable runner saves each episode atomically; run one detached CPU
process. Never rerun a completed old-cost or new-cost episode.

Use all 40 existing rl-val B worlds, sensor `fcf8cdae58ca3947`, all V0–V3,
and the unchanged `configs/policy/baselines.yaml` grid: 22 method/parameter choices
per world and variant, **3,520 episodes**. Reset and policy seeds remain those
declared in M7. B1–B3 are selected per family/variant by mean raw AUC, then lower
mean false declarations, then canonical parameter JSON. B4/B5 have empty
parameters. Random/oracle run as references. No new grid or sensor fit is allowed.

Verify the complete expected key set, identities, cache checksums, B split and
world set, finite metrics, 500 s budget and episode endings. The oracle must
exceed random in every family/variant under the existing sanity rule; failures
stop selection for review rather than changing the rule. Existing full-coverage
detector sweeps are not repeated; sensor and detector settings are unchanged.

Report every method/variant mean AUC, found fraction and false declarations,
and per-family AUC. Report all B5 paired AUC changes relative to saved original
run `8cd76bdb72f965a1`, never executing that run again. Match world, variant,
method, parameters and reset seed; validate the old completion/selection metadata
checksums and raw saved plan identity. Do not reserialize the old plan with
newer schema defaults to assign it a new ID. Historical source verification
retains the `results-final` context.

For each variant sort by world ID, generate 10,000 whole-world bootstrap draws
with a fresh `default_rng(20261106)`, and use the same draw matrix for new method
means and paired B5 changes. Percentile 95% intervals use linear quantiles.
Report the same resampled means for the saved B5 reference. Normalised AUC uses
the new pooled oracle/random means; if their difference is nonpositive, report
undefined. These are B selection-set results; the intervals do not correct
selection bias or shared imagery. All four variants are reported irrespective
of sign; V3 is the main reference for the next phase.

## Outputs and stop

Write `configs/policy/selected_live.yaml` only after the grid and sanity checks
pass. Keep the original `selected.yaml`, energy config, evaluation protocols,
world sets, historical results and all schemas unchanged. Publish
`baseline-results.json` and `baseline-results.md` here, including numerical
provenance and the retained D63 coverage failure. Raw episode logs stay in
ignored `data/m12/`.

Run `uv run python -m pytest`, commit and push explicit file lists without
attribution. Stop for owner review of completed Phase 1: the unchanged overhead
estimate and CI, the explicit coverage amendment, and B5's measured AUC change.
No imitation, PPO training, test generation, T evaluation or new flight starts.
D64 remains reserved for the later evaluation freeze; the next new decision is D66.
