# M9 existing-cache phase: protocol v1

This phase evaluates frozen M7 baselines and all six selected D41 PPO policies.
The owner approved the existing-cache phase first. It covers E1, E2 and prior
misspecification in E3; it does not complete every experiment or ablation in M9.
No training, model selection or sensor fitting uses these results.

`configs/eval/protocol_v1.yaml` declares scenarios and RNG settings.
`configs/eval/artifacts_v1.json` pins the six checkpoint paths and SHA-256 values,
the B selection report, the frozen baseline settings and all 80 test world IDs
and seeds. These numerical identifiers contain no dataset imagery. Freeze this
protocol in git before main T evaluation. Changes after T evaluation require a
new version and Decision Log entry, with every method rerun.

Checkpoint hashes are byte SHA-256 values. The selection report and resolved
baseline config use canonical content IDs, avoiding CRLF/LF differences between
Windows and Linux checkouts. Each run separately records its actual input bytes.

## Comparisons

Run random, oracle and B1–B5 with V0–V3. Run PPO with V0 and V3, retaining seeds
17, 29 and 43 individually. Baselines use the existing per-family/per-variant
settings without tuning for new batteries or priors. PPO samples actions from
the masked categorical distribution, using three independent seeded draws per
world/checkpoint. Random also has three draws; deterministic baselines and the
oracle have one. This gives 54 episodes per world/scenario, 25,920 in total.

Every method uses the same launch/variant seed, `environment_seed + world.seed`.
Action draw r uses `action_seed + 1000000*r + world.seed`. Repeats change the
policy RNG only. Navigation, first-look rule, known K, north-up coordinates,
declaration/suppression settings and the 300-step cap remain fixed. Models run
on CPU in evaluation mode. No argmax checkpoint reselection is allowed.

## Scenarios

| Name | Battery (s) | Supplied-prior change |
|---|---:|---|
| nominal | 500 | none |
| battery_tight | 250 | none |
| battery_generous | 750 | none |
| prior_shift | 500 | each component mean moves 50 m east |
| prior_rotation | 500 | each ellipse rotates 90° about its own mean |
| prior_spread | 500 | both component standard deviations multiply by 2 |

Scenario choices precede M9 T outcomes. Changes are one factor at a time, with
unchanged mixture weights. The supplied prior is truncated and renormalised on
the original area; means are not clipped. Targets, true priors, world specs and
cache IDs remain unchanged. The override is applied before the initial view,
to both the public q map and belief initialisation.

## Metrics and uncertainty

Report found fraction, success, AUC, time to first find censored at the budget,
false declarations, energy, path length, per-level time and false clears.
Include the initial observation in AUC and false-clear diagnostics. Preserve
the old step-only diagnostic as a separate field rather than rewriting history.
Use the existing elapsed-time AUC convention, normalised by the scenario battery,
carrying the final found fraction to the horizon. Battery remains energy-based.

Average action repeats within each world before any bootstrap. Report each PPO
training seed, then mean and sample standard deviation across the three seeds.
For aggregate PPO comparisons, average training seeds within each world; pair
worlds against each baseline. Use 2,000 whole-world bootstrap draws and percentile
95% intervals, with the declared bootstrap seed. These intervals describe world
sampling conditional on the three trained policies; seed spread is separate.
Treat intervals as descriptive, without claims from multiple significance tests.

Report pooled results and four actual T families (sharp/diffuse × single/group),
20 worlds each. Preserve the eight original tags for provenance, but do not
interpret low/high as distractor-density contrasts. T has no distractors (D43).
Normalised AUC uses random/oracle means for the same scenario, variant and group;
nonpositive denominators are undefined. The oracle is a reference, not an upper
bound. Report paired PPO-minus-B1–B5 AUC and V1/V2/V3-minus-V0 baseline AUC.
Use paired nominal-versus-perturbed comparisons for robustness.

Figures and tables must regenerate from saved episode results: found-over-time,
battery sweep, normalised scores, prior robustness and altitude usage versus B3.
Preserve all seeds, failed hypotheses and D42's qualified M8 acceptance.

## Execution gates and deferred work

First pass unit tests, frozen-artifact checks and a B-only smoke run. Checkpoint
hashes and cache metadata must match. Save resolved config, code hashes, package
versions and git provenance. Write each episode atomically with its job identity
and checksum; interrupted jobs resume without overwriting completed episodes.
Only launch the main T evaluation from the committed, tested implementation.

Run with `uv run python scripts/evaluate.py`. `--smoke` selects one B validation
world and tests all policies on nominal conditions plus each perturbation with
V3 random. `--limit N` stops after N newly completed episodes; rerun the same
command without the limit to continue. A run lock rejects concurrent writers.
`runs/eval/m9-v1/<run_id>/status.json` reports completion. The run identity includes
source/input hashes, resolved settings and numerical package versions.

Regenerate all numeric tables and figures with
`uv run python scripts/report_evaluation.py runs/eval/m9-v1/<run_id>`.
The report refuses incomplete or smoke results and verifies every episode.
On Windows, use the WSL command prefix documented in `docs/env.md`.

Preflight on 2026-10-29: 290 full-suite tests passed in 52.85 s. Final B smoke
`64717ac37c981377` completed 59/59 episodes. Resume was also exercised with an
earlier B smoke, stopping at five saved episodes and continuing to completion.
`scripts/continue_evaluation.py` runs the full phase and automatically writes the
report under the ignored run directory, then marks it `awaiting_review`.

Detector/camera changes, new appearance conditions, training-based ablations and
other re-look/declare/count/map/reward experiments remain deferred. Distractor
robustness is untested on T. Imperfect calibration, small world counts, static
composited worlds and the simulation-only scope remain explicit limitations.
