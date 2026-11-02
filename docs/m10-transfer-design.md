# M10 transfer design — fixed before new T outcomes

Select 24 existing T worlds: three from each of the eight original families,
ranked within family by the hash of `["m10-transfer-selection-v1", world_id]`.
Selection reads world metadata only. Original T remains distractor-free; family
low/high labels do not establish a distractor-density contrast.

Run V3 B5 and all three frozen D41 PPO seeds (17/29/43), with one paired action
seed per policy/world. Repeat all 96 cases with the original sensor and the B
Gazebo-refitted sensor: **192 live episodes**. Use D52's measured 500-second
energy budget and normal seeded L5 edge launch. No policy retraining or selection
is permitted. Method/sensor case order is deterministically shuffled by identity
to reduce time-of-run confounding. Report every seed; bootstrap whole worlds
after averaging PPO seeds, with paired original/refit/cached comparisons.

Build 96 matching cached references with the original sensor and identical
environment/action seeds, using the shared evaluator. These are distinct M10
references, not restarts or replacements of M9. Keep shared time-based AUC,
found fraction, first-find time, declarations and path metrics. Report measured
energy, simulator time and wall time separately. The cached renderer and live
sensor differ in appearance, actual pose, whole-cell visibility and timing; the
aggregate deployment gap does not isolate any one cause.

Before any T execution: verify the complete D52 calibration collection and fit,
freeze the exact original/refit model and policy hashes, pass the current full
regression suite, and pass one B live refit smoke for B5 and each PPO seed. The
original all-seed pilots and full-budget B run remain prerequisites. All manifest
and source identities must remain unchanged during evaluation. Acquisition or
execution failure stops progress, preserving completed cases; it does not count
as a poor policy result or get silently dropped.

Report level-wise detector outcomes on the intersection of live whole-cell
visibility and the nominal cached footprint for each executed view, using cached
variant 0 as a fixed diagnostic reference. These repeated observations are
clustered by world and do not constitute independent samples. Ground truth is
available only to verification and reporting, never policy inputs.

After evaluation, produce numeric tables, paired intervals and a deterministic
demo from the first selected T world/B5/original case, independent of its result.
The demo combines a top-down map, footprint/belief and camera inset; it remains
local because it contains dataset imagery. Report calibration limitations and
negative policy findings. M10 completion requires successful artifact/integrity
checks and the final results review; a finished acquisition job alone is not M10
acceptance.
