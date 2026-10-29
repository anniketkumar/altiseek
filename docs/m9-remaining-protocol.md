# M9 remaining experiments (D46)

The owner requested completion of the remaining M9 scope on 2026-10-29 and
approved a separate B distractor stress test. Preserve accepted protocol v1.
This extension follows observation of v1 results; it implements the previously
listed experiments, not a search for settings that beat B5 on T. No further
tuning or checkpoint selection may use T or stress-test outcomes.

## Fixed CPU training design

V3, seeds 17/29/43, 131,072 steps each, D41 AUC reward, gamma 1, four workers,
two torch threads, unchanged CNN widths/optimizer. Validate every 16,384 steps
with three sampled action draws on B. Select the trained checkpoint with highest
mean AUC, earliest on ties. Raw AUC is common to all new cases; a fresh control
avoids confounding with historical family-normalised checkpoint selection.

| Case | Intended change |
|---|---|
| control | Fresh unchanged V3 control under the common selection rule |
| relook | Each distinct cached variant updates a cell/level once |
| declare | Seventh action declares the highest-belief unresolved cell in the current footprint |
| unknown_k | Initial belief assumes K=2; remove true K and found/K from observations |
| shaping | Add 0.01 times the potential difference; potential = negative sum of public belief entropy |
| egocentric | Centre the north-up map on the drone; remove absolute x/y channels |
| appearance | Train/validate policy on condition 0 only |
| appearance_control | Mixed-condition training matched to condition-0 family counts; same condition-0 validation |

These are **24 new runs**. Settings: `configs/env/ablation_*.yaml` and
`configs/policy/m9_*.yaml`. `scripts/run_m9_ablations.py` runs serially, resumes
saved checkpoints and never launches test evaluation. `--smoke` runs 1,024 steps
for seed 17 per case in separate directories. A suite lock rejects duplicates.

Operational contracts:

- Re-look is intentionally optimistic about independence, without sensor retuning.
  Reusing the same variant is uninformative. Coverage channels expose the fraction
  of distinct variants used at each level.
- Learned DECLARE costs one dwell (2 s), one step and corresponding hover energy.
  It resolves one cell without a new detector view; it is masked when no unresolved
  cell remains in the footprint. Moves/reset do not auto-declare. Found-target
  suppression remains active.
- Unknown K uses assumed count 2 in belief, count-vector entry zero, and found
  feedback scaled by the fixed max_targets bound. True K normalises reward/AUC
  only, never network input. The task still ends when all targets are found.
- AUC shaping is applied once in the training wrapper. Terminal potential is
  zero, including cap/battery endings; its sum telescopes to an initial-state
  constant. Report unshaped AUC and false declarations separately.
- Egocentric maps retain every cell without wrapping/cropping, on a 239x239 padded
  canvas centred on the drone, north up. The vector and six move meanings remain
  fixed. CNN widths/adaptive pooling stay fixed; report the larger input's CPU cost.

## Appearance holdout

Condition 0 has 54 B training and 13 B validation worlds. The mixed control draws
the same number per family from all 160 B training worlds, with RNG seed 20261029.
Both select on the same 13 condition-0 validation worlds. T has 69 condition-0
and 11 condition-1 worlds; report the latter's low statistical power.

This is a **policy-training appearance holdout**: the detector/sensor remain
frozen, and sensor fitting previously used both conditions. Do not claim the
whole perception pipeline never saw condition 1. Glint remains excluded because
its rendering failed fidelity (D30).

## GPU robustness artifacts

Bulk inference runs on cloud GPU only. Preserve original artifacts and validate
new contracts/IDs and live inference before use.

1. Detector swap: checkpoint `78aba879cc50e025` at
   `data/kaggle/altiseek-detector-v2/detector/train/weights/best.pt`; cache 40 B
   sensor-fit and 80 T worlds. Refit V0–V3 on B with the existing procedure and
   threshold, without detector training or threshold tuning.
2. Camera: original detector, 320x320 instead of 640x640, same HFOV/physical
   footprints/lattice/levels/worlds/battery. Cache the same 40+80 worlds and refit
   the sensor on B. Change actual image sampling and detector input, never merely
   relabel old detections.
3. Distractors: 40 new paired B scenarios, 10 per prior/target family, seeds from
   5,000,000 and disjoint from existing sets. Pair 0–3 versus 30–40 distractors
   while retaining background, targets, prior and appearance. Generate 80 caches
   with the original detector/camera; keep the original sensor to measure shift.
   No B imagery enters T. This is B stress evidence, not T generalisation.

The owner reports 7 h 4 min GPU allowance. Planned 320 additional worlds are
approximately the previous production scale. Pilot the new cases and bound
sessions with headroom for validation/saving. Resume partial output after the
weekly quota refresh if necessary; never silently reduce the experiment or run
bulk inference locally to meet a deadline.

## Evaluation and completion

Before further test/stress evaluation, freeze selected checkpoints, generated
artifact IDs and a machine-readable extension protocol in git. Use v1's paired
environment/action seeds and world bootstrap procedure. Report all training
seeds. Compare ablations with the fresh control and appearance with its matched
control, splitting seen/unseen appearance. For detector/camera shifts, evaluate
all frozen baseline variants and original selected PPO seeds, explicitly tied
to each refitted sensor contract.

Report AUC, found fraction, success, false declarations, censored first find,
energy and altitude use, plus paired intervals. Seven-action/egocentric policies
need matching adapters; test those before held-out runs. All results remain
separate from v1. Complete M9 only when every experiment above has verified
saved results and reproducible tables/figures. The absence of T distractors
remains explicit after the approved B stress test.
