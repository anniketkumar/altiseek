# Results

Result tables and the figure index (CLAUDE.md §6). Each number comes from the saved output of the
stage named with it, and each figure is that stage's own plot.

## Figures

| Figure | Stage | Shows |
|---|---|---|
| [horizon_fit_mavic_4k.png](figures/horizon_fit_mavic_4k.png) | fit-camera | `mavic_4k` focal length fitted from the sea horizon (D12) |
| [leakage_calibration.png](figures/leakage_calibration.png) | check-leakage | Near-duplicate thresholds and their calibration (D17) |
| [eval_vs_gsd.png](figures/eval_vs_gsd.png) | eval-detector | Swimmer recall, precision and false positives per view vs GSD, detector `78aba879cc50e025` |
| [level_design.png](figures/level_design.png) | design-levels | Split B recall vs GSD against the §8 recall limits: no feasible design |

## Detector `78aba879cc50e025` (M2)

- **Model:** YOLO11s initialised from `yolo11s.pt`, 640 px input, the five ODv2 classes with
  `swimmer` as the target (D19, D21, D22).
- **Provenance:** split `783b4403804291a2`, code commit `ae30740`. Registered in
  `registry/manifest.json`; the checkpoint is in the output of the Kaggle kernel
  `aniketkumarpradhan/altiseek-detector-v2`.
- **Training:** 7,149 train views (split A without session 2020-08-28) and 1,124
  internal-validation views (session 2020-08-28), on 2× T4. Early stopping ended training after 48
  epochs; the best epoch is 28 (internal-validation mAP50 0.650 over all classes, mAP50-95 0.366).
- **Operating threshold:** τ = 0.462, the F1-optimal swimmer threshold on internal-validation
  views.
- **Evaluation views:** real frames rectified to each GSD bin through the default camera pipeline;
  at most 120 frames per split and bin and 4 tiles per frame.
- **Matching:** AP50, recall and precision use IoU ≥ 0.5. The centre columns count a swimmer as
  found when a detection's centre lies within 1 m of the swimmer's centre. 1 m is 50 px at
  2 cm/px, so centre matching is lenient at fine GSDs.

### Swimmers by split, at τ

| Split | Views | Swimmers | AP50 | Recall | Precision | FP / view | Centre recall | Centre precision |
|---|---|---|---|---|---|---|---|---|
| val (A, 2020-08-28) | 1,825 | 1,969 | 0.850 | 0.864 | 0.857 | 0.155 | 0.920 | 0.912 |
| B | 1,207 | 1,741 | 0.580 | 0.558 | 0.715 | 0.321 | 0.727 | 0.932 |
| T | 636 | 298 | 0.883 | 0.839 | 0.912 | 0.038 | 0.889 | 0.967 |

### Swimmers by GSD bin, at τ

The evaluation has 12 bins from 2 to 30 cm/px; a bin without a row had no views. See
[Coarse GSDs are under-sampled](#coarse-gsds-are-under-sampled).

**B.** No views at 16, 20, 25 or 30 cm/px.

| GSD (cm/px) | Views | Swimmers | AP50 | Recall | Precision | FP / view | Centre recall |
|---|---|---|---|---|---|---|---|
| 2 | 157 | 118 | 0.638 | 0.458 | 0.794 | 0.089 | 0.534 |
| 3 | 283 | 268 | 0.626 | 0.541 | 0.732 | 0.187 | 0.679 |
| 4 | 288 | 317 | 0.627 | 0.596 | 0.730 | 0.243 | 0.763 |
| 5 | 202 | 303 | 0.603 | 0.617 | 0.736 | 0.332 | 0.795 |
| 6.5 | 163 | 412 | 0.574 | 0.566 | 0.706 | 0.595 | 0.750 |
| 8 | 58 | 128 | 0.490 | 0.469 | 0.612 | 0.655 | 0.719 |
| 10 | 43 | 138 | 0.511 | 0.543 | 0.676 | 0.837 | 0.732 |
| 13 | 13 | 57 | 0.525 | 0.491 | 0.700 | 0.923 | 0.632 |

**T.** The 20 cm/px row is a single view.

| GSD (cm/px) | Views | Swimmers | AP50 | Recall | Precision | FP / view | Centre recall |
|---|---|---|---|---|---|---|---|
| 2 | 219 | 65 | 0.808 | 0.738 | 0.828 | 0.046 | 0.800 |
| 3 | 155 | 73 | 0.889 | 0.808 | 0.952 | 0.019 | 0.849 |
| 4 | 97 | 45 | 0.981 | 0.911 | 0.953 | 0.021 | 0.933 |
| 5 | 66 | 35 | 0.962 | 0.971 | 0.944 | 0.030 | 1.000 |
| 6.5 | 46 | 29 | 0.938 | 0.931 | 0.931 | 0.043 | 0.966 |
| 8 | 28 | 24 | 0.898 | 0.875 | 0.955 | 0.036 | 0.917 |
| 10 | 16 | 15 | 0.823 | 0.800 | 0.857 | 0.125 | 0.933 |
| 13 | 8 | 10 | 0.720 | 0.800 | 0.800 | 0.250 | 1.000 |
| 20 | 1 | 2 | 0.505 | 0.000 | — | 0.000 | 0.000 |

**val** (internal validation, A).

| GSD (cm/px) | Views | Swimmers | AP50 | Recall | Precision | FP / view | Centre recall |
|---|---|---|---|---|---|---|---|
| 2 | 369 | 94 | 0.798 | 0.915 | 0.642 | 0.130 | 0.989 |
| 3 | 444 | 306 | 0.896 | 0.931 | 0.843 | 0.119 | 0.984 |
| 4 | 410 | 461 | 0.868 | 0.911 | 0.861 | 0.166 | 0.970 |
| 5 | 238 | 398 | 0.859 | 0.887 | 0.885 | 0.193 | 0.952 |
| 6.5 | 98 | 116 | 0.818 | 0.776 | 0.826 | 0.194 | 0.845 |
| 8 | 190 | 390 | 0.849 | 0.844 | 0.887 | 0.221 | 0.897 |
| 10 | 42 | 102 | 0.957 | 0.804 | 0.988 | 0.024 | 0.814 |
| 13 | 18 | 54 | 0.831 | 0.759 | 0.891 | 0.278 | 0.796 |
| 16 | 14 | 42 | 0.756 | 0.357 | 0.938 | 0.071 | 0.381 |
| 20 | 2 | 6 | 0.401 | 0.167 | 1.000 | 0.000 | 0.167 |

### Recall by swimmer size, at τ

Size is √(width × height) of the box in view pixels; swimmer counts in brackets.

| Swimmer size | val | B | T |
|---|---|---|---|
| 0–4 px | 0.00 (1) | — (0) | 0.00 (1) |
| 4–8 px | 0.25 (20) | 0.18 (45) | 0.54 (13) |
| 8–16 px | 0.71 (287) | 0.47 (570) | 0.89 (53) |
| 16–32 px | 0.89 (802) | 0.59 (885) | 0.85 (110) |
| 32+ px | 0.91 (859) | 0.71 (241) | 0.84 (121) |

### Coarse GSDs are under-sampled

- **Training views** per target GSD, rounded to 1 cm/px (make-views summary in the job log):

  | GSD (cm/px) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17–27 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Views | 554 | 1,393 | 1,212 | 1,150 | 997 | 768 | 619 | 525 | 363 | 240 | 177 | 81 | 57 | 55 | 42 | 40 |

  Only 40 of 8,273 views (0.5%) are coarser than 16.5 cm/px, where a log-uniform draw over
  2–30 cm/px puts at least 22% of draws. make-views keeps a draw only if one of its 20 random
  windows has at least half its area on usable source pixels (`min_coverage` 0.5). Most A frames
  are 20–60 m high and 45–75° off nadir. Their usable ground footprint cannot cover half of a
  window 100 m or more wide (16 cm/px × 640 px = 102 m), so make-views silently drops coarse draws.
- **Evaluation views:** eval-detector spreads each bin's 120 frames evenly over the split, whether
  or not a frame can serve that bin. A frame serves a bin only if one of its tiles reaches the same
  coverage. So B has no view coarser than 13 cm/px, T has one at 20 cm/px and val two.

### Observations

- **B is much harder than T or internal validation.**
  - At the same τ, recall is 0.56 on B, against 0.84 on T and 0.86 on val.
  - Centre matching closes part of the gap (B: recall 0.73, precision 0.93). Many B detections lie
    within 1 m of a swimmer, but their boxes miss IoU 0.5 against the annotation.
  - B, the single day 2021-08-21, has 6.0 swimmer boxes per `mavic_4k` frame, against 1.6 in T
    and 4.2 in A. It also has 995 of the 1,068 `mavic_4k` `life_saving_appliances` boxes.
  - The deficit holds at every object size: for 32+ px swimmers, recall is 0.71 on B against 0.84
    on T. So it is not only a small-object effect.
  - Per-view results are not saved yet, so the gap cannot be broken down by video, altitude or
    group size.
- **Recall dips at the finest GSD on B and T, but not on val.** At 2 cm/px, recall is 0.46 on B
  and 0.74 on T, against 0.62 and 0.97 at 5 cm/px; val has 0.91. The cause is not known yet.
- **Coarse GSDs are not measured on B,** and the detector saw almost no training views there.
  Val, the only split with views at 16 and 20 cm/px, falls to 0.36 and 0.17 there (14 and 2
  views).

## Level design, first run (M3)

`design-levels` on B's IoU recall curve above (`configs/levels/design.yaml`) finds **no feasible
design**, so D7 stays pending. The curve cannot meet either §8 recall rule:

- **Finest-level recall ≥ 0.8:** the best measured is 0.62, at 5 cm/px.
- **Coarsest-level recall ≤ 0.3:** the lowest measured is 0.46, at 2 cm/px, and the curve ends at
  13 cm/px. Recall is never extrapolated.
