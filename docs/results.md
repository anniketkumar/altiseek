# Results

Result tables and the figure index (CLAUDE.md §6). Each number comes from the saved output of the
stage named with it, and each figure is that stage's own plot.

## Figures

| Figure | Stage | Shows |
|---|---|---|
| [horizon_fit_mavic_4k.png](figures/horizon_fit_mavic_4k.png) | fit-camera | `mavic_4k` focal length fitted from the sea horizon (D12) |
| [leakage_calibration.png](figures/leakage_calibration.png) | check-leakage | Near-duplicate thresholds and their calibration (D17) |
| [eval_vs_gsd_700e0ff3_d25.png](figures/eval_vs_gsd_700e0ff3_d25.png) | eval-detector | Swimmer recall, precision and false positives per valid view vs GSD, detector `700e0ff3455259e8`, 5% tiles (D25) |
| [level_design_700e0ff3_d25.png](figures/level_design_700e0ff3_d25.png) | design-levels | B centre recall (D25) against the §8 limits: no feasible design |
| [level_design_700e0ff3_d25_c065.png](figures/level_design_700e0ff3_d25_c065.png) | design-levels | The same curve with coarsest ≤ 0.65: best-ranked design |
| [level_design_700e0ff3_d25_a360.png](figures/level_design_700e0ff3_d25_a360.png) | design-levels | Area 360 m, coarsest ≤ 0.6: best-ranked design |
| [level_design_700e0ff3_d25_r512.png](figures/level_design_700e0ff3_d25_r512.png) | design-levels | R = 512, coarsest ≤ 0.6: best-ranked design |
| [eval_vs_gsd_700e0ff3_d23.png](figures/eval_vs_gsd_700e0ff3_d23.png) | eval-detector | Swimmer recall, precision and false positives per valid view vs GSD, detector `700e0ff3455259e8`, 25% tiles (D23) |
| [eval_vs_gsd_78aba879_d23.png](figures/eval_vs_gsd_78aba879_d23.png) | eval-detector | The same for detector `78aba879cc50e025`, re-evaluated under D23 |
| [eval_vs_gsd_78aba879_d21.png](figures/eval_vs_gsd_78aba879_d21.png) | eval-detector | `78aba879cc50e025` under the superseded D21 evaluation |
| [level_design_700e0ff3_iou.png](figures/level_design_700e0ff3_iou.png) | design-levels | B IoU recall of `700e0ff3455259e8` against the §8 limits |
| [level_design_700e0ff3_centre.png](figures/level_design_700e0ff3_centre.png) | design-levels | B centre recall of `700e0ff3455259e8` against the §8 limits |
| [level_design_78aba879_d21.png](figures/level_design_78aba879_d21.png) | design-levels | B IoU recall of `78aba879cc50e025` under D21 |

## Detectors (M2)

Both are YOLO11s initialised from `yolo11s.pt`, with 640 px input and the five ODv2 classes,
`swimmer` being the target (D19, D22). Both were trained on split A of split `783b4403804291a2`
on 2× T4 and are registered in `registry/manifest.json`.

| Detector | Kaggle run | Code | Training views | Training | Internal validation, best epoch | τ |
|---|---|---|---|---|---|---|
| `78aba879cc50e025` | `altiseek-detector-v2` | `ae30740` | D21: 7,149 train, 1,124 val; 0.5% coarser than 16.5 cm/px | Stopped early after 48 epochs; best 28 | mAP50 0.650, mAP50-95 0.366 | 0.349 |
| `700e0ff3455259e8` | `altiseek-detector-v3` | `9ad5c57` | D23: 6,653 train, 1,178 val, stratified by GSD | Stopped early after 55 epochs; best 35 | mAP50 0.613, mAP50-95 0.343 | 0.376 |

- **Internal validation:** each detector's mAP is over all classes on its own validation views, so
  the two are not comparable.
- **τ:** the F1-optimal swimmer threshold on D23's internal-validation evaluation views.
- **`700e0ff3455259e8` training views per GSD bin, rendered / planned** (make-views summary in the
  job log). A bin plans fewer views when few frames can serve it, at most 3 views per frame:

  | Bin (cm/px) | 2.0–2.5 | 2.5–3.1 | 3.1–3.9 | 3.9–4.9 | 4.9–6.2 | 6.2–7.7 | 7.7–9.7 | 9.7–12.2 | 12.2–15.2 | 15.2–19.1 | 19.1–23.9 | 23.9–30.0 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Train | 600/600 | 600/600 | 600/600 | 600/600 | 600/600 | 600/600 | 600/600 | 600/600 | 600/600 | 596/600 | 600/600 | 57/57 |
  | Val | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 78/78 |

  19% of the training views are coarser than 15 cm/px. Only 19 training frames can serve the top
  bin.

## Evaluation (D23)

Both detectors are evaluated on the same views: 12 GSD bins over 2–30 cm/px. Per split and bin,
up to 120 of the frames that can serve the bin are used, with up to 4 tiles each. A tile needs at
least 25% valid pixels; the rest is fill.

- **Matching:** recall, precision and AP50 use IoU ≥ 0.5. Centre recall and centre precision match
  a detection whose centre is within 1 m of the swimmer's centre. 1 m is 50 px at 2 cm/px but only
  3.3 px at 30 cm/px.
- **False positives** are per full view of valid pixels. Valid views add up each view's valid
  share.
- **Arrows** read `78aba879cc50e025` → `700e0ff3455259e8`.

### By split

| Split | Detector | Views | Valid views | Swimmers | AP50 | Recall | Centre recall | Precision | Centre precision | FP / valid view |
|---|---|---|---|---|---|---|---|---|---|---|
| val | `78aba879cc50e025` | 3,207 | 1,620 | 4,442 | 0.74 | 0.77 | 0.84 | 0.79 | 0.86 | 0.57 |
| val | `700e0ff3455259e8` | 3,207 | 1,620 | 4,442 | 0.81 | 0.83 | 0.89 | 0.87 | 0.93 | 0.34 |
| B | `78aba879cc50e025` | 3,250 | 1,630 | 7,679 | 0.37 | 0.51 | 0.83 | 0.53 | 0.87 | 2.11 |
| B | `700e0ff3455259e8` | 3,250 | 1,630 | 7,679 | 0.42 | 0.53 | 0.81 | 0.61 | 0.92 | 1.62 |
| T | `78aba879cc50e025` | 2,467 | 1,188 | 1,718 | 0.71 | 0.78 | 0.92 | 0.79 | 0.94 | 0.29 |
| T | `700e0ff3455259e8` | 2,467 | 1,188 | 1,718 | 0.74 | 0.79 | 0.92 | 0.83 | 0.97 | 0.23 |

### By GSD bin

**B**

| GSD (cm/px) | Views (valid) | Swimmers | Recall | Centre recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 405 (221) | 163 | 0.56 → 0.55 | 0.69 → 0.64 | 0.73 → 0.80 | 0.15 → 0.10 | 0.63 → 0.60 |
| 3 | 449 (284) | 379 | 0.63 → 0.60 | 0.82 → 0.78 | 0.67 → 0.72 | 0.41 → 0.31 | 0.63 → 0.58 |
| 4 | 437 (257) | 511 | 0.71 → 0.64 | 0.90 → 0.84 | 0.71 → 0.70 | 0.59 → 0.54 | 0.65 → 0.61 |
| 5 | 404 (203) | 509 | 0.71 → 0.61 | 0.91 → 0.78 | 0.71 → 0.73 | 0.72 → 0.56 | 0.65 → 0.59 |
| 6.5 | 319 (181) | 656 | 0.70 → 0.65 | 0.93 → 0.88 | 0.68 → 0.69 | 1.19 → 1.08 | 0.59 → 0.57 |
| 8 | 297 (141) | 733 | 0.67 → 0.65 | 0.92 → 0.88 | 0.65 → 0.70 | 1.83 → 1.47 | 0.53 → 0.56 |
| 10 | 256 (101) | 723 | 0.61 → 0.61 | 0.88 → 0.86 | 0.61 → 0.65 | 2.75 → 2.38 | 0.46 → 0.49 |
| 13 | 165 (67) | 755 | 0.56 → 0.60 | 0.83 → 0.85 | 0.61 → 0.65 | 4.03 → 3.63 | 0.40 → 0.49 |
| 16 | 167 (64) | 753 | 0.49 → 0.52 | 0.82 → 0.82 | 0.52 → 0.60 | 5.32 → 4.12 | 0.30 → 0.39 |
| 20 | 157 (49) | 823 | 0.37 → 0.47 | 0.81 → 0.81 | 0.38 → 0.55 | 10.45 → 6.48 | 0.18 → 0.34 |
| 25 | 114 (40) | 971 | 0.29 → 0.36 | 0.76 → 0.77 | 0.31 → 0.44 | 15.36 → 11.25 | 0.10 → 0.22 |
| 30 | 80 (22) | 703 | 0.12 → 0.23 | 0.64 → 0.66 | 0.14 → 0.31 | 22.92 → 16.42 | 0.02 → 0.11 |

**T**

| GSD (cm/px) | Views (valid) | Swimmers | Recall | Centre recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 400 (219) | 81 | 0.81 → 0.72 | 0.89 → 0.80 | 0.81 → 0.83 | 0.07 → 0.05 | 0.80 → 0.75 |
| 3 | 317 (172) | 101 | 0.80 → 0.74 | 0.93 → 0.84 | 0.78 → 0.85 | 0.13 → 0.08 | 0.74 → 0.80 |
| 4 | 306 (160) | 128 | 0.86 → 0.84 | 0.95 → 0.95 | 0.81 → 0.82 | 0.16 → 0.14 | 0.85 → 0.83 |
| 5 | 295 (144) | 167 | 0.85 → 0.80 | 0.99 → 0.96 | 0.77 → 0.78 | 0.29 → 0.26 | 0.79 → 0.76 |
| 6.5 | 213 (102) | 164 | 0.85 → 0.82 | 0.99 → 0.98 | 0.80 → 0.81 | 0.33 → 0.31 | 0.78 → 0.81 |
| 8 | 249 (113) | 192 | 0.88 → 0.86 | 1.00 → 0.97 | 0.84 → 0.86 | 0.27 → 0.25 | 0.83 → 0.78 |
| 10 | 232 (107) | 200 | 0.91 → 0.90 | 0.99 → 0.99 | 0.88 → 0.89 | 0.23 → 0.21 | 0.85 → 0.84 |
| 13 | 169 (73) | 225 | 0.87 → 0.86 | 1.00 → 0.98 | 0.85 → 0.87 | 0.48 → 0.39 | 0.82 → 0.81 |
| 16 | 164 (58) | 236 | 0.76 → 0.81 | 0.95 → 0.93 | 0.74 → 0.87 | 1.08 → 0.50 | 0.66 → 0.79 |
| 20 | 76 (27) | 132 | 0.48 → 0.64 | 0.76 → 0.87 | 0.63 → 0.73 | 1.40 → 1.14 | 0.38 → 0.57 |
| 25 | 34 (11) | 68 | 0.18 → 0.35 | 0.35 → 0.54 | 0.50 → 0.65 | 1.10 → 1.19 | 0.16 → 0.36 |
| 30 | 12 (3) | 24 | 0.04 → 0.08 | 0.12 → 0.21 | 0.33 → 0.40 | 0.66 → 0.99 | 0.02 → 0.15 |

**val** (internal validation, A)

| GSD (cm/px) | Views (valid) | Swimmers | Recall | Centre recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 480 (269) | 106 | 0.87 → 0.90 | 0.95 → 0.94 | 0.46 → 0.70 | 0.39 → 0.15 | 0.69 → 0.87 |
| 3 | 480 (279) | 224 | 0.92 → 0.91 | 0.99 → 0.99 | 0.74 → 0.81 | 0.26 → 0.17 | 0.87 → 0.89 |
| 4 | 480 (271) | 339 | 0.96 → 0.96 | 0.99 → 0.99 | 0.88 → 0.91 | 0.17 → 0.13 | 0.95 → 0.96 |
| 5 | 458 (232) | 417 | 0.89 → 0.89 | 0.97 → 0.95 | 0.84 → 0.87 | 0.31 → 0.23 | 0.85 → 0.86 |
| 6.5 | 385 (187) | 524 | 0.88 → 0.87 | 0.95 → 0.94 | 0.82 → 0.87 | 0.55 → 0.36 | 0.83 → 0.84 |
| 8 | 274 (136) | 539 | 0.89 → 0.85 | 0.95 → 0.91 | 0.85 → 0.87 | 0.60 → 0.50 | 0.85 → 0.83 |
| 10 | 231 (94) | 580 | 0.86 → 0.86 | 0.92 → 0.92 | 0.88 → 0.87 | 0.76 → 0.81 | 0.83 → 0.82 |
| 13 | 151 (60) | 581 | 0.82 → 0.85 | 0.90 → 0.90 | 0.86 → 0.92 | 1.32 → 0.73 | 0.81 → 0.86 |
| 16 | 140 (47) | 573 | 0.72 → 0.83 | 0.85 → 0.89 | 0.77 → 0.90 | 2.57 → 1.18 | 0.68 → 0.81 |
| 20 | 66 (25) | 252 | 0.39 → 0.61 | 0.44 → 0.63 | 0.70 → 0.94 | 1.74 → 0.36 | 0.42 → 0.71 |
| 25 | 40 (13) | 175 | 0.11 → 0.59 | 0.12 → 0.62 | 0.22 → 0.91 | 5.33 → 0.77 | 0.12 → 0.66 |
| 30 | 22 (8) | 132 | 0.01 → 0.38 | 0.01 → 0.57 | 0.01 → 0.52 | 8.53 → 5.94 | 0.02 → 0.28 |

### By swimmer size

Recall (IoU) at τ. Size is √(width × height) of the box in view pixels; swimmer counts in brackets.

| Swimmer size | val | B | T |
|---|---|---|---|
| 0–4 px | 0.00 → 0.03 (32) | 0.00 → 0.06 (777) | 0.00 → 0.08 (90) |
| 4–8 px | 0.30 → 0.59 (656) | 0.39 → 0.49 (2,944) | 0.67 → 0.77 (442) |
| 8–16 px | 0.82 → 0.85 (1,542) | 0.66 → 0.64 (2,225) | 0.90 → 0.87 (599) |
| 16–32 px | 0.89 → 0.88 (1,478) | 0.73 → 0.66 (1,432) | 0.85 → 0.83 (420) |
| 32+ px | 0.91 → 0.91 (734) | 0.74 → 0.68 (301) | 0.86 → 0.79 (167) |

### Observations

- **The retrain gained at coarse GSDs and lost a little at fine ones.**
  - At 25 cm/px, recall rose from 0.11 to 0.59 on val, from 0.18 to 0.35 on T and from 0.29 to
    0.36 on B. Recall of 4–8 px swimmers rose on every split.
  - Recall of swimmers 16 px and larger fell on B and T, and B's centre recall at 5 cm/px fell from
    0.91 to 0.78.
  - D23 spends fewer views on fine GSDs: about 2,100 of `700e0ff3455259e8`'s views are finer than
    3.9 cm/px, against about 3,200 of `78aba879cc50e025`'s below 4.5 cm/px.
- **On B, centre recall stays at 0.64 or higher across 2–30 cm/px for both detectors,** while IoU
  recall falls to 0.12–0.23 at 30 cm/px. At coarse GSDs a swimmer is 2–3 px wide, where IoU is
  unstable (§9).
- **val and T lose recall at coarse GSDs much faster than B, but each split's coarse end is one
  scene.**
  - Only frames flown high can fill a coarse view at 25% coverage, and few videos were flown high.
  - Share of each bin's swimmers that come from its largest video (number of videos in brackets):

    | Split | 2 cm/px | 5 | 10 | 16 | 20 | 25 | 30 |
    |---|---|---|---|---|---|---|---|
    | B | 82% (4) | 32% (4) | 34% (4) | 71% (3) | 84% (2) | 98% (2) | 100% (1) |
    | T | 37% (4) | 50% (4) | 100% (2) | 100% (1) | 100% (1) | 100% (1) | 100% (1) |
    | val | 94% (2) | 88% (2) | 80% (2) | 73% (2) | 100% (1) | 100% (1) | 100% (1) |

  - B's coarse end is `NorthSea_Sat_DJI_0129`, about 88 m up; T's is `NorthSea_Thu_DJI_0107`.
  - Tight groups do not explain B's flat curve. B's swimmers are less clustered than T's: the
    median distance to the nearest other swimmer is 4.0 m in B against 2.6 m in T
    ([data notes](data_notes.md#swimmer-spacing)).
- **IoU and centre matching disagree in every B and T video, for both detectors,** which points at
  the annotations rather than at either model ([data notes](data_notes.md#annotation-offsets)). In
  image 10465 every box in the frame is offset from its swimmer by about half a box.

## Evaluation with 5% tiles (D25)

Detector `700e0ff3455259e8` again, with evaluation tiles needing only 5% valid pixels, so frames
flown lower can serve the coarse bins (Kaggle run `altiseek-eval-coarse`, commit `349ad86`). τ is
0.377. The frames per bin and tiles per frame are as in D23.

| Split | Views | Valid views | Swimmers | AP50 | Recall | Centre recall | Precision | Centre precision | FP / valid view |
|---|---|---|---|---|---|---|---|---|---|
| val | 3,450 | 1,565 | 5,556 | 0.73 | 0.76 | 0.87 | 0.81 | 0.93 | 0.64 |
| B | 3,161 | 1,351 | 6,895 | 0.38 | 0.47 | 0.74 | 0.59 | 0.92 | 1.69 |
| T | 2,433 | 830 | 1,741 | 0.61 | 0.60 | 0.73 | 0.79 | 0.96 | 0.34 |

**B**

| GSD (cm/px) | Views (valid) | Swimmers | Centre recall | Recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 443 (213) | 132 | 0.64 | 0.45 | 0.59 | 0.19 | 0.42 |
| 3 | 440 (261) | 337 | 0.72 | 0.58 | 0.76 | 0.23 | 0.61 |
| 4 | 410 (204) | 417 | 0.78 | 0.57 | 0.67 | 0.57 | 0.54 |
| 5 | 393 (183) | 495 | 0.80 | 0.59 | 0.69 | 0.73 | 0.55 |
| 6.5 | 300 (152) | 590 | 0.84 | 0.64 | 0.71 | 1.02 | 0.57 |
| 8 | 277 (117) | 621 | 0.87 | 0.64 | 0.69 | 1.55 | 0.56 |
| 10 | 233 (77) | 666 | 0.82 | 0.59 | 0.67 | 2.46 | 0.52 |
| 13 | 158 (52) | 716 | 0.81 | 0.51 | 0.58 | 5.02 | 0.38 |
| 16 | 137 (36) | 703 | 0.75 | 0.47 | 0.59 | 6.41 | 0.38 |
| 20 | 130 (26) | 729 | 0.70 | 0.41 | 0.55 | 9.56 | 0.32 |
| 25 | 120 (17) | 743 | 0.62 | 0.28 | 0.41 | 17.05 | 0.18 |
| 30 | 120 (13) | 746 | 0.56 | 0.14 | 0.22 | 29.62 | 0.06 |

**T**

| GSD (cm/px) | Views (valid) | Swimmers | Centre recall | Recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 424 (197) | 67 | 0.78 | 0.66 | 0.76 | 0.07 | 0.62 |
| 3 | 340 (156) | 89 | 0.91 | 0.83 | 0.89 | 0.06 | 0.83 |
| 4 | 284 (121) | 106 | 0.94 | 0.87 | 0.84 | 0.14 | 0.83 |
| 5 | 262 (91) | 101 | 0.90 | 0.79 | 0.82 | 0.20 | 0.78 |
| 6.5 | 178 (69) | 121 | 0.98 | 0.83 | 0.79 | 0.38 | 0.80 |
| 8 | 163 (49) | 124 | 0.95 | 0.83 | 0.83 | 0.43 | 0.80 |
| 10 | 151 (38) | 141 | 0.93 | 0.76 | 0.76 | 0.90 | 0.72 |
| 13 | 133 (28) | 162 | 0.75 | 0.63 | 0.81 | 0.87 | 0.70 |
| 16 | 132 (23) | 181 | 0.62 | 0.52 | 0.82 | 0.87 | 0.62 |
| 20 | 126 (26) | 195 | 0.87 | 0.72 | 0.82 | 1.19 | 0.68 |
| 25 | 120 (18) | 216 | 0.56 | 0.38 | 0.69 | 2.01 | 0.41 |
| 30 | 120 (16) | 238 | 0.27 | 0.13 | 0.50 | 2.06 | 0.18 |

**val** (internal validation, A)

| GSD (cm/px) | Views (valid) | Swimmers | Centre recall | Recall | Precision | FP / valid view | AP50 |
|---|---|---|---|---|---|---|---|
| 2 | 480 (267) | 115 | 0.95 | 0.90 | 0.72 | 0.15 | 0.89 |
| 3 | 480 (268) | 212 | 0.99 | 0.91 | 0.81 | 0.17 | 0.88 |
| 4 | 480 (232) | 263 | 0.98 | 0.95 | 0.89 | 0.13 | 0.95 |
| 5 | 458 (229) | 414 | 0.95 | 0.89 | 0.87 | 0.24 | 0.86 |
| 6.5 | 386 (186) | 516 | 0.93 | 0.87 | 0.87 | 0.35 | 0.84 |
| 8 | 274 (135) | 539 | 0.91 | 0.85 | 0.87 | 0.50 | 0.83 |
| 10 | 231 (93) | 580 | 0.92 | 0.86 | 0.87 | 0.81 | 0.82 |
| 13 | 151 (58) | 586 | 0.88 | 0.80 | 0.88 | 1.11 | 0.81 |
| 16 | 135 (40) | 586 | 0.82 | 0.75 | 0.86 | 1.71 | 0.75 |
| 20 | 130 (26) | 586 | 0.79 | 0.70 | 0.82 | 3.41 | 0.69 |
| 25 | 125 (16) | 586 | 0.78 | 0.58 | 0.69 | 9.40 | 0.47 |
| 30 | 120 (13) | 573 | 0.75 | 0.39 | 0.47 | 18.77 | 0.22 |

### Videos and coverage

Swimmers per bin on B, from `views.csv`. Centre / IoU recall on tiles with at least 25% valid
pixels and on the rest, swimmers in brackets:

| GSD (cm/px) | Tiles ≥ 25% valid | Tiles < 25% valid | Videos | Largest video |
|---|---|---|---|---|
| 2 | 0.65 / 0.45 (128) | 0.50 / 0.50 (4) | 4 | 85% |
| 3 | 0.73 / 0.60 (328) | 0.22 / 0.11 (9) | 4 | 48% |
| 4 | 0.82 / 0.59 (390) | 0.26 / 0.19 (27) | 4 | 36% |
| 5 | 0.81 / 0.61 (467) | 0.64 / 0.29 (28) | 4 | 35% |
| 6.5 | 0.87 / 0.67 (530) | 0.57 / 0.40 (60) | 4 | 29% |
| 8 | 0.90 / 0.68 (534) | 0.67 / 0.43 (87) | 4 | 30% |
| 10 | 0.85 / 0.62 (505) | 0.73 / 0.52 (161) | 4 | 28% |
| 13 | 0.84 / 0.54 (533) | 0.71 / 0.40 (183) | 4 | 29% |
| 16 | 0.83 / 0.53 (253) | 0.70 / 0.44 (450) | 4 | 30% |
| 20 | 0.82 / 0.49 (224) | 0.65 / 0.37 (505) | 4 | 35% |
| 25 | 0.76 / 0.42 (106) | 0.60 / 0.25 (637) | 4 | 36% |
| 30 | 0.62 / 0.22 (97) | 0.56 / 0.13 (649) | 4 | 36% |

- **Every B bin from 4 to 30 cm/px now draws on all four videos,** none supplying more than 36%.
  Under D23, one video supplied 98–100% of the swimmers at 25–30 cm/px.
- **Low-coverage tiles score lower in aggregate, but not within a video.** The two kinds of tile come
  from different frames: full coarse tiles mostly from the one high-altitude video, low-coverage
  tiles from all four. Within single videos the difference goes both ways, −0.46 to +0.11 in
  centre recall at 10–30 cm/px; the largest gaps rest on 12–17 swimmers. Weighted by swimmers, it
  averages about −0.03.
- **Scenes differ a lot at coarse GSDs.** Centre recall at 30 cm/px is 0.36 in
  `NorthSea_Sat_DJI_0131`, 0.40 in `_0126`, 0.62 in `_0122` and 0.62–0.69 in `_0129`.
- **T's coarse bins are still one video** (`NorthSea_Thu_DJI_0107` above 16 cm/px). T only
  reports results and plays no part in level design.
- **FP per valid view is noisy at coarse GSDs:** at 30 cm/px, B's 120 views add up to 13 full views
  of valid pixels.

## Level design (M3)

All runs read split B. Unless stated, the settings are `configs/levels/design.yaml`: R = 640,
HFOV 80°, area 300 m, finest level ≥ 0.8 and coarsest ≤ 0.3. D24 settles the measure: centre
recall, from `700e0ff3455259e8`. D7 is still the owner's choice (CLAUDE.md §25).

| Detector | Evaluation | Recall | Rules changed | Result |
|---|---|---|---|---|
| `78aba879cc50e025` | D21 | IoU | — | None. Finest ≥ 0.8: best 0.62 at 5 cm/px. Coarsest ≤ 0.3: lowest 0.46, curve ends at 13 cm/px |
| `78aba879cc50e025` | D23 | IoU | — | None. Finest ≥ 0.8: best 0.71 at 4 cm/px |
| `78aba879cc50e025` | D23 | centre | — | None. Coarsest ≤ 0.3: lowest 0.64 at 30 cm/px |
| `700e0ff3455259e8` | D23 | IoU | — | None. Finest ≥ 0.8: best 0.65 at 8 cm/px |
| `700e0ff3455259e8` | D23 | centre | — | None. Coarsest ≤ 0.3: lowest 0.64 at 2 cm/px |
| `700e0ff3455259e8` | D25 | centre | — | None. Coarsest ≤ 0.3: lowest 0.56 at 30 cm/px |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.4 or 0.5 | None, for the same reason |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.6 | None. Recall reaches 0.6 only past about 26.5 cm/px, i.e. footprints of 170–192 m, and no footprint in that range tiles 300 m in whole cells |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.65 | 76 designs |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.6, area 360 m | 79 designs |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.6, R = 512 | 176 designs |
| `700e0ff3455259e8` | D25 | centre | coarsest ≤ 0.6, R = 480 | None (footprints and battery) |

**Candidate designs** on the D25 curve. "Rank" is design-levels' ranking: smallest largest recall
step first, then fewest finest-level views. Each design is listed as its levels, footprint at
GSD → centre recall:

| Rules changed | Rank | Cell | Battery | Levels |
|---|---|---|---|---|
| coarsest ≤ 0.65 | 1 | 2.5 m | 364 s | 37.5 m at 5.9 cm/px → 0.82; 100 m at 15.6 → 0.75; 120 m at 18.8 → 0.71; 150 m at 23.4 → 0.65 |
| coarsest ≤ 0.65 | 5 | 2.5 m | 542 s | 50 m at 7.8 cm/px → 0.87; 75 m at 11.7 → 0.81; 100 m at 15.6 → 0.75; 120 m at 18.8 → 0.71; 150 m at 23.4 → 0.65 |
| coarsest ≤ 0.6, area 360 m | 1 | 2.5 m | 380 s | 80 m at 12.5 cm/px → 0.81; 120 m at 18.8 → 0.71; 180 m at 28.1 → 0.59 |
| coarsest ≤ 0.6, area 360 m | 4 | 5 m | 411 s | 45 m at 7.0 cm/px → 0.85; 60 m at 9.4 → 0.83; 120 m at 18.8 → 0.71; 180 m at 28.1 → 0.59 |
| coarsest ≤ 0.6, R = 512 | 1 | 2.5 m | 402 s | 60 m at 11.7 cm/px → 0.81; 75 m at 14.6 → 0.77; 100 m at 19.5 → 0.70; 120 m at 23.4 → 0.65; 150 m at 29.3 → 0.57 |
| coarsest ≤ 0.6, R = 512 | 5 | 2.5 m | 600 s | 30 m at 5.9 cm/px → 0.82; 75 m at 14.6 → 0.77; 100 m at 19.5 → 0.70; 120 m at 23.4 → 0.65; 150 m at 29.3 → 0.57 |

- **The ranking can skip the detector's best GSDs.** Several top-ranked designs start at
  11.7–12.5 cm/px, although recall peaks at 8 cm/px (0.87), because evenly spaced recall steps
  rank first.
- **Every design spans a narrow recall range:** finest and coarsest level differ by at most about
  0.26, against the 0.5 (0.8 → 0.3) that §8 asks for. False positives also rise with GSD (B, IoU matching: 1.6 per valid view at 8 cm/px,
  6.4 at 16), so coarse levels give weaker evidence than recall alone shows.
- **A level table chosen now should be re-checked at M6,** when r_ℓ and f_ℓ are measured on rendered
  sensor-fit worlds.

## Superseded: `78aba879cc50e025` under D21

The first evaluation used D21's views. Its overall swimmer numbers at τ = 0.462 were:

| Split | Views | Swimmers | AP50 | Recall | Precision | FP / view | Centre recall | Centre precision |
|---|---|---|---|---|---|---|---|---|
| val | 1,825 | 1,969 | 0.850 | 0.864 | 0.857 | 0.155 | 0.920 | 0.912 |
| B | 1,207 | 1,741 | 0.580 | 0.558 | 0.715 | 0.321 | 0.727 | 0.932 |
| T | 636 | 298 | 0.883 | 0.839 | 0.912 | 0.038 | 0.889 | 0.967 |

It did not cover the coarse GSDs, which is why D23 replaced it:

- **Training views** per target GSD, rounded to 1 cm/px (make-views summary in the job log):

  | GSD (cm/px) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17–27 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Views | 554 | 1,393 | 1,212 | 1,150 | 997 | 768 | 619 | 525 | 363 | 240 | 177 | 81 | 57 | 55 | 42 | 40 |

  Only 40 of 8,273 views (0.5%) were coarser than 16.5 cm/px; a log-uniform draw over 2–30 cm/px
  puts at least 22% of draws there. make-views kept a draw only if one of its 20 random windows had
  at least half its area on usable source pixels (`min_coverage` 0.5), so it silently dropped
  coarse draws.
- **Evaluation views:** eval-detector spread each bin's 120 frames over the whole split, whether or
  not a frame could serve that bin. B had no view coarser than 13 cm/px.
