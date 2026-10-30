# M9 training review

All 24 planned runs completed their 131,072-step budget.
Checkpoint selection uses B only, raw mean AUC and earliest ties. These are selection-set
scores, not held-out performance. Appearance cases use only 13 validation worlds;
their scores are not directly comparable with the 40-world scores of other cases.

| Case | Seed | Selected step | Selected AUC | Final AUC | Zero-p AUC | Steps/s |
|---|---|---|---|---|---|---|
| control | 17 | 114688 | 0.743 | 0.660 | 0.715 | 76.5 |
| control | 29 | 81920 | 0.730 | 0.730 | 0.697 | 93.9 |
| control | 43 | 131072 | 0.737 | 0.737 | 0.709 | 79.6 |
| relook | 17 | 114688 | 0.830 | 0.821 | 0.798 | 75.7 |
| relook | 29 | 65536 | 0.834 | 0.830 | 0.787 | 61.5 |
| relook | 43 | 65536 | 0.834 | 0.790 | 0.780 | 63.9 |
| declare | 17 | 65536 | 0.642 | 0.438 | 0.649 | 76.8 |
| declare | 29 | 131072 | 0.614 | 0.614 | 0.596 | 95.3 |
| declare | 43 | 131072 | 0.660 | 0.660 | 0.665 | 93.4 |
| unknown_k | 17 | 32768 | 0.740 | 0.678 | 0.675 | 74.4 |
| unknown_k | 29 | 98304 | 0.740 | 0.705 | 0.724 | 83.9 |
| unknown_k | 43 | 114688 | 0.746 | 0.710 | 0.708 | 83.5 |
| shaping | 17 | 65536 | 0.729 | 0.660 | 0.711 | 76.3 |
| shaping | 29 | 49152 | 0.698 | 0.673 | 0.685 | 83.1 |
| shaping | 43 | 114688 | 0.744 | 0.721 | 0.651 | 58.8 |
| egocentric | 17 | 98304 | 0.779 | 0.768 | 0.748 | 27.6 |
| egocentric | 29 | 131072 | 0.762 | 0.762 | 0.724 | 28.6 |
| egocentric | 43 | 65536 | 0.756 | 0.754 | 0.744 | 20.7 |
| appearance | 17 | 32768 | 0.761 | 0.640 | 0.651 | 90.5 |
| appearance | 29 | 114688 | 0.705 | 0.658 | 0.616 | 90.3 |
| appearance | 43 | 131072 | 0.690 | 0.690 | 0.563 | 90.3 |
| appearance_control | 17 | 65536 | 0.775 | 0.632 | 0.623 | 87.5 |
| appearance_control | 29 | 114688 | 0.727 | 0.701 | 0.690 | 87.7 |
| appearance_control | 43 | 32768 | 0.721 | 0.712 | 0.606 | 86.9 |

## Interpretation limits

Several curves fluctuate and final performance can fall below the selected checkpoint.
Completion of the fixed budget does not establish stable convergence. Report every seed,
retain B-selected checkpoints, and do not extend or reselect after viewing T results.

Belief removal is a diagnostic: a small or negative paired effect does not establish
leakage or prove the policy ignores all spatial input. Prior, coverage and verified
feedback remain available (D42). The input-access and adapter checks passed before T.

The re-look configuration assumes independent evidence across distinct variants; higher
B AUC is not evidence that this assumption holds in real sensing. Learned DECLARE also
changes action cost and verification behavior as predeclared, so it is a task ablation.

Reproduce: `uv run python scripts/report_m9_training.py`.
