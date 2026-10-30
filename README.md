# AltiSeek

**Learning where — and how high — to look for people at sea.**

A simulated drone searches open water for people within a fixed battery budget. Flying high covers
more sea per image but shrinks a person to a few pixels; flying low makes detection reliable but
costs far more battery. AltiSeek studies that tradeoff end to end:

1. a small detector fine-tuned on real drone footage ([SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/))
   turns simulated camera views into detections;
2. an explicit, altitude-aware sensor model turns detections, and missed detections, into Bayesian
   updates of a per-cell probability map;
3. a policy (classical search baselines vs. PPO) reads the map and chooses the next horizontal move,
   climb or descent;
4. planned M10 work will run the pipeline closed-loop in PX4 SITL + Gazebo through ROS 2.

Simulation only; no hardware, and no real-world performance claims.

The design, data contracts, milestones and decision log are in [CLAUDE.md](CLAUDE.md).

## Current results

M0-M9 are complete, with the documented M8 qualifications and M9 scope limits.
The frozen nominal V3 PPO achieves AUC 0.744 versus B5's 0.903. Re-look improves
PPO but remains below B5; detector/camera changes hurt frozen PPO, and distractors
increase false declarations. These are simulation findings, with no T-driven tuning.

See the [M9 acceptance review](docs/m9-extension-acceptance.md),
[main results](docs/m9-results/results.md),
[extension results](docs/m9-extension-results/results.md) and
[all-seed training report](docs/m9-training-results.md) for uncertainty,
limitations and reproduction commands. [M10 first flight](docs/m10-first-flight.md) passed;
[full M10 continuation](docs/m10-continuation.md) is authorized and its static camera gate
has passed. Live flight/search integration and transfer evaluation remain. M11 write-up remains.

## Setup

Requires [uv](https://docs.astral.sh/uv/) ≥ 0.13.4. Python 3.12 is used locally and in WSL; Kaggle uses Python 3.13 (see
[docs/env.md](docs/env.md)).

```bash
uv sync         # creates .venv with Python 3.12 and the pinned dependencies
uv run pytest   # unit tests, plus ruff lint and format checks
```

Datasets, rendered worlds, caches, checkpoints and run outputs live under `data/` and `runs/`, which
are never committed.

## Reproducing the data stage (M1)

Each command reads `configs/data/default.yaml`, accepts `key=value` overrides, and writes its outputs
and a metadata record (git commit, package versions, config, file hashes) under `data/processed/`.
Run them from the repository root, in order:

```bash
uv run python scripts/prepare_data.py   # fetch annotations if missing; frame and box tables
uv run python scripts/fit_camera.py     # focal length from the sea horizon (fetches ~120 frames)
uv run python scripts/make_splits.py    # capture-day sessions and the A/B/T split
uv run python scripts/scan_images.py    # one resumable pass over all 10,477 images (1-2 h)
uv run python scripts/check_leakage.py  # cross-split near-duplicate search; non-zero exit on failure
```

The data are [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/) Object Detection v2
(Varga et al., WACV 2022), released under CC0 1.0. Images are fetched on demand from the dataset's
public share, and only near-nadir frames are kept on disk. Verified dataset facts, the camera model,
the splits and the leakage-check result are in [docs/data_notes.md](docs/data_notes.md).

## Licence

AltiSeek is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only).
The detector is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
(`ultralytics==8.4.184`), which is itself AGPL-3.0. The SeaDronesSee dataset is CC0 1.0; its code is
MIT.

## Citation

If you use the data, cite the dataset paper:

> Leon Amadeus Varga, Benjamin Kiefer, Martin Messmer, Andreas Zell. *SeaDronesSee: A Maritime
> Benchmark for Detecting Humans in Open Water.* WACV 2022. [arXiv:2105.01922](https://arxiv.org/abs/2105.01922)
