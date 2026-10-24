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
4. the same pipeline runs closed-loop in PX4 SITL + Gazebo through ROS 2.

Simulation only; no hardware, and no real-world performance claims.

The design, data contracts, milestones and decision log are in [CLAUDE.md](CLAUDE.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) ≥ 0.13.4. Python 3.12 is used everywhere (see
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
