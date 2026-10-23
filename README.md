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
