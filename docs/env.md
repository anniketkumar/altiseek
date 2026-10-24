# Environment

## Python: 3.12 and 3.13 (D9)

`pyproject.toml` sets `requires-python = ">=3.12,<3.14"`, and CI tests both versions. A single
version is not possible: ROS 2 Jazzy fixes 3.12 for the ROS nodes, and Kaggle runs 3.13. Checked
on 2026-10-24:

| Environment | Python | Evidence |
|---|---|---|
| Local development (Windows 11) | 3.12.10 | `python --version`; the venv uses this interpreter |
| Kaggle (cloud GPU, D20) | **3.13.15** | Probe kernel `altiseek-env-probe`. The image builds `FROM` Colab's runtime of 2026-09-17 ([Dockerfile.tmpl](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)), which is newer than the 3.12-based runtime 2026.07 in the [Colab FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) |
| WSL2: ROS 2 / PX4 / Gazebo | 3.12 (Ubuntu 24.04 system Python) | [PX4 ROS 2 user guide](https://docs.px4.io/main/en/ros2/user_guide): "The supported and recommended ROS 2 platform for working with PX4 is ROS 2 "Jazzy" LTS on Ubuntu 24.04", with Gazebo Harmonic. rclpy comes from the system Python |

Adding or dropping a Python version means a Decision Log entry and a change to `requires-python`.

## Kaggle (D20)

Measured by the probe kernels on 2026-10-24:

| Item | Value |
|---|---|
| GPU (accelerator `NvidiaTeslaT4`) | 2× Tesla T4, 15.6 GB each |
| CPU / RAM | 4 cores / 33.7 GB |
| Disk | `/kaggle/working` (saved output) 20.9 GB free; `/tmp` 1.1 TB, not kept |
| Preinstalled | torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.1.3, Pillow 13.0.0, matplotlib 3.10.0, pydantic 2.13.5, PyYAML 6.0.3, OpenCV 4.14.0. No ultralytics |
| Internet | Works (the account is phone-verified) |
| Dataset share | About 10 images/s with 16 parallel requests; about 1/s for a single connection |

- Kernels are pushed, monitored and collected with the Kaggle CLI 2.2.5, installed as a uv tool
  (`uv tool install kaggle==2.2.5`).
- Authentication is a `KGAT_` API token in `~/.kaggle/access_token`, outside the repository. Never
  commit or print it.
- The free GPU quota is about 30 h per week, and one session lasts at most 12 h, so long jobs must
  checkpoint and resume (CLAUDE.md §2).
- A kernel attached as a source (`kernel_sources`) is mounted at
  `/kaggle/input/notebooks/<user>/<slug>/`, holding that kernel's saved output (seen 2026-10-24).
- `kaggle kernels logs -f` prints plain text while a kernel runs. The log saved with its output
  (`<slug>.log`) is a JSON list of `{"stream_name", "time", "data"}` records.
- The CLI cannot cancel a running kernel. Pushing again starts a new version alongside it.

## Dependencies

Direct dependencies are pinned exactly in `pyproject.toml`, the single place they are declared.
`uv.lock` is generated from it and pins the transitive dependencies, for Windows and Linux only.

- **Pins follow Kaggle.** Packages that Kaggle preinstalls are pinned to Kaggle's versions (D13), so
  installing this package in a kernel keeps them.
- **CPU-only PyTorch locally and in CI.** `[tool.uv.sources]` takes torch and torchvision from the
  PyTorch CPU index (`2.11.0+cpu`). On Kaggle, the preinstalled `2.11.0+cu128` satisfies the same
  `==2.11.0` pin.
- **Detector:** Ultralytics 8.4.184 (AGPL-3.0, D19).

## Tooling

| Tool | Version | Notes |
|---|---|---|
| uv | 0.13.5 | `[tool.uv] required-version = ">=0.13.4"`: older releases trust a compromised CA ([GHSA-hq85-5cpj-mr62](https://github.com/advisories/GHSA-hq85-5cpj-mr62)). Upgrade with `uv self update` |
| hatchling | 1.32.4 | Build backend |
| ruff | 0.17.2 | Lint and format. `tests/test_style.py` runs both, so `pytest` fails on style errors |
| pytest | 9.1.1 | Config in `pyproject.toml` |
| kaggle | 2.2.5 | uv tool, not a project dependency |

CI (`.github/workflows/ci.yml`) runs `uv sync --locked` and `uv run pytest` on Ubuntu 24.04 with
Python 3.12 and 3.13, and on Windows with Python 3.12.

## Local setup

```bash
uv sync         # .venv with Python 3.12, the pinned dependencies and the dev group
uv run pytest
```

## Machine state

- No local GPU. Detector training and cache generation run on Kaggle (CLAUDE.md §2).
- WSL2 is not installed on the development machine yet. It is needed from M10 (Ubuntu 24.04,
  ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL). Installing it needs administrator rights and a reboot.
