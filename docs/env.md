# Environment

## Python: 3.12 everywhere

`pyproject.toml` sets `requires-python = ">=3.12,<3.13"`. Each environment was checked on
2026-10-24:

| Environment | Python | Evidence |
|---|---|---|
| Local development (Windows 11) | 3.12.10 | `python --version`; the venv uses this interpreter |
| Google Colab | 3.12.13 | [Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html): latest pinnable runtime 2026.07 = Python 3.12.13, PyTorch 2.11.0, numpy 2.0.2 |
| Kaggle | same as Colab | [Kaggle/docker-python `Dockerfile.tmpl`](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl) builds `FROM us-docker.pkg.dev/colab-images/public/runtime:release-colab-external-images_20260917-060051_RC00`. Confirm the exact version in the first Kaggle session (M2) |
| WSL2: ROS 2 / PX4 / Gazebo | 3.12 (Ubuntu 24.04 system Python) | [PX4 ROS 2 user guide](https://docs.px4.io/main/en/ros2/user_guide): "The supported and recommended ROS 2 platform for working with PX4 is ROS 2 "Jazzy" LTS on Ubuntu 24.04", with Gazebo Harmonic. rclpy comes from the system Python, which fixes the version |

Moving to another Python version means a Decision Log entry and a change to `requires-python`.

## Tooling

| Tool | Version | Notes |
|---|---|---|
| uv | 0.13.5 | `[tool.uv] required-version = ">=0.13.4"`: older releases trust a compromised CA ([GHSA-hq85-5cpj-mr62](https://github.com/advisories/GHSA-hq85-5cpj-mr62)). Upgrade with `uv self update` |
| hatchling | 1.32.4 | Build backend |
| ruff | 0.17.2 | Lint and format. `tests/test_style.py` runs both, so `pytest` fails on style errors |
| pytest | 9.1.1 | Config in `pyproject.toml` |

Direct dependencies are pinned exactly in `pyproject.toml`, the single place they are declared;
`uv.lock` is generated from it and pins the transitive dependencies. Packages that Colab/Kaggle
preinstall are pinned to Colab's version where one is known (numpy 2.0.2, from the runtime FAQ), so
installing this package in a cloud session does not replace them. CI (`.github/workflows/ci.yml`)
runs `uv sync --locked` and `uv run pytest` on Ubuntu 24.04 and Windows.

## Local setup

```bash
uv sync         # .venv with Python 3.12, the pinned dependencies and the dev group
uv run pytest
```

## Machine state

- No local GPU. Detector training and cache generation run on Kaggle/Colab (CLAUDE.md §2).
- WSL2 is not installed on the development machine yet. It is needed from M10 (Ubuntu 24.04,
  ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL). Installing it needs administrator rights and a reboot.
