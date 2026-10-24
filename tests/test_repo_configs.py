"""Every YAML file under configs/ validates against its schema (by file, else by group)."""

from pathlib import Path

import pytest

from altiseek.config import Config, load_config
from altiseek.data.config import DataConfig
from altiseek.detector.config import DetectorConfig
from altiseek.env.energy import EnergyModel
from altiseek.geometry.design_config import LevelDesign
from altiseek.imaging.config import CameraSpec

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SCHEMAS: dict[str, type[Config]] = {
    "camera": CameraSpec,
    "data": DataConfig,
    "detector": DetectorConfig,
    "env/energy": EnergyModel,
    "levels/design": LevelDesign,
}
CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_config_validates_against_its_group_schema(path):
    relative = path.relative_to(CONFIG_DIR)
    key = relative.with_suffix("").as_posix()
    schema = SCHEMAS.get(key) or SCHEMAS.get(relative.parts[0])
    assert schema is not None, f"no schema registered for configs/{key}"
    load_config(path, schema)
