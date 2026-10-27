"""Every YAML file under configs/ validates against its schema (by file, else by group)."""

from pathlib import Path

import pytest

from altiseek.cache.config import CacheConfig
from altiseek.calibration.config import SensorConfig
from altiseek.config import Config, load_config
from altiseek.data.config import DataConfig
from altiseek.detector.config import DetectorConfig
from altiseek.env.config import SearchConfig
from altiseek.env.energy import EnergyModel
from altiseek.geometry.design_config import LevelDesign
from altiseek.geometry.level_design import coverage_energy_s
from altiseek.geometry.levels import LevelTable
from altiseek.imaging.config import CameraSpec
from altiseek.policies.config import BaselineRun
from altiseek.world.config import LibraryConfig, RenderConfig, WorldConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SCHEMAS: dict[str, type[Config]] = {
    "cache": CacheConfig,
    "camera": CameraSpec,
    "data": DataConfig,
    "detector": DetectorConfig,
    "env/energy": EnergyModel,
    "env/search": SearchConfig,
    "levels": LevelTable,
    "levels/design": LevelDesign,
    "sensor": SensorConfig,
    "policy/baselines": BaselineRun,
    "world": WorldConfig,
    "world/library": LibraryConfig,
    "world/render": RenderConfig,
}
CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_config_validates_against_its_group_schema(path):
    relative = path.relative_to(CONFIG_DIR)
    key = relative.with_suffix("").as_posix()
    schema = SCHEMAS.get(key) or SCHEMAS.get(relative.parts[0])
    assert schema is not None, f"no schema registered for configs/{key}"
    load_config(path, schema)


def test_chosen_levels_pass_the_coverage_cost_check():
    """§8 step 4 for the D7 levels: a full sweep at the finest level does not fit the battery, and
    one at the coarsest costs at most ``coarse_share`` of it."""
    table = load_config(CONFIG_DIR / "levels/default.yaml", LevelTable)
    design = load_config(CONFIG_DIR / "levels/design.yaml", LevelDesign)
    assert table.area_m == design.area_m
    finest, coarsest = (
        coverage_energy_s(table, lvl, design.camera.hfov_deg, design.energy)
        for lvl in (table.levels[0], table.levels[-1])
    )
    assert finest > design.energy.battery_s >= coarsest / design.coarse_share
