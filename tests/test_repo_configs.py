"""Every YAML file under configs/ validates against the schema of its group."""

from pathlib import Path

import pytest

from altiseek.config import Config, load_config
from altiseek.data.config import DataConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SCHEMAS: dict[str, type[Config]] = {"data": DataConfig}
CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_config_validates_against_its_group_schema(path):
    group = path.relative_to(CONFIG_DIR).parts[0]
    assert group in SCHEMAS, f"no schema registered for configs/{group}/"
    load_config(path, SCHEMAS[group])
