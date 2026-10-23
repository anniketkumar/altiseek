"""Config composition (_base_), overrides, validation and round-tripping."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from altiseek.config import (
    Config,
    ConfigError,
    apply_overrides,
    compose,
    load_config,
    merge,
    to_yaml,
)


class Camera(Config):
    resolution_px: int
    hfov_deg: float
    jpeg_quality: int | None = None


class Stage(Config):
    name: str
    seed: int
    camera: Camera
    levels_cells: tuple[int, ...]


VALID = "name: s\nseed: 0\ncamera: {resolution_px: 640, hfov_deg: 80}\nlevels_cells: [5, 10]\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_child_inherits_base_mappings_merge_and_lists_replace(tmp_path):
    write(
        tmp_path / "base.yaml",
        "name: base\nseed: 1\n"
        "camera: {resolution_px: 640, hfov_deg: 80}\n"
        "levels_cells: [5, 10, 20]\n",
    )
    child = write(
        tmp_path / "child.yaml",
        "_base_: base.yaml\nseed: 2\ncamera: {hfov_deg: 90}\nlevels_cells: [5, 10]\n",
    )
    assert load_config(child, Stage) == Stage(
        name="base",
        seed=2,
        camera=Camera(resolution_px=640, hfov_deg=90.0),
        levels_cells=(5, 10),
    )


def test_nested_base_composes_another_config_group(tmp_path):
    write(tmp_path / "camera" / "default.yaml", "resolution_px: 640\nhfov_deg: 80\n")
    stage = write(
        tmp_path / "env" / "default.yaml",
        "name: env\nseed: 0\n"
        "camera:\n  _base_: ../camera/default.yaml\n  jpeg_quality: 90\n"
        "levels_cells: [5]\n",
    )
    assert load_config(stage, Stage).camera == Camera(
        resolution_px=640, hfov_deg=80.0, jpeg_quality=90
    )


def test_bases_resolve_relative_to_the_file_that_names_them(tmp_path):
    write(tmp_path / "a" / "root.yaml", "x: 1\ny: 1\n")
    write(tmp_path / "a" / "mid.yaml", "_base_: root.yaml\ny: 2\n")
    leaf = write(tmp_path / "b" / "leaf.yaml", "_base_: ../a/mid.yaml\nz: 3\n")
    assert compose(leaf) == {"x": 1, "y": 2, "z": 3}


def test_later_bases_win_and_own_keys_win_over_bases(tmp_path):
    write(tmp_path / "one.yaml", "x: 1\ny: 1\n")
    write(tmp_path / "two.yaml", "y: 2\nz: 2\n")
    top = write(tmp_path / "top.yaml", "_base_: [one.yaml, two.yaml]\nz: 3\n")
    assert compose(top) == {"x": 1, "y": 2, "z": 3}


def test_base_cycle_is_an_error(tmp_path):
    write(tmp_path / "a.yaml", "_base_: b.yaml\n")
    write(tmp_path / "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        compose(tmp_path / "a.yaml")


def test_missing_base_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        compose(write(tmp_path / "x.yaml", "_base_: missing.yaml\n"))


def test_base_must_be_a_path_or_list_of_paths(tmp_path):
    with pytest.raises(ConfigError, match="_base_ must be"):
        compose(write(tmp_path / "x.yaml", "_base_: {a: 1}\n"))


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "just a string\n"])
def test_top_level_must_be_a_mapping(tmp_path, text):
    with pytest.raises(ConfigError, match="mapping"):
        compose(write(tmp_path / "x.yaml", text))


def test_invalid_yaml_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        compose(write(tmp_path / "x.yaml", "a: [1\n"))


def test_empty_file_composes_to_an_empty_mapping(tmp_path):
    assert compose(write(tmp_path / "x.yaml", "")) == {}


def test_overrides_are_parsed_as_yaml_and_leave_the_input_untouched():
    tree = {"camera": {"hfov_deg": 80}}
    out = apply_overrides(
        tree,
        [
            "camera.hfov_deg=90.5",
            "seed=7",
            "levels_cells=[5, 10]",
            "camera.jpeg_quality=null",
            "name=run a",
        ],
    )
    assert out == {
        "camera": {"hfov_deg": 90.5, "jpeg_quality": None},
        "seed": 7,
        "levels_cells": [5, 10],
        "name": "run a",
    }
    assert tree == {"camera": {"hfov_deg": 80}}


@pytest.mark.parametrize("item", ["seed", "=3", "camera..hfov_deg=1", "seed=[1"])
def test_malformed_override_is_an_error(item):
    with pytest.raises(ConfigError):
        apply_overrides({}, [item])


def test_override_cannot_descend_into_a_scalar():
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_overrides({"seed": 1}, ["seed.value=2"])


def test_unknown_key_in_file_fails_validation(tmp_path):
    path = write(tmp_path / "x.yaml", VALID + "camera_hfov_deg: 90\n")
    with pytest.raises(ConfigError, match="camera_hfov_deg"):
        load_config(path, Stage)


def test_misspelt_override_fails_validation(tmp_path):
    path = write(tmp_path / "x.yaml", VALID)
    with pytest.raises(ConfigError, match="hfvo_deg"):
        load_config(path, Stage, ["camera.hfvo_deg=90"])


def test_wrong_type_fails_validation(tmp_path):
    path = write(tmp_path / "x.yaml", VALID)
    with pytest.raises(ConfigError, match="seed"):
        load_config(path, Stage, ["seed=many"])


def test_configs_are_frozen(tmp_path):
    config = load_config(write(tmp_path / "x.yaml", VALID), Stage)
    with pytest.raises(ValidationError):
        config.seed = 3


def test_resolved_yaml_round_trips(tmp_path):
    config = load_config(write(tmp_path / "x.yaml", VALID), Stage, ["camera.jpeg_quality=85"])
    resolved = write(tmp_path / "resolved.yaml", to_yaml(config))
    assert load_config(resolved, Stage) == config


def test_merge_does_not_share_state_with_its_inputs():
    base = {"a": {"b": [1]}}
    override = {"a": {"c": [2]}}
    out = merge(base, override)
    out["a"]["b"].append(9)
    out["a"]["c"].append(9)
    assert base == {"a": {"b": [1]}}
    assert override == {"a": {"c": [2]}}
