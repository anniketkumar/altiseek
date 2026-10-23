"""Hierarchical YAML configs validated against pydantic schemas.

Every stage entry point reads one YAML file, composes it, applies command-line
overrides and validates the result against the stage's schema, a :class:`Config`
subclass. The resolved config (:func:`to_yaml`) is what each run saves.

Composition: any mapping in a config file may carry the reserved key ``_base_``,
naming one YAML file or a list of them by a path relative to the file that names
them. Those files are composed recursively and deep-merged in list order, then the
mapping's own keys are merged on top. At the top of a file this is inheritance;
under a key it pulls another config group into that sub-tree, e.g.
``camera: {_base_: ../camera/default.yaml}``. Mappings merge key by key; every other
value (scalars and lists) replaces. ``_base_`` is not resolved inside lists.

Overrides such as ``"camera.hfov_deg=90"`` (the value is parsed as YAML) are applied
after composition. Schemas reject unknown keys, so a misspelt key in a file or an
override is an error instead of a silently ignored setting.
"""

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

BASE_KEY = "_base_"


class ConfigError(ValueError):
    """A config cannot be read, composed, overridden or validated."""


class Config(BaseModel):
    """Base for config schemas and their nested sections: unknown keys fail; instances frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def compose(path: str | Path) -> dict[str, Any]:
    """Read the YAML config file at ``path`` and resolve every ``_base_`` in it; no validation."""
    return _compose_file(Path(path).resolve(), chain=())


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge two config trees into a new one: mappings merge by key, other values replace."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(tree: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``tree`` with ``"dotted.key=value"`` overrides applied in order.

    Values are parsed as YAML, so ``3`` is an int, ``[1, 2]`` a list and ``null`` is None.
    Missing keys are created; a misspelt key therefore fails schema validation later.
    """
    out = copy.deepcopy(dict(tree))
    for item in overrides:
        dotted, sep, raw = item.partition("=")
        keys = dotted.strip().split(".")
        if not sep or not all(keys):
            raise ConfigError(f"override {item!r} is not of the form dotted.key=value")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as err:
            raise ConfigError(f"override {item!r} has an invalid YAML value: {err}") from err
        node = out
        for depth, key in enumerate(keys[:-1]):
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                prefix = ".".join(keys[: depth + 1])
                raise ConfigError(f"override {item!r}: {prefix} is not a mapping")
        node[keys[-1]] = value
    return out


def load_config[C: Config](path: str | Path, schema: type[C], overrides: Iterable[str] = ()) -> C:
    """Compose the config file at ``path``, apply ``overrides`` and validate it as ``schema``."""
    tree = apply_overrides(compose(path), overrides)
    try:
        return schema.model_validate(tree)
    except ValidationError as err:
        raise ConfigError(f"{path} does not validate as {schema.__name__}:\n{err}") from err


def to_yaml(config: Config) -> str:
    """Serialise a validated config as YAML; loading the text back gives an equal config."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def _compose_file(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"{BASE_KEY} cycle: {cycle}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ConfigError(f"cannot read config {path}: {err}") from err
    except yaml.YAMLError as err:
        raise ConfigError(f"{path} is not valid YAML: {err}") from err
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping, not {type(data).__name__}")
    return _resolve(data, path, (*chain, path))


def _resolve(node: dict[str, Any], path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    """Resolve ``_base_`` in ``node`` and its nested mappings; ``path`` is the file being read."""
    bases = node.get(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) for base in bases):
        raise ConfigError(f"{path}: {BASE_KEY} must be a path or a list of paths")
    merged: dict[str, Any] = {}
    for base in bases:
        merged = merge(merged, _compose_file((path.parent / base).resolve(), chain))
    own = {
        key: _resolve(value, path, chain) if isinstance(value, dict) else value
        for key, value in node.items()
        if key != BASE_KEY
    }
    return merge(merged, own)
