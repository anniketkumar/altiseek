"""Stage resume recovery and fail-closed handling without cloud inference."""

import importlib.util
import json
from pathlib import Path

import pytest

from altiseek.cache.jobs import resume_metadata
from altiseek.registry.run import sha256_file
from tests.cache.test_cache import FakeDetector, built, cfg, library
from tests.world.test_render import world

__all__ = ["built", "cfg", "library"]

module_spec = importlib.util.spec_from_file_location(
    "build_cache", Path(__file__).resolve().parents[2] / "scripts/build_cache.py"
)
assert module_spec is not None
assert module_spec.loader is not None
build_cache = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(build_cache)


def test_orphan_mismatch_is_not_replaced(tmp_path, cfg, built):
    spec, cache = built
    path = tmp_path / "orphan.npz"
    cache.save(path)
    assert resume_metadata(path, spec, cfg) is None
    other = cfg.model_copy(update={"detector_id": "different"})
    with pytest.raises(ValueError, match="header mismatch"):
        resume_metadata(path, spec, other)


def test_resumed_stage_keeps_completed_worlds_outside_limit(tmp_path, cfg, library, monkeypatch):
    specs = [world().model_copy(update={"seed": i}).identified() for i in (1, 2)]
    cfg = cfg.model_copy(
        update={"out_dir": tmp_path / "caches", "world_set_ids": {"sensor-fit": "set"}}
    )
    monkeypatch.setattr(build_cache, "load_config", lambda *a: cfg)
    monkeypatch.setattr(build_cache, "load_worlds", lambda *a: specs)
    monkeypatch.setattr(build_cache, "load_libraries", lambda *a: {"B": library})
    monkeypatch.setattr(build_cache, "gpu_detector", lambda *a: FakeDetector())

    def metadata(path, **kwargs):
        result = {
            "created_utc": "2026-10-26T00:00:00+00:00",
            **kwargs["extra"],
            "outputs": {"cache": {"sha256": sha256_file(kwargs["outputs"]["cache"])}},
        }
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(build_cache, "write_metadata", metadata)
    build_cache.main(["--sets", "sensor-fit"])
    original = json.loads((cfg.out_dir / "artifacts.json").read_text())
    assert len(original) == 2

    def unexpected_build(*args):
        raise AssertionError("completed worlds must not be rebuilt")

    monkeypatch.setattr(build_cache, "build_world", unexpected_build)
    build_cache.main(["--sets", "sensor-fit", "--limit", "1"])
    result = json.loads((cfg.out_dir / "sensor-fit/completed.json").read_text())
    assert result["complete"]
    assert len(result["worlds"]) == 2
    assert json.loads((cfg.out_dir / "artifacts.json").read_text()) == original

    path = cfg.out_dir / "sensor-fit" / f"{specs[0].world_id}.npz"
    with path.open("ab") as fh:
        fh.write(b"changed bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resume_metadata(path, specs[0], cfg)
