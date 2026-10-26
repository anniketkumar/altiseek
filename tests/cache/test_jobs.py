"""Stage resume recovery and fail-closed handling without cloud inference."""

import importlib.util
import json
import runpy
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


def test_attached_input_identity_and_split_checks(tmp_path, cfg):
    read = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/cache_pilot.py"))[
        "attached_config"
    ]
    cfg = cfg.model_copy(
        update={
            "libraries": {"B": tmp_path / "unused"},
            "library_ids": {"B": "lib"},
            "world_set_ids": {"sensor-fit": "set"},
        }
    )
    assert read(cfg, tmp_path) is None
    root = tmp_path / "input"
    (root / "libraries/B").mkdir(parents=True)
    marker = root / "m5-inputs.json"
    manifest = {
        "split_id": "783b4403804291a2",
        "library_ids": cfg.library_ids,
        "world_set_ids": cfg.world_set_ids,
    }
    marker.write_text(json.dumps(manifest))
    meta = root / "libraries/B/library.meta.json"
    meta.write_text(json.dumps({"split_id": manifest["split_id"], "split": "B"}))
    attached = read(cfg, tmp_path)
    assert attached.libraries["B"] == root / "libraries/B"
    assert attached.worlds_dir == root / "worlds"
    meta.write_text(json.dumps({"split_id": "wrong", "split": "B"}))
    with pytest.raises(ValueError, match="split mismatch"):
        read(cfg, tmp_path)
    marker.write_text(json.dumps(manifest | {"library_ids": {"B": "changed"}}))
    with pytest.raises(ValueError, match="frozen IDs"):
        read(cfg, tmp_path)


def test_manual_notebook_requires_preserved_inputs():
    make = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/prepare_cache_notebook.py")
    )["notebook"]
    notebook = make("a" * 40)
    code = "".join(notebook["cells"][1]["source"])
    compile(code, "pilot.ipynb", "exec")
    assert "--require-input-bundle" in code
    assert "KGAT_" not in code
    production = make("a" * 40, production=True)
    code = "".join(production["cells"][1]["source"])
    compile(code, "production.ipynb", "exec")
    assert "scripts/cache_production.py" in code
    assert "cache-production.log" in code


def test_production_resume_preserves_verified_bytes_and_reports_partial(tmp_path, cfg, built):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/cache_production.py")
    )
    spec, cache = built
    cfg = cfg.model_copy(
        update={"out_dir": tmp_path / "working/caches", "world_set_ids": {"sensor-fit": "set"}}
    )
    source = tmp_path / "input/pilot/caches/sensor-fit" / f"{spec.world_id}.npz"
    cache.save(source)
    meta = {
        "created_utc": "2026-10-26T00:00:00+00:00",
        "cache_id": cache.header.cache_id,
        "world_id": spec.world_id,
        "world_set_id": "set",
        "detector_id": "fake",
        "views": len(cache.views),
        "outputs": {"cache": {"sha256": sha256_file(source)}},
    }
    source.with_suffix(".meta.json").write_text(json.dumps(meta))
    worlds = {"sensor-fit": [spec]}
    assert module["resume_inputs"](cfg, tmp_path / "input", worlds) == 1
    assert module["resume_inputs"](cfg, tmp_path / "input", worlds) == 0
    target = cfg.out_dir / "sensor-fit" / source.name
    assert target.read_bytes() == source.read_bytes()
    assert module["completion"](cfg, worlds) == {
        "sensor-fit": {"expected": 1, "built": 1, "validated": 0}
    }
    target.with_suffix(".validation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cache_id": "wrong",
                "inputs": {"cache": {"sha256": sha256_file(target)}},
                "score_atol": 1e-4,
                "box_atol_px": 0.05,
                "view_indices": list(range(20)),
            }
        )
    )
    with pytest.raises(ValueError, match="validation"):
        module["completion"](cfg, worlds)
    with source.open("ab") as fh:
        fh.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        module["resume_inputs"](cfg, tmp_path / "input", worlds)
