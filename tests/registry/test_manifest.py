"""The artifact manifest."""

from dataclasses import replace

import pytest

from altiseek.registry.manifest import Artifact, find, load_manifest, register

DETECTOR = Artifact(
    kind="detector",
    artifact_id="4a7ff425dd6afff2",
    sha256="4a7ff425dd6afff2" + "0" * 48,
    location="kaggle:user/altiseek-detector-v1@1:detector/train/weights/best.pt",
    parents={"split_id": "783b4403804291a2"},
    created_utc="2026-10-24T17:00:00+00:00",
    notes={"tau": 0.3},
)


def test_register_round_trips_and_is_idempotent(tmp_path):
    path = tmp_path / "manifest.json"
    assert load_manifest(path) == []
    assert register(path, DETECTOR) is True
    assert register(path, DETECTOR) is False
    assert load_manifest(path) == [DETECTOR]
    assert find(path, "detector", "4a7ff425dd6afff2") == DETECTOR
    assert find(path, "cache", "4a7ff425dd6afff2") is None


def test_an_id_cannot_be_reused_for_different_contents(tmp_path):
    path = tmp_path / "manifest.json"
    register(path, DETECTOR)
    with pytest.raises(ValueError, match="registered differently"):
        register(path, replace(DETECTOR, location="elsewhere"))


def test_unknown_kinds_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown artifact kind"):
        register(tmp_path / "m.json", replace(DETECTOR, kind="model"))
