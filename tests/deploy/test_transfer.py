import json

import pytest

from altiseek.calibration.model import digest
from altiseek.deploy.transfer import authorize_world, load_protocol, paired_seed
from altiseek.registry.run import sha256_file
from tests.cache.test_cache import built, cfg, library

__all__ = ["built", "cfg", "library"]


def test_protocol_rejects_mutation_and_changed_inputs(tmp_path):
    source = tmp_path / "model"
    source.write_text("frozen")
    body = {"status": "frozen", "inputs": {str(source): sha256_file(source)}}
    body["protocol_id"] = digest(body)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(body))
    assert load_protocol(path)["protocol_id"] == body["protocol_id"]
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        load_protocol(path)
    body["status"] = "draft"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="not frozen"):
        load_protocol(path)


def test_scope_and_seeds_are_paired_without_outcomes(built):
    world, _ = built
    protocol = {"b_preflight_world_id": world.world_id, "world_ids": []}
    with pytest.raises(ValueError, match="preflight"):
        authorize_world(protocol, world)  # Fixture is not the frozen rl-val set.
    target = world.model_copy(update={"split": "T", "world_set": "test"}).identified()
    with pytest.raises(ValueError, match="held-out"):
        authorize_world(protocol, target)
    protocol["world_ids"] = [target.world_id]
    authorize_world(protocol, target)
    assert paired_seed(target.world_id, "launch") == paired_seed(target.world_id, "launch")
    assert paired_seed(target.world_id, "actions", 17) != paired_seed(
        target.world_id, "actions", 29
    )
