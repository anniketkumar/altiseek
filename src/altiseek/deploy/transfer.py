"""Frozen transfer identities and deterministic paired seeds; no result-based selection."""

import json
from pathlib import Path

from altiseek.calibration.model import digest
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


def paired_seed(world_id: str, purpose: str, training_seed: int = 0) -> int:
    return int(digest(["m10-v1", world_id, purpose, training_seed])[:8], 16) % (2**31)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if protocol.get("status") != "frozen" or protocol.get("protocol_id") != digest(body):
        raise ValueError("Transfer protocol is not frozen or its identity changed")
    for name, checksum in protocol["inputs"].items():
        if sha256_file(name) != checksum:
            raise ValueError(f"Frozen transfer input changed: {name}")
    return protocol


def authorize_world(protocol: dict, world: WorldSpec) -> None:
    if world.world_id != world.content_id():
        raise ValueError("Invalid world identity")
    if world.split == "B":
        if world.world_id != protocol["b_preflight_world_id"] or world.world_set != "rl-val":
            raise ValueError("World is outside frozen B preflight scope")
    elif world.split == "T":
        if world.world_id not in protocol["world_ids"] or world.world_set != "test":
            raise ValueError("World is outside frozen held-out transfer scope")
    else:
        raise ValueError("Unsupported transfer world split")


def sensor_path(protocol: dict, case: str) -> Path:
    if case not in ("original", "refit"):
        raise ValueError("Unknown transfer sensor case")
    item = protocol["sensors"][case]
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Frozen transfer sensor checksum mismatch")
    return path
