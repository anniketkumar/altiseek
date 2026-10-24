"""The artifact registry (CLAUDE.md §18): every detector, sensor model, world set and cache with
its ID, parent IDs and storage location.

The manifest is a committed JSON file. Artifacts themselves never enter git; ``location`` says
where the bytes live, e.g. ``kaggle:<user>/<kernel>@<version>:<path in its output>``. Registering
is idempotent: an entry with the same kind and ID must match exactly, so an ID can never be
reused for different contents.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("detector", "sensor_model", "world_set", "cache")


@dataclass(frozen=True)
class Artifact:
    """One registered artifact."""

    kind: str
    artifact_id: str
    sha256: str
    location: str
    #: IDs this artifact was built from, by name (e.g. ``split_id``, ``detector_id``).
    parents: dict[str, str]
    created_utc: str
    notes: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: str | Path) -> list[Artifact]:
    """All entries, oldest first; an absent file is an empty manifest."""
    path = Path(path)
    if not path.exists():
        return []
    return [Artifact(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))]


def find(path: str | Path, kind: str, artifact_id: str) -> Artifact | None:
    """The entry for ``kind``/``artifact_id``, if registered."""
    return next(
        (a for a in load_manifest(path) if (a.kind, a.artifact_id) == (kind, artifact_id)), None
    )


def register(path: str | Path, artifact: Artifact) -> bool:
    """Add ``artifact``; returns False if an identical entry already exists.

    Raises ValueError for an unknown kind, or when the same kind and ID are already registered
    with different contents.
    """
    if artifact.kind not in KINDS:
        raise ValueError(f"unknown artifact kind {artifact.kind!r}; expected one of {KINDS}")
    entries = load_manifest(path)
    existing = next(
        (a for a in entries if (a.kind, a.artifact_id) == (artifact.kind, artifact.artifact_id)),
        None,
    )
    if existing is not None:
        if existing != artifact:
            raise ValueError(f"{artifact.kind} {artifact.artifact_id} is registered differently")
        return False
    entries.append(artifact)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([asdict(a) for a in entries], indent=2) + "\n", encoding="utf-8"
    )
    return True
