"""Metadata records that every stage writes next to its outputs (CLAUDE.md §18).

A record holds the stage name, the UTC creation time, the git commit (and whether the
working tree had uncommitted changes), the Python and installed package versions, the
resolved config, and the SHA-256 of every input and output file.
"""

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from altiseek.config import Config


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(cwd: str | Path = ".") -> dict[str, Any] | None:
    """The HEAD commit and a dirty flag for the repository at ``cwd``; None outside git."""
    try:
        commit = _git(cwd, "rev-parse", "HEAD")
        dirty = bool(_git(cwd, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def write_metadata(
    path: str | Path,
    *,
    stage: str,
    config: Config,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, str | Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the metadata record for one stage run to ``path`` as JSON and return it."""
    record = {
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": platform.python_version(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
        },
        "config": config.model_dump(mode="json"),
        "inputs": {name: _file_entry(p) for name, p in inputs.items()},
        "outputs": {name: _file_entry(p) for name, p in outputs.items()},
        **(extra or {}),
    }
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _file_entry(path: str | Path) -> dict[str, str]:
    return {"path": Path(path).as_posix(), "sha256": sha256_file(path)}


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout.strip()
