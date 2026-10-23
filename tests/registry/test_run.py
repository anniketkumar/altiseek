"""Run metadata records."""

import hashlib
import json
import subprocess

from altiseek.config import Config
from altiseek.registry.run import git_state, sha256_file, write_metadata


class Settings(Config):
    seed: int


def test_record_holds_config_and_file_hashes(tmp_path):
    source = tmp_path / "in.txt"
    source.write_bytes(b"abc")
    result = tmp_path / "out.txt"
    result.write_bytes(b"xyz")
    record = write_metadata(
        tmp_path / "meta.json",
        stage="test-stage",
        config=Settings(seed=3),
        inputs={"source": source},
        outputs={"result": result},
        extra={"summary": {"n": 1}},
    )
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")) == record
    assert record["stage"] == "test-stage"
    assert record["config"] == {"seed": 3}
    assert record["inputs"]["source"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert record["outputs"]["result"]["sha256"] == hashlib.sha256(b"xyz").hexdigest()
    assert record["summary"] == {"n": 1}
    assert "pydantic" in {name.lower() for name in record["packages"]}


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "big.bin"
    data = bytes(range(256)) * 10_000
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_git_state_outside_a_repository_is_none(tmp_path, monkeypatch):
    # Stop git from finding an enclosing repository (e.g. a home directory under version control).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_state(tmp_path) is None


def test_git_state_reports_commit_and_dirty_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "f.txt").write_text("1", encoding="utf-8")
    git("add", "f.txt")
    git("commit", "-q", "-m", "c")
    clean = git_state(tmp_path)
    assert clean is not None
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    (tmp_path / "f.txt").write_text("2", encoding="utf-8")
    assert git_state(tmp_path) == {"commit": clean["commit"], "dirty": True}
