"""The linter and formatter run as part of the test suite (CLAUDE.md §5)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args", [("check",), ("format", "--check")], ids=["lint", "format"])
def test_ruff(args):
    result = subprocess.run(
        [sys.executable, "-m", "ruff", *args, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
