"""Rejected frames never become accepted evidence through retry handling."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "capture_retry", Path(__file__).resolve().parents[2] / "scripts/deploy/retry_m10_capture.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retry_recomputes_capture_and_returns_only_accepted_result():
    calls, rejected = [], []

    def capture():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("Actual camera missed policy goal by 1.152 m")
        return "accepted"

    assert MODULE.retry_capture(capture, lambda *args: rejected.append(args)) == "accepted"
    assert len(calls) == 3
    assert len(rejected) == 2


@pytest.mark.parametrize(
    ("reason", "expected_calls"),
    [
        ("Image/pose mismatch 0.060 s", 3),
        ("Actual camera missed policy goal by 1.189 m", 3),
        ("Checksum mismatch", 1),
    ],
)
def test_bounded_retries_and_integrity_stop(reason, expected_calls):
    calls = []

    def capture():
        calls.append(1)
        raise ValueError(reason)

    with pytest.raises(ValueError, match=reason):
        MODULE.retry_capture(capture, lambda *args: None)
    assert len(calls) == expected_calls
