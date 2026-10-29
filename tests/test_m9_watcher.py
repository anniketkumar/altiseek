import runpy
import subprocess
from pathlib import Path
from unittest.mock import patch


def test_watcher_retries_failed_request_without_relaunching_kernel(tmp_path):
    watcher = runpy.run_path(str(Path("scripts/watch_m9_cache.py")))
    command = ["kaggle", "kernels", "status", "owner/job"]
    responses = [
        subprocess.CompletedProcess(command, 1, "", "temporary failure"),
        subprocess.CompletedProcess(command, 0, "RUNNING\n", ""),
    ]
    with (
        patch("subprocess.run", side_effect=responses) as run,
        patch("time.sleep") as sleep,
    ):
        assert watcher["network_command"](command, tmp_path / "status.json") == "RUNNING"
    assert run.call_count == 2
    assert all(call.args[0] == command for call in run.call_args_list)
    sleep.assert_called_once_with(60)
