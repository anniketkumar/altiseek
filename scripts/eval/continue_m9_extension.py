"""Run frozen M9 v2 evaluation, regenerate reports, then stop for review."""

from pathlib import Path

from filelock import FileLock

from altiseek.cache.jobs import write_json
from altiseek.eval.extension import run_extension
from altiseek.eval.extension_report import report


def main() -> None:
    out = Path("runs/eval/m9-v2")
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(out / "continuation.lock", timeout=0):
        status = out / "continuation-status.json"
        write_json(status, {"state": "evaluating"})
        root = run_extension(
            Path("configs/eval/protocol_v2.yaml"),
            Path("configs/eval/artifacts_v2.json"),
            out,
            smoke=False,
            limit=0,
        )
        write_json(status, {"state": "reporting", "run_id": root.name})
        report(root, root / "report")
        write_json(status, {"state": "awaiting_review", "run_id": root.name})


if __name__ == "__main__":
    main()
