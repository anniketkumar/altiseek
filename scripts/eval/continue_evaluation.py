"""Run/resume frozen M9 evaluation, then generate its report without reselection."""

import traceback
from pathlib import Path

from altiseek.cache.jobs import write_json
from altiseek.eval.reporting import report
from altiseek.eval.runner import run


def main() -> None:
    output = Path("runs/eval/m9-v1")
    status = output / "continuation-status.json"
    write_json(status, {"status": "evaluating"})
    try:
        root = run(
            Path("configs/eval/protocol_v1.yaml"),
            Path("configs/eval/artifacts_v1.json"),
            output,
            smoke=False,
            limit=0,
        )
        write_json(status, {"status": "reporting", "run_id": root.name})
        report(root, root / "report")
        write_json(status, {"status": "awaiting_review", "run_id": root.name})
    except BaseException:
        write_json(status, {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
