"""register-detector: add a trained detector to the artifact manifest (CLAUDE.md §18).

Reads a downloaded kernel output (``scripts/kaggle_job.py`` puts it under ``data/kaggle/<slug>``),
checks that the checkpoint's hash matches its ``detector_id``, and records its lineage: split ID,
camera-model hash and code commit, plus the evaluation summary if present.

    uv run python scripts/register_detector.py data/kaggle/altiseek-detector-v1 \
        --location kaggle:aniketkumarpradhan/altiseek-detector-v1
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from altiseek.registry.manifest import Artifact, register
from altiseek.registry.run import sha256_file


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Register a trained detector.")
    parser.add_argument("run_dir", type=Path, help="downloaded kernel output")
    parser.add_argument("--location", required=True, help="where the kernel output lives")
    parser.add_argument("--manifest", type=Path, default=Path("registry/manifest.json"))
    args = parser.parse_args(argv)

    detector_dir = args.run_dir / "detector"
    record = json.loads((detector_dir / "detector.json").read_text(encoding="utf-8"))
    sha = sha256_file(detector_dir / record["checkpoint"])
    if sha[:16] != record["detector_id"]:
        sys.exit(f"checkpoint hash {sha[:16]} does not match detector_id {record['detector_id']}")
    meta = json.loads((detector_dir / "train_detector.meta.json").read_text(encoding="utf-8"))
    m1 = args.run_dir / "m1"
    split_id = json.loads((m1 / "make_splits.meta.json").read_text(encoding="utf-8"))["split_id"]
    eval_json = detector_dir / "eval" / "eval.json"
    evaluation = json.loads(eval_json.read_text(encoding="utf-8")) if eval_json.exists() else {}
    artifact = Artifact(
        kind="detector",
        artifact_id=record["detector_id"],
        sha256=sha,
        location=f"{args.location}:detector/{record['checkpoint']}",
        parents={
            "split_id": split_id,
            "camera_models_sha256": sha256_file(m1 / "camera_models.json")[:16],
            "code_commit": (meta.get("git") or {}).get("commit", "unknown"),
        },
        created_utc=meta["created_utc"],
        notes={
            "init_weights": record["init_weights"],
            "classes": record["classes"],
            "imgsz": record["imgsz"],
            "tau": evaluation.get("tau"),
            "overall": evaluation.get("overall"),
        },
    )
    added = register(args.manifest, artifact)
    print(("registered" if added else "already registered"), artifact.kind, artifact.artifact_id)


if __name__ == "__main__":
    main()
