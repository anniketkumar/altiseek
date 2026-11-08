"""Detached, resumable D66 V3 pilot and the single predeclared learning-rate fallback."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from altiseek.cache.jobs import write_json
from altiseek.eval.pilot_review import collapsed
from altiseek.registry.run import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run(output: Path, warm_sha256: str, rate: float) -> Path:
    found = []
    for path in output.glob("*/plan.json"):
        plan = read(path)
        if (
            plan["input_hashes"].get("warm_start_weights") == warm_sha256
            and plan["config"]["learning_rate"] == rate
            and plan["seed"] == 17
            and plan["variant"] == "V3"
        ):
            found.append(path.parent)
    if len(found) != 1:
        raise ValueError("Pilot run identity missing or ambiguous; inspect before resuming")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warm_start", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/m12/pilot"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warm_hash = sha256_file(args.warm_start / "policy.pt")
    status_path = args.out / "status.json"
    state = {
        "stage": "primary-early",
        "warm_start_sha256": warm_hash,
        "controller_sha256": sha256_file(__file__),
    }
    if status_path.exists():
        saved = read(status_path)
        if saved["warm_start_sha256"] != warm_hash:
            raise ValueError("Another cloned policy owns this pilot directory")
        if saved["stage"] == "awaiting_owner_review":
            print("Pilot already complete; preserving results.", flush=True)
            return

    def command(name: str, config: str, stop_after: int | None = None) -> None:
        state["stage"] = name
        write_json(status_path, state)
        argv = [
            sys.executable,
            "-u",
            "scripts/ppo/train_ppo.py",
            "--config",
            config,
            "--search-config",
            "configs/env/search_live.yaml",
            "--variant",
            "V3",
            "--seed",
            "17",
            "--out",
            str(args.out),
            "--warm-start",
            str(args.warm_start),
            "--baseline-report",
            "docs/m12-ppo-revisit/baseline-results.json",
            "--selected-baselines",
            "configs/policy/selected_live.yaml",
        ]
        if stop_after is not None:
            argv.extend(["--stop-after", str(stop_after)])
        with (args.out / f"{name}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(argv, stdout=log, stderr=log, check=False)
        if result.returncode:
            state.update(
                stage="stopped_for_diagnosis", failed_command=name, returncode=result.returncode
            )
            write_json(status_path, state)
            raise RuntimeError(f"{name} failed; inspect its log and resume after diagnosis")

    primary_cfg = "configs/policy/ppo_live_pilot.yaml"
    command("primary-early", primary_cfg, 262144)
    primary = find_run(args.out, warm_hash, 0.0003)
    baseline = read(primary / "eval-000000000.json")["mean_auc"]
    early = [
        read(primary / f"eval-{step:09d}.json")["mean_auc"]
        for step in (65536, 131072, 196608, 262144)
    ]
    trigger = collapsed(early, baseline)
    decision = {
        "primary_run": primary.name,
        "step_zero_auc": baseline,
        "early_auc": early,
        "collapse_triggered": trigger,
        "rule": "two consecutive early raw AUCs at least 0.10 below step zero",
    }
    write_json(args.out / "fallback-decision.json", decision)
    state.update(decision)
    if trigger:
        command("fallback", "configs/policy/ppo_live_pilot_low_lr.yaml")
        selected = find_run(args.out, warm_hash, 0.0001)
    else:
        command("primary-complete", primary_cfg)
        selected = primary
    completion = read(selected / "complete.json")
    if completion["steps"] != 1048576:
        raise ValueError("Pilot ended before its declared budget")
    state.update(
        stage="awaiting_owner_review",
        completed_run=selected.name,
        completion_sha256=sha256_file(selected / "complete.json"),
    )
    write_json(status_path, state)
    with (args.out / "report.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/ppo/report_m12_pilot.py",
                str(args.out),
                str(args.warm_start.parent),
            ],
            stdout=log,
            stderr=log,
            check=False,
        )
    if result.returncode:
        state["stage"] = "reporting_failed"
        write_json(status_path, state)
        raise RuntimeError("Pilot completed but reporting failed; inspect report.log")
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
