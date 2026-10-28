"""Resume the D41 pilot, apply its frozen gate, then complete and report the six-run suite."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.policies.ppo import PPOConfig

ROOT = Path("runs/ppo/m8-auc")
CONFIG = Path("configs/policy/ppo_auc.yaml")


def run(script: str, arguments: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, "-u", f"scripts/{script}.py", *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    status = ROOT / "continuation-status.json"
    write_json(status, {"stage": "pilot", "complete": False})
    try:
        run(
            "train_ppo",
            [
                "--config",
                str(CONFIG),
                "--variant",
                "V3",
                "--seed",
                "17",
                "--out",
                str(ROOT / "V3-seed17"),
                "--stop-after",
                "65536",
            ],
            ROOT / "V3-seed17.log",
        )
        plans = list((ROOT / "V3-seed17").glob("*/plan.json"))
        if len(plans) != 1:
            raise ValueError("expected exactly one pilot identity")
        pilot = plans[0].parent
        plan = json.loads(plans[0].read_text(encoding="utf-8"))
        if digest(plan) != pilot.name or PPOConfig.model_validate(plan["config"]) != load_config(
            CONFIG, PPOConfig
        ):
            raise ValueError("pilot identity/configuration mismatch")
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in pilot.glob("eval-*.json")]
        reports = [r for r in reports if 0 < r["steps"] <= 65536]
        if {r["steps"] for r in reports} != {16384, 32768, 49152, 65536}:
            raise ValueError("pilot evaluations incomplete")
        best = max(reports, key=lambda r: (r["mean_normalised_auc"], -r["steps"]))
        finite, records = True, 0
        for path in pilot.glob("logs-*/progress.csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("train/loss"):
                        records += 1
                    for key in ("train/loss", "train/approx_kl", "train/value_loss"):
                        if row.get(key):
                            finite = finite and math.isfinite(float(row[key]))
        baseline = json.loads(Path("docs/baseline-results.json").read_text(encoding="utf-8"))
        threshold = next(
            r["auc"]
            for r in baseline["aggregate"]
            if r["variant"] == "V3" and r["method"] == "random"
        )
        passed = best["mean_auc"] >= threshold and finite and records > 0
        write_json(
            ROOT / "pilot-gate.json",
            {
                "run_id": pilot.name,
                "selected_step": best["steps"],
                "auc": best["mean_auc"],
                "random_auc": threshold,
                "finite_optimizer_metrics": finite,
                "optimizer_records": records,
                "passed": passed,
            },
        )
        if not passed:
            write_json(status, {"stage": "pilot_needs_review", "complete": False})
            print("Pilot gate failed; results preserved, suite expansion stopped.", flush=True)
            return
        write_json(status, {"stage": "suite", "complete": False})
        run("run_ppo_suite", ["--config", str(CONFIG), "--out", str(ROOT)], ROOT / "suite.log")
        write_json(status, {"stage": "report", "complete": False})
        run("report_ppo", [str(ROOT), "--config", str(CONFIG)], ROOT / "report.log")
        write_json(status, {"stage": "awaiting_scientific_review", "complete": True})
        print(
            "Six runs and report complete; review M8 acceptance before marking the milestone.",
            flush=True,
        )
    except Exception as error:
        write_json(status, {"stage": "failed", "complete": False, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
