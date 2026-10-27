"""Run all declared PPO seeds serially; safe to relaunch after interruption."""

import argparse
import subprocess
import sys
from pathlib import Path

from altiseek.cache.jobs import write_json
from altiseek.config import load_config
from altiseek.policies.ppo import PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/policy/ppo.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/ppo/m8-initial"))
    args = parser.parse_args()
    cfg = load_config(args.config, PPOConfig)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    # V3 first gives the primary model's learning curves before the V0 ablation.
    for variant in reversed(cfg.variants):
        for seed in cfg.seeds:
            name = f"{variant}-seed{seed}"
            row = {"variant": variant, "seed": seed, "status": "running"}
            rows.append(row)
            write_json(args.out / "status.json", {"runs": rows, "complete": False})
            command = [
                sys.executable,
                "-u",
                "scripts/train_ppo.py",
                "--config",
                str(args.config),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--out",
                str(args.out / name),
            ]
            print(f"Starting {name}", flush=True)
            with (args.out / f"{name}.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            row["status"] = "complete" if result.returncode == 0 else "failed"
            write_json(args.out / "status.json", {"runs": rows, "complete": False})
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect its log before resuming")
            print(f"Finished {name}", flush=True)
    write_json(args.out / "status.json", {"runs": rows, "complete": True})


if __name__ == "__main__":
    main()
