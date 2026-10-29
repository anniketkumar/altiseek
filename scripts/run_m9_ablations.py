"""Serial, resumable B-only ablation training; never launches held-out evaluation."""

import argparse
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

from altiseek.cache.jobs import write_json

CASES = {
    "control": ("m9_ablations", "search"),
    "relook": ("m9_ablations", "ablation_relook"),
    "declare": ("m9_ablations", "ablation_declare"),
    "unknown_k": ("m9_ablations", "ablation_unknown_k"),
    "shaping": ("m9_ablations", "ablation_shaping"),
    "egocentric": ("m9_ablations", "ablation_egocentric"),
    "appearance": ("m9_appearance", "search"),
    "appearance_control": ("m9_appearance_control", "search"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1024 steps, first seed per case")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--out", type=Path, default=Path("runs/ppo/m9-ablations"))
    args = parser.parse_args()
    root = args.out / "smoke" if args.smoke else args.out
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "suite.lock", timeout=0):
        for case in args.case or CASES:
            policy, search = CASES[case]
            for seed in (17,) if args.smoke else (17, 29, 43):
                name = f"{case}-seed{seed}"
                write_json(root / "status.json", {"state": "training", "case": case, "seed": seed})
                command = [
                    sys.executable,
                    "-u",
                    "scripts/train_ppo.py",
                    "--config",
                    f"configs/policy/{policy}.yaml",
                    "--search-config",
                    f"configs/env/{search}.yaml",
                    "--variant",
                    "V3",
                    "--seed",
                    str(seed),
                    "--out",
                    str(root / name),
                ]
                if args.smoke:
                    command.append("--smoke")
                with (root / f"{name}.log").open("a", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    write_json(
                        root / "status.json",
                        {
                            "state": "failed",
                            "case": case,
                            "seed": seed,
                            "returncode": result.returncode,
                        },
                    )
                    raise RuntimeError(f"{name} failed; inspect its log")
        write_json(root / "status.json", {"state": "training_complete", "smoke": args.smoke})


if __name__ == "__main__":
    main()
