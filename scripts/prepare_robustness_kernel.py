"""Prepare a private, pinned M9 cache kernel; submission is a separate CLI action."""

import argparse
import json
import re
from pathlib import Path

RUNNER = """import subprocess
import sys
from pathlib import Path

repo = Path('/tmp/altiseek-m9')
subprocess.run(['git', 'clone', '--quiet',
                'https://github.com/anniketkumar/altiseek.git', str(repo)], check=True)
subprocess.run(['git', 'checkout', '--quiet', {commit!r}], cwd=repo, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-input',
                '-e', '.'], cwd=repo, check=True)
with Path('/kaggle/working/cache-robustness.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', 'scripts/cache_robustness.py'],
                               cwd=repo, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    result = process.wait()
if result:
    raise RuntimeError(f'cache runner exited {{result}}')
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--slug", default="altiseek-m9-cache")
    parser.add_argument("--resume-source", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("data/kaggle/m9-kernel"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("commit must be a full git SHA")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.py").write_text(RUNNER.format(commit=args.commit), encoding="utf-8")
    metadata = {
        "id": f"aniketkumarpradhan/{args.slug}",
        "title": args.slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["aniketkumarpradhan/altiseek-m5-inputs"],
        "kernel_sources": [
            "aniketkumarpradhan/altiseek-detector-v2",
            "aniketkumarpradhan/altiseek-detector-v3",
            *args.resume_source,
        ],
        "competition_sources": [],
        "model_sources": [],
    }
    (args.out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared private kernel at {args.out}; commit {args.commit}")


if __name__ == "__main__":
    main()
