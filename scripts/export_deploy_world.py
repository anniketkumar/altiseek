"""Export one B integration world's exact shared composition to native tiled assets.

T export is deliberately unavailable until a transfer protocol is frozen.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from altiseek.config import load_config
from altiseek.deploy.assets import export_tiles
from altiseek.registry.run import sha256_file
from altiseek.world.config import WorldConfig
from altiseek.world.library import Library
from altiseek.world.render import compose_world
from altiseek.world.spec import WorldSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--root", type=Path, default=Path.home() / "altiseek-m10/worlds")
    args = parser.parse_args()
    spec = WorldSpec.load(args.world)
    if spec.split != "B":
        raise ValueError("Integration exporter accepts B only; no T protocol is frozen")
    output = args.root / spec.world_id
    output.mkdir(parents=True, exist_ok=True)
    with (output / "export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = output / "export.json"
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["world_id"] != spec.world_id or any(
                sha256_file(output / name) != digest for name, digest in saved["sha256"].items()
            ):
                raise ValueError("Existing export failed identity/checksum validation")
            print(f"Verified existing export: {metadata}", flush=True)
            return
        started = time.monotonic()
        cfg = load_config("configs/world/default.yaml", WorldConfig)
        library = Library.load(cfg.libraries[spec.split])
        print(f"Composing B world {spec.world_id}", flush=True)
        rgb = compose_world(spec, library)
        ground = export_tiles(rgb, output / "textures", spec.gsd_mpp)
        (output / "ground.sdf").write_text(ground, encoding="utf-8")
        spec.save(output / "world.json")
        files = [
            *sorted((output / "textures").iterdir()),
            output / "ground.sdf",
            output / "world.json",
        ]
        report = {
            "world_id": spec.world_id,
            "library_id": spec.library_id,
            "split": spec.split,
            "world_set": spec.world_set,
            "gsd_mpp": spec.gsd_mpp,
            "shape": list(rgb.shape),
            "composed_pixels_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "sha256": {str(path.relative_to(output)): sha256_file(path) for path in files},
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
            "elapsed_s": time.monotonic() - started,
        }
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metadata)
        print(f"Completed: {metadata} in {report['elapsed_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
