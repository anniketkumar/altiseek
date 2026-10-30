"""Register strictly collected M9 caches, B-fitted sensors and B stress world sets."""

from pathlib import Path

from altiseek.eval.runner import read
from altiseek.registry.manifest import Artifact, load_manifest, register
from altiseek.registry.run import sha256_file


def main() -> None:
    manifest = Path("registry/manifest.json")
    collection = read(Path("data/m9-robustness/collection.json"))
    count = 0
    for case in ("detector", "camera", "stress"):
        if collection["cases"][case].get("complete") is not True:
            raise ValueError("cannot register incomplete cache collections")
        root = Path("data/m9-robustness") / case
        for path in (root / "artifacts.json", root / "sensor-artifacts.json"):
            for artifact in load_manifest(path):
                location = Path(artifact.location.removeprefix("local:"))
                if sha256_file(location) != artifact.sha256:
                    raise ValueError("registered artifact bytes changed")
                count += register(manifest, artifact)
    for name in ("stress-low", "stress-high"):
        root = Path("data/m9-stress/worlds") / name
        meta = read(root / "world_set.meta.json")
        checksum = sha256_file(root / "worlds.csv")
        if checksum[:16] != meta["world_set_id"] or meta["split"] != "B":
            raise ValueError("stress set identity mismatch")
        count += register(
            manifest,
            Artifact(
                kind="world_set",
                artifact_id=checksum[:16],
                sha256=checksum,
                location=f"local:{root.as_posix()}",
                parents={"library_id": meta["library_id"], "code_commit": meta["git"]["commit"]},
                created_utc=meta["created_utc"],
                notes={"worlds": 40, "split": "B", "role": "stress"},
            ),
        )
    print(f"Registered {count} new M9 artifacts", flush=True)


if __name__ == "__main__":
    main()
