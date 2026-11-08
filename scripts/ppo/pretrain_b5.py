"""D66: world-disjoint masked actor/critic pretraining; CPU and epoch-resumable."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from altiseek.cache.config import CacheConfig
from altiseek.cache.jobs import load_worlds, resume_metadata, write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import SensorModel, digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.search import SearchEnv
from altiseek.policies.imitation import ImitationConfig, epoch_metrics, save_weights, split_worlds
from altiseek.policies.ppo import FiniteSearch, PPOConfig
from altiseek.policies.training import make_model
from altiseek.registry.run import sha256_file, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/policy/imitation_live.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/m12/imitation"))
    args = parser.parse_args()
    cfg = load_config(args.config, ImitationConfig)
    torch.set_num_threads(cfg.threads)
    collection = json.loads((args.collection / "complete.json").read_text(encoding="utf-8"))
    demo_plan = json.loads((args.collection / "plan.json").read_text(encoding="utf-8"))
    if (
        digest(demo_plan) != args.collection.name
        or collection["collection_id"] != args.collection.name
    ):
        raise ValueError("Collection identity mismatch")
    cache = load_config("configs/cache/default.yaml", CacheConfig)
    sensor = SensorModel.load(
        "data/sensor_models/fcf8cdae58ca3947.json",
        detector_id=cache.detector_id,
        fit_world_set_id=cache.world_set_ids["sensor-fit"],
    )
    worlds = load_worlds(cache, "rl-train")
    for world in worlds:
        if (
            resume_metadata(cache.out_dir / "rl-train" / f"{world.world_id}.npz", world, cache)
            is None
        ):
            raise ValueError("Unverified B pretraining cache")
    partition = split_worlds(worlds, cfg.holdout_per_family)
    if collection["partition"] != partition or demo_plan["config"] != cfg.model_dump(mode="json"):
        raise ValueError("Demonstration split/config mismatch")
    # Only rl-train worlds are given to the pretraining model; rl-val is never sampled.
    train_worlds = [w for w in worlds if w.world_id in partition["train"]]
    episodes = {
        name: [e for e in collection["episodes"] if e["world_id"] in ids]
        for name, ids in partition.items()
    }
    if {e["world_id"] for e in collection["episodes"]} != set(demo_plan["world_ids"]):
        raise ValueError("Incomplete demonstration worlds")
    expected = {
        (w.world_id, repeat): cfg.reset_seed_base + repeat * 1000000 + w.seed
        for w in worlds
        for repeat in range(cfg.repeats)
    }
    actual = {(e["world_id"], e["repeat"]): e["seed"] for e in collection["episodes"]}
    if actual != expected or len(collection["episodes"]) != len(expected):
        raise ValueError("Incomplete or duplicated demonstration reset grid")
    metadata = json.loads((args.collection / "run.meta.json").read_text(encoding="utf-8"))
    if metadata["outputs"]["completion"]["sha256"] != sha256_file(
        args.collection / "complete.json"
    ):
        raise ValueError("Demonstration completion checksum mismatch")
    if demo_plan["world_ids"] != [w.world_id for w in worlds]:
        raise ValueError("Demonstration world order mismatch")
    for name, path in (
        ("sensor", Path("data/sensor_models/fcf8cdae58ca3947.json")),
        ("energy", Path("configs/env/energy_live.yaml")),
        ("selected", Path("configs/policy/selected_live.yaml")),
        ("search", Path("configs/env/search_live.yaml")),
    ):
        if sha256_file(path) != demo_plan["input_hashes"][name]:
            raise ValueError("Demonstration input changed before pretraining")
    search = load_config("configs/env/search_live.yaml", SearchConfig).model_copy(
        update={"discount": 1.0}
    )
    ppo = load_config("configs/policy/ppo_live_pilot.yaml", PPOConfig)
    plan = {
        "implementation": "m12-b5-pretrain-v1",
        "config": cfg.model_dump(mode="json"),
        "collection_id": args.collection.name,
        "partition": partition,
        "completion_sha256": sha256_file(args.collection / "complete.json"),
        "source_hashes": {
            name: sha256_file(name)
            for name in ("src/altiseek/policies/imitation.py", "scripts/ppo/pretrain_b5.py")
        },
    }
    root = args.out / digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    if (root / "complete.json").exists():
        print(f"Already complete; preserving {root}", flush=True)
        return
    vector = DummyVecEnv(
        [
            lambda: FiniteSearch(
                SearchEnv(train_worlds, cache, sensor, search, pool=CachePool(40)), "auc"
            )
        ]
    )
    model = make_model(vector, ppo, cfg.seed)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = cfg.learning_rate
    rng = np.random.default_rng(cfg.seed)
    history, start, stale, best, anchor = [], 1, 0, float("inf"), float("inf")
    state_path = root / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(state["policy"])
        model.policy.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["shuffle_rng"]
        history, start, stale, best, anchor = (
            state[k] for k in ("history", "next_epoch", "stale", "best", "anchor")
        )
    print(
        f"Pretrain {root}; train={len(partition['train'])}, "
        f"holdout={len(partition['holdout'])} worlds",
        flush=True,
    )
    try:
        for epoch in range(start, cfg.max_epochs + 1):
            if stale >= cfg.patience:
                break
            train = epoch_metrics(
                model, args.collection, episodes["train"], cfg, rng=rng, train=True
            )
            holdout = epoch_metrics(model, args.collection, episodes["holdout"], cfg)
            history.append({"epoch": epoch, "train": train, "holdout": holdout})
            if holdout["loss"] < best:
                best = holdout["loss"]
                save_weights(
                    model,
                    root / "best",
                    {
                        "pretrain_id": root.name,
                        "epoch": epoch,
                        "holdout_loss": best,
                        "collection_id": args.collection.name,
                    },
                )
            if holdout["loss"] <= anchor - cfg.min_improvement:
                anchor, stale = holdout["loss"], 0
            else:
                stale += 1
            temporary = root / "state.tmp.pt"
            torch.save(
                {
                    "policy": model.policy.state_dict(),
                    "optimizer": model.policy.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "shuffle_rng": rng.bit_generator.state,
                    "history": history,
                    "next_epoch": epoch + 1,
                    "stale": stale,
                    "best": best,
                    "anchor": anchor,
                },
                temporary,
            )
            temporary.replace(state_path)
            write_json(root / "learning-curve.json", history)
            print(
                f"epoch={epoch} train={train['loss']:.5f} holdout={holdout['loss']:.5f} "
                f"accuracy={holdout['accuracy']:.3f} stale={stale}",
                flush=True,
            )
        best_record = json.loads((root / "best/complete.json").read_text(encoding="utf-8"))
        write_json(
            root / "complete.json",
            {
                "pretrain_id": root.name,
                "best": best_record,
                "epochs": len(history),
                "collection_id": args.collection.name,
            },
        )
        write_metadata(
            root / "run.meta.json",
            stage="pretrain-m12-b5",
            config=cfg,
            inputs={"collection": args.collection / "complete.json", "config": args.config},
            outputs={"completion": root / "complete.json", "curve": root / "learning-curve.json"},
        )
        print(f"Completed {root}", flush=True)
    finally:
        vector.close()


if __name__ == "__main__":
    main()
