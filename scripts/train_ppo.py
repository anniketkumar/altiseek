"""Resumable CPU PPO runs on B; complete checkpoints, TensorBoard and numeric validation."""

import argparse
import json
import time
from pathlib import Path

from stable_baselines3.common.logger import configure

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.policies.ppo import PPOConfig
from altiseek.policies.training import (
    evaluate,
    inputs,
    load_checkpoint,
    make_model,
    save_checkpoint,
    vector_env,
)
from altiseek.registry.run import sha256_file, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/policy/ppo.yaml"))
    parser.add_argument("--variant", choices=("V0", "V3"), default="V3")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--vectorization", choices=("dummy", "subproc"))
    parser.add_argument(
        "--stop-after", type=int, help="pause at a complete evaluation boundary; same run identity"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, PPOConfig)
    changes = {}
    if args.smoke or args.benchmark:
        changes.update(total_steps=1024, eval_every=512, n_steps=64, epochs=2)
        for key in ("n_envs", "threads", "vectorization"):
            if getattr(args, key) is not None:
                changes[key] = getattr(args, key)
    elif any(getattr(args, k) is not None for k in ("n_envs", "threads", "vectorization")):
        raise ValueError("production settings must be frozen in the config")
    cfg = PPOConfig.model_validate(cfg.model_dump() | changes)
    if args.stop_after is not None and (
        args.stop_after <= 0
        or args.stop_after % cfg.eval_every
        or args.stop_after > cfg.total_steps
    ):
        raise ValueError("stop-after must be an evaluation boundary within the configured budget")
    if args.seed not in cfg.seeds or args.variant not in cfg.variants:
        raise ValueError("seed or variant not in declared configuration")
    search = load_config("configs/env/search.yaml", SearchConfig).model_copy(
        update={"sensor_variant": args.variant, "discount": cfg.gamma}
    )
    cache, sensor, worlds, paths = inputs()
    paths.update({"config": args.config, "search": Path("configs/env/search.yaml")})
    paths["lockfile"] = Path("uv.lock")
    sources = [
        Path("scripts/train_ppo.py"),
        *Path("src/altiseek/policies").glob("*.py"),
        *Path("src/altiseek/env").glob("*.py"),
        *Path("src/altiseek/belief").glob("*.py"),
    ]
    plan = {
        "implementation": "m8-maskable-auc-v2" if cfg.objective == "auc" else "m8-maskable-v1",
        "config": cfg.model_dump(mode="json"),
        "search": search.model_dump(mode="json"),
        "seed": args.seed,
        "variant": args.variant,
        "smoke": args.smoke,
        "benchmark": args.benchmark,
        "source_hashes": {p.as_posix(): sha256_file(p) for p in sorted(sources)},
        "input_hashes": {name: sha256_file(p) for name, p in paths.items()},
        "world_sets": {s: cache.world_set_ids[s] for s in worlds},
        "sensor_model_id": sensor.sensor_model_id,
    }
    root = args.out / digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    print(f"Run {root}: {args.variant} seed={args.seed}, {cfg.total_steps} steps", flush=True)
    if (root / "complete.json").exists():
        print("Already complete; preserving saved run.", flush=True)
        return
    env = vector_env(worlds["rl-train"], cache, sensor, search, cfg, args.seed)
    checkpoints = sorted(root.glob("step-*/complete.json"))
    model = (
        load_checkpoint(checkpoints[-1].parent, env)
        if checkpoints
        else make_model(env, cfg, args.seed)
    )
    logger = configure(str(root / f"logs-{model.num_timesteps:09d}"), ["csv", "tensorboard"])
    model.set_logger(logger)
    write_metadata(
        root / "run.meta.json",
        stage="train-ppo",
        config=cfg,
        inputs=paths,
        outputs={"plan": root / "plan.json"},
        extra={"run_id": root.name},
    )
    validation_worlds = worlds["rl-val"][:2] if args.smoke else worlds["rl-val"]
    try:
        while True:
            step = model.num_timesteps
            evaluation = root / f"eval-{step:09d}.json"
            if not args.benchmark and not evaluation.exists():
                result = evaluate(
                    model,
                    validation_worlds,
                    cache,
                    sensor,
                    search,
                    action_rule=cfg.evaluation_actions,
                    repeats=cfg.evaluation_repeats,
                )
                write_json(evaluation, result)
                logger.record("validation/auc", result["mean_auc"])
                logger.record("validation/normalised_auc", result["mean_normalised_auc"])
                logger.dump(step)
                print(
                    f"step={step} val AUC={result['mean_auc']:.4f} "
                    f"normalised={result['mean_normalised_auc']:.4f}",
                    flush=True,
                )
            if not args.benchmark and cfg.evaluation_actions == "sampled":
                argmax_path = root / f"argmax-{step:09d}.json"
                if not argmax_path.exists():
                    write_json(
                        argmax_path, evaluate(model, validation_worlds, cache, sensor, search)
                    )
            if args.stop_after is not None and step >= args.stop_after and step < cfg.total_steps:
                write_json(root / "paused.json", {"run_id": root.name, "steps": step})
                print(f"Paused at {step}; resume this same run without --stop-after.", flush=True)
                return
            if step >= cfg.total_steps:
                break
            tick = time.perf_counter()
            model.learn(
                total_timesteps=cfg.eval_every,
                reset_num_timesteps=False,
                use_masking=True,
                log_interval=1,
            )
            seconds = time.perf_counter() - tick
            checkpoint = root / f"step-{model.num_timesteps:09d}"
            save_checkpoint(model, checkpoint)
            write_json(
                checkpoint / "timing.json",
                {
                    "steps": cfg.eval_every,
                    "seconds": seconds,
                    "steps_per_second": cfg.eval_every / seconds,
                },
            )
            print(f"step={model.num_timesteps} {cfg.eval_every / seconds:.1f} steps/s", flush=True)
        if args.benchmark:
            result = {"run_id": root.name, "benchmark": True, "steps": model.num_timesteps}
        else:
            reports = [
                (p, json.loads(p.read_text(encoding="utf-8")))
                for p in sorted(root.glob("eval-*.json"))
            ]
            trained = [(p, r) for p, r in reports if r["steps"] > 0]
            best_path, best = max(
                trained, key=lambda pr: (pr[1]["mean_normalised_auc"], -pr[1]["steps"])
            )
            best_root = root / f"step-{best['steps']:09d}"
            best_model = load_checkpoint(best_root, env)
            zero = evaluate(
                best_model,
                validation_worlds,
                cache,
                sensor,
                search,
                zero_belief=True,
                action_rule=cfg.evaluation_actions,
                repeats=cfg.evaluation_repeats,
            )
            write_json(root / "zero-belief.json", zero)
            result = {
                "run_id": root.name,
                "steps": model.num_timesteps,
                "selected_step": best["steps"],
                "validation": best_path.name,
                "checkpoint_sha256": sha256_file(best_root / "model.zip"),
            }
        write_json(root / "complete.json", result)
        write_metadata(
            root / "complete.meta.json",
            stage="complete-ppo",
            config=cfg,
            inputs=paths,
            outputs={"completion": root / "complete.json"},
            extra={"run_id": root.name},
        )
        print(f"Completed {root}", flush=True)
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
