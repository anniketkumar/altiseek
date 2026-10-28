"""Frozen-checkpoint B-only diagnostics: sampling, input sensitivity and reward alignment."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from altiseek.cache.jobs import write_json
from altiseek.cache.storage import CachePool
from altiseek.calibration.model import digest
from altiseek.config import load_config
from altiseek.env.config import SearchConfig
from altiseek.env.logging import EpisodeLog
from altiseek.env.navigation import Navigation
from altiseek.env.search import SearchEnv
from altiseek.policies.ppo import PPOConfig
from altiseek.policies.training import inputs
from altiseek.registry.run import sha256_file, write_metadata


def probabilities(model: MaskablePPO, obs: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return (
            model.policy.get_distribution(tensor, action_masks=mask)
            .distribution.probs.cpu()
            .numpy()[0]
        )


def returns(records: list[dict[str, Any]], gamma: float) -> dict[str, float]:
    rewards = np.array([r["reward"] for r in records[1:]])
    return {
        "action_return": float(rewards.sum()),
        "discounted_action_return": float(np.sum(rewards * gamma ** np.arange(len(rewards)))),
    }


def sensitivity(model: MaskablePPO, obs: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    base = probabilities(model, obs, mask)
    result = {}
    # Same shape/range, no privileged values: causal input interventions at fixed state/mask.
    for name in (
        "flip_belief",
        "zero_belief",
        "flip_prior",
        "zero_coverage",
        "flip_drone",
        "zero_map",
    ):
        changed = {k: v.copy() for k, v in obs.items()}
        maps = changed["map"]
        levels = len(obs["vector"]) - 4
        if name == "flip_belief":
            maps[0] = maps[0, ::-1, ::-1]
        elif name == "zero_belief":
            # Match the frozen M8 zero-p diagnostic after its log preprocessing.
            old = model.policy.features_extractor.zero_belief
            model.policy.features_extractor.zero_belief = True
        elif name == "flip_prior":
            maps[1] = maps[1, ::-1, ::-1]
        elif name == "zero_coverage":
            maps[2 : 2 + levels] = 0
        elif name == "flip_drone":
            maps[levels + 4] = maps[levels + 4, ::-1, ::-1]
        else:
            maps[:] = 0
        altered = probabilities(model, changed, mask)
        if name == "zero_belief":
            model.policy.features_extractor.zero_belief = old
        result[name] = {
            "tv": float(np.abs(base - altered).sum() / 2),
            "argmax_changed": bool(base.argmax() != altered.argmax()),
        }
    return {"probabilities": base.tolist(), "interventions": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ppo-diagnosis"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config("configs/policy/ppo.yaml", PPOConfig)
    cache, sensor, worlds, input_paths = inputs()
    saved_path = Path("runs/ppo/m8-initial/selected-results.json")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    plan = {
        "version": "m8-diagnosis-v1",
        "selected_report_sha256": sha256_file(saved_path),
        "repeats": args.repeats,
        "source_sha256": sha256_file(Path(__file__)),
        "world_set_id": cache.world_set_ids["rl-val"],
    }
    root = args.out / digest(plan)
    write_json(root / "plan.json", plan)
    print(f"Diagnostic run {root}", flush=True)
    input_paths["selected"] = saved_path
    summary_rows, probes = [], []
    for selected in saved["selected"]:
        path = Path(selected["checkpoint"])
        if sha256_file(path) != selected["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        input_paths[f"model_{selected['variant']}_{selected['seed']}"] = path
        model = MaskablePPO.load(path, device="cpu")
        model.policy.set_training_mode(False)
        search = load_config("configs/env/search.yaml", SearchConfig).model_copy(
            update={"sensor_variant": selected["variant"]}
        )
        nav = Navigation(cache.render.levels, cache.render.camera.hfov_deg, search.energy)
        pool = CachePool(40)
        original = {r["world_id"]: r for r in selected["episodes"]}
        for mode in ("deterministic", "stochastic"):
            episodes = []
            for repeat in range(1 if mode == "deterministic" else args.repeats):
                for world in worlds["rl-val"]:
                    key = {
                        "variant": selected["variant"],
                        "seed": selected["seed"],
                        "mode": mode,
                        "repeat": repeat,
                        "world_id": world.world_id,
                    }
                    result_path = root / f"{digest(key)}.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        episodes.append(result)
                        probes.extend(result["probes"])
                        continue
                    env = EpisodeLog(
                        SearchEnv([world], cache, sensor, search, navigation=nav, pool=pool)
                    )
                    obs, info = env.reset(seed=20261027 + world.seed)
                    rng = np.random.default_rng(20261028 + repeat * 1000000 + world.seed)
                    visits, episode_probes, entropies, informative = [], [], [], []
                    while not info["ended"]:
                        mask = info["action_mask"]
                        p = probabilities(model, obs, mask)
                        if mode == "deterministic" and len(visits) % 10 == 0:
                            episode_probes.append({**key, **sensitivity(model, obs, mask)})
                        action = (
                            int(p.argmax())
                            if mode == "deterministic"
                            else int(rng.choice(6, p=p / p.sum()))
                        )
                        obs, _, _, _, info = env.step(action)
                        visits.append(env.search.node)
                        informative.append(info["events"]["informative_cells"])
                        entropies.append(float(-np.sum(p[p > 0] * np.log(p[p > 0]))))
                    metrics = env.summary() | returns(env.records, cfg.gamma)
                    metrics.update(
                        {
                            "repeat_fraction": 1 - len(set(visits)) / len(visits) if visits else 0,
                            "zero_information_fraction": float(np.mean(np.array(informative) == 0))
                            if informative
                            else 0,
                            "action_entropy": float(np.mean(entropies)) if entropies else 0,
                        }
                    )
                    if mode == "deterministic":
                        for name in ("auc", "reward", "found_fraction", "steps"):
                            if not np.isclose(
                                metrics[name], original[world.world_id]["metrics"][name], atol=1e-10
                            ):
                                raise ValueError(f"deterministic replay changed {name}")
                    result = {
                        "key": key,
                        "metrics": metrics,
                        "records": env.records,
                        "probes": episode_probes,
                    }
                    write_json(result_path, result)
                    episodes.append(result)
                    probes.extend(episode_probes)
            row = {
                "variant": selected["variant"],
                "seed": selected["seed"],
                "mode": mode,
                "episodes": len(episodes),
                **{
                    name: float(np.mean([r["metrics"][name] for r in episodes]))
                    for name in (
                        "auc",
                        "found_fraction",
                        "reward",
                        "action_return",
                        "discounted_action_return",
                        "repeat_fraction",
                        "zero_information_fraction",
                        "action_entropy",
                    )
                },
            }
            summary_rows.append(row)
            print(json.dumps(row), flush=True)
            write_json(root / "progress.json", summary_rows)
    alignment = []
    baseline_root = Path("data/baselines/8cd76bdb72f965a1")
    for filename in json.loads((baseline_root / "complete.json").read_text(encoding="utf-8"))[
        "files"
    ]:
        row = json.loads((baseline_root / filename).read_text(encoding="utf-8"))
        if row["key"]["variant"] not in cfg.variants:
            continue
        alignment.append(
            {
                "method": row["key"]["method"],
                "variant": row["key"]["variant"],
                "world_id": row["key"]["world_id"],
                "auc": row["metrics"]["auc"],
                "reward": row["metrics"]["reward"],
                **returns(row["records"], cfg.gamma),
            }
        )
    intervention_summary = []
    for variant in cfg.variants:
        subset = [p for p in probes if p["variant"] == variant]
        for name in subset[0]["interventions"]:
            values = [p["interventions"][name] for p in subset]
            intervention_summary.append(
                {
                    "variant": variant,
                    "intervention": name,
                    "observations": len(values),
                    "mean_tv": float(np.mean([v["tv"] for v in values])),
                    "argmax_change_fraction": float(np.mean([v["argmax_changed"] for v in values])),
                }
            )
    grouped = defaultdict(list)
    for row in alignment:
        grouped[row["variant"], row["method"]].append(row)
    reward_summary = [
        {
            "variant": v,
            "method": m,
            **{
                key: float(np.mean([r[key] for r in rows]))
                for key in ("auc", "reward", "action_return", "discounted_action_return")
            },
        }
        for (v, m), rows in grouped.items()
    ]
    write_json(
        root / "summary.json",
        {
            "run_id": root.name,
            "policies": summary_rows,
            "interventions": intervention_summary,
            "baseline_grid_rewards": reward_summary,
        },
    )
    write_metadata(
        root / "run.meta.json",
        stage="diagnose-ppo",
        config=cfg,
        inputs=input_paths,
        outputs={"plan": root / "plan.json", "summary": root / "summary.json"},
    )
    print(f"Completed {root}", flush=True)


if __name__ == "__main__":
    main()
