"""Review numeric invariants in a completed, checksummed M9 extension report."""

import argparse
from pathlib import Path

import numpy as np

from altiseek.cache.jobs import write_json
from altiseek.calibration.model import digest
from altiseek.eval.extension import ExtensionProtocol, keys
from altiseek.eval.runner import read
from altiseek.registry.run import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = read(args.run / "report/results.json")
    assert result["new_episodes"] == 18720
    assert result["reference_episodes"] == 4320
    assert len(result["episode_sha256"]) == 23040
    lookup = {(r["case"], r["variant"], r["method"], r["group"]): r for r in result["rows"]}
    assert len(lookup) == len(result["rows"])
    for row in result["rows"]:
        metrics = row["metrics"]
        for name, value in metrics.items():
            assert np.isfinite([value["mean"], *value["ci95"]]).all(), name
            assert value["ci95"][0] <= value["ci95"][1]
        for name in ("auc", "found_fraction", "success"):
            assert 0 <= metrics[name]["mean"] <= 1 + 1e-9
        assert 0 <= metrics["energy_s"]["mean"] <= 500 + 1e-6
        assert 0 <= metrics["first_find_s"]["mean"] <= 500 + 1e-6
        curve = np.array(row["found_curve"])
        assert len(curve) == 101
        assert np.isfinite(curve).all()
        assert np.all(np.diff(curve) >= -1e-12)
        assert np.isclose(curve[-1], metrics["found_fraction"]["mean"])
        assert row["worlds"] > 0
        if row["method"] == "PPO":
            assert set(row["training_seeds"]) == {"17", "29", "43"}
        if row["group"] == "all":
            assert row["worlds"] == (40 if row["case"].startswith("stress-") else 80)
        if row["method"] == "oracle":
            random = lookup[(row["case"], row["variant"], "random", row["group"])]
            assert metrics["auc"]["mean"] > random["metrics"]["auc"]["mean"]
    old = read(Path("docs/m9-results/results.json"))
    for row in old["rows"]:
        if row["scenario"] != "nominal":
            continue
        reused = lookup[("original", row["variant"], row["method"], row["family"])]
        for name, value in row["metrics"].items():
            assert np.isclose(value["mean"], reused["metrics"][name]["mean"], atol=1e-12)
    pairs = read(Path("data/m9-stress/worlds/pairs.json"))["pairs"]
    assert len(pairs) == 40
    declared = exhausted = 0
    plan = read(args.run / "plan.json")
    jobs = keys(ExtensionProtocol.model_validate(plan["protocol"]), plan["manifest"])
    for key in jobs:
        if key["scenario"] != "declare":
            continue
        path = args.run / "episodes" / f"{digest(key)}.json"
        row = read(path)
        for before, after in zip(row["records"], row["records"][1:], strict=False):
            if after["action"] == 6:
                remaining = 500 - before["energy_s"]
                cost = min(2, remaining)
                assert np.isclose(after["time_s"] - before["time_s"], cost)
                assert np.isclose(after["energy_s"] - before["energy_s"], cost)
                assert after["path_m"] == before["path_m"]
                assert "variant" not in after["events"]
                if remaining < 2 - 1e-9:
                    assert after["energy_s"] == 500
                    assert after["found"] == before["found"]
                    assert after["false_declarations"] == before["false_declarations"]
                    assert after["events"]["found"] == after["events"]["false"] == 0
                    exhausted += 1
                else:
                    declared += 1
    audit = {
        "passed": True,
        "aggregate_rows": len(lookup),
        "paired_rows": len(result["paired"]),
        "new_episodes": 18720,
        "reference_episodes": 4320,
        "paid_declare_actions_checked": declared,
        "unaffordable_declare_attempts_checked": exhausted,
        "report_sha256": sha256_file(args.run / "report/results.json"),
    }
    write_json(args.run / "review-audit.json", audit)
    print(audit, flush=True)


if __name__ == "__main__":
    main()
