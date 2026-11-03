"""M11 figures from saved, published results only; no episode is run or rerun.

Writes docs/figures/m11_found_curves.png (M9 nominal V3 found-fraction curves per
family) and docs/figures/m10_transfer.png (Gazebo versus cached comparison).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated reference palette, first three categorical slots (light surface).
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
REFERENCE = "#8f8e88"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def found_curves(repo: Path, out: Path) -> None:
    rows = json.loads((repo / "docs/m9-results/results.json").read_text())["rows"]
    pick = {
        (r["family"], r["method"]): r["found_curve"]
        for r in rows
        if r["scenario"] == "nominal" and r["variant"] == "V3" and r["family"] != "all"
    }
    families = sorted({f for f, _ in pick})
    t = np.linspace(0, 500, 101)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, family in zip(axes.flat, families, strict=True):
        style(ax)
        for method, label, dash in (
            ("oracle", "Oracle (privileged reference)", (0, (5, 3))),
            ("random", "Random", (0, (1, 2))),
        ):
            ax.plot(
                t,
                pick[(family, method)],
                color=REFERENCE,
                linewidth=1.5,
                linestyle=dash,
                label=label,
            )
        for (method, label), colour, marker in zip(
            (("B3", "B3"), ("B5", "B5"), ("PPO", "PPO (3 seeds)")), SERIES, MARKERS, strict=True
        ):
            ax.plot(
                t,
                pick[(family, method)],
                color=colour,
                linewidth=2,
                label=label,
                marker=marker,
                markevery=20,
                markersize=6,
            )
        ax.set_title(family.replace("-", ", "), fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("Mission time (s), 500 s battery", fontsize=9, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of targets found", fontsize=9, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Learned/heuristic series first, then the grey references.
    order = [2, 3, 4, 0, 1]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Found fraction over time, test worlds, V3 sensor (M9 nominal; 20 worlds per family)",
        fontsize=11,
        color=INK,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "m11_found_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def transfer(repo: Path, out: Path) -> None:
    tables = json.loads((repo / "docs/m10-transfer-results/results.json").read_text())["tables"]
    auc = tables["summary"]["auc"]
    groups = (
        ("B5", "B5"),
        ("PPO-17", "PPO 17"),
        ("PPO-29", "PPO 29"),
        ("PPO-43", "PPO 43"),
        ("PPO", "PPO mean"),
    )
    conditions = (
        ("cached", "Cached simulator"),
        ("live-original", "Gazebo, original sensor"),
        ("live-refit", "Gazebo, refit sensor"),
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    style(left)
    x = np.arange(len(groups))
    for ci, ((condition, label), colour, marker) in enumerate(
        zip(conditions, SERIES, MARKERS, strict=True)
    ):
        means = np.array([auc[g][condition]["mean"] for g, _ in groups])
        low = means - np.array([auc[g][condition]["ci95"][0] for g, _ in groups])
        high = np.array([auc[g][condition]["ci95"][1] for g, _ in groups]) - means
        left.errorbar(
            x + (ci - 1) * 0.22,
            means,
            yerr=[low, high],
            fmt=marker,
            color=colour,
            markersize=8,
            elinewidth=1.5,
            capsize=3,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
        )
    left.set_xticks(x, [name for _, name in groups])
    left.set_ylim(0, 1)
    left.set_ylabel("Found-fraction AUC (95% world bootstrap)", fontsize=9, color=MUTED)
    left.set_title(
        "Search performance, 24 held-out test worlds", fontsize=10, color=INK, loc="left"
    )
    left.legend(frameon=False, fontsize=8, loc="lower left")

    style(right)
    levels = [row for row in tables["detector_levels"] if row["recall"]["live_rate"] is not None]
    names = [row["level"] for row in levels]
    xl = np.arange(len(levels))
    for ci, (key, label) in enumerate(
        (("cached_rate", "Cached simulator"), ("live_rate", "Gazebo, live detector"))
    ):
        right.plot(
            xl,
            [row["recall"][key] for row in levels],
            color=SERIES[ci],
            marker=MARKERS[ci],
            markersize=8,
            linewidth=2,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
        )
    for i, row in enumerate(levels):
        right.annotate(
            f"n={row['recall']['live']['trials']}",
            (i, row["recall"]["live_rate"]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=MUTED,
        )
    right.set_xticks(xl, names)
    right.set_ylim(0, 1)
    right.set_xlabel("Altitude level (higher = coarser)", fontsize=9, color=MUTED)
    right.set_ylabel("Cell recall at the operating threshold", fontsize=9, color=MUTED)
    right.set_title("Detector on the same executed views", fontsize=10, color=INK, loc="left")
    right.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "m10_transfer.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "docs/figures"
    found_curves(repo, out)
    transfer(repo, out)
    print("wrote", out / "m11_found_curves.png", out / "m10_transfer.png")


if __name__ == "__main__":
    main()
