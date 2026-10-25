"""Draw Fig. 2 (accuracy per round), Fig. 3 (round-time breakdown) and Fig. 4 (poisoning)
from the JSON files written by run_experiments.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN, GRAY = "#2f6fdb", "#ef6c3a", "#22a77a", "#888888"


def _load(out, name):
    p = Path(out) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig2(out):
    c = _load(out, "fig2_curves")
    if not c:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, (key, title) in zip(axes, [("iid", "(a) IID"), ("noniid", "(b) Non-IID (Dirichlet 0.5)")]):
        r = range(1, len(c[key]["FL (FedAvg)"]) + 1)
        ax.plot(r, c[key]["FL (FedAvg)"], color=BLUE, lw=2, label="FL (FedAvg)")
        ax.plot(r, c[key]["BC-FL screen"], color=ORANGE, lw=2, ls="--", label="BC-FL")
        ax.axhline(c[key]["Centralized"], color=GRAY, ls=":", label="Centralized")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylim(0.85, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out) / "fig2_accuracy.png", dpi=200)
    plt.close(fig)


def fig3(out):
    b = _load(out, "fig3_breakdown")
    if not b:
        return
    parts = [("train", "Local training", BLUE), ("hash_sign", "Hash + sign", "#f2b134"),
             ("contract", "Contract checks", ORANGE), ("screen_verify", "Screening + verify", "#e86fa8"),
             ("block_commit", "Block commit (3 peers)", GREEN), ("bookkeeping", "Round bookkeeping", GRAY)]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    left = 0.0
    for key, label, col in parts:
        ax.barh("BC-FL", b[key], left=left, color=col, label=label)
        left += b[key]
    ax.text(left + 0.15, 0, f"{b['round']:.2f} ms", va="center", fontsize=8)
    ax.barh("FL", b["fl_round"], color=BLUE)
    ax.text(b["fl_round"] + 0.15, 1, f"{b['fl_round']:.2f} ms", va="center", fontsize=8)
    ax.set_xlabel("Mean wall-clock time per round (ms)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.45), frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig3_overhead.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4(out):
    p = _load(out, "table7_poisoning")
    if not p:
        return
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, (key, title) in zip(axes, [("iid", "(a) IID"), ("noniid", "(b) Non-IID")]):
        labels = list(p[key])
        x = np.arange(len(labels))
        for i, (col, name, hatch, lab) in enumerate([("FL", BLUE, "", "FL (FedAvg)"),
                                                     ("Audit", ORANGE, "////", "BC-FL (audit only)"),
                                                     ("Screen", GREEN, "", "BC-FL (audit+screen)")]):
            vals = [p[key][l][col] / 100 for l in labels]
            errs = [p[key][l][col + "_std"] / 100 for l in labels]
            ax.bar(x + (i - 1) * 0.26, vals, 0.26, yerr=errs, color=name, hatch=hatch,
                   edgecolor="white", capsize=2, label=lab)
        ax.set_xticks(x, [l.replace(" ×5", "").replace(" ", "\n") for l in labels], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(Path(out) / "fig4_poisoning.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all(out="results"):
    fig2(out)
    fig3(out)
    fig4(out)


if __name__ == "__main__":
    make_all(sys.argv[1] if len(sys.argv) > 1 else "results")
