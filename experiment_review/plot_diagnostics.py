#!/usr/bin/env python3
"""Plot saved review measurements; does not execute or modify any notebook.

Example:
  python3 experiment_review/plot_diagnostics.py --data-dir experiment_review \
    --log /path/to/gauss_embed_dlm_wass_10m_prior_run.log

Inputs are checkpoint_metrics.json and context_metrics.json in --data-dir,
plus the legacy 10m text log. Outputs are dlm_diagnostics.png and .pdf.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dlm_review_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--log", type=Path, default=Path(__file__).resolve().parent.parent / "gauss_embed_dlm_wass_10m_prior_run.log")
    args = parser.parse_args()
    records = json.loads((args.data_dir / "checkpoint_metrics.json").read_text())
    model = next(r for r in records if r["checkpoint"] == "gauss_embed_dlm_wass_5m.pt")
    contexts = json.loads((args.data_dir / "context_metrics.json").read_text())
    context = next(r for r in contexts if r["checkpoint"] == model["checkpoint"])
    log_rows = []
    for line in args.log.read_text().splitlines():
        match = re.match(r"step (\d+)/(\d+)", line)
        if match:
            values = {k: float(v) for k, v in re.findall(r"(\w+)=([-+\d.eE]+)", line)}
            log_rows.append({"step": int(match[1]), **values})
    if not log_rows:
        raise ValueError(f"No training rows found in {args.log}")
    assert model["optimizer_steps"] == [10000], "Update chart's checkpoint label for new data."

    blue, orange, teal, gray = "#2563A6", "#D87627", "#23867F", "#677385"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BAC1C9", "axes.labelcolor": "#273240",
        "xtick.color": "#465362", "ytick.color": "#465362",
        "text.color": "#202B38", "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.3))
    fig.subplots_adjust(left=.075, right=.975, bottom=.20, top=.845, hspace=.53, wspace=.29)
    fig.suptitle("Denoising diagnostics: objective progress is not language progress", x=.075, y=.976,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.075, .933,
             "5M checkpoint: 10,000 optimizer steps  |  CPU reconstruction and paired context tests\n"
             "Training-distribution diagnostics, not held-out evaluation. 10M panel uses a legacy log with source mismatch.",
             fontsize=10.5, color=gray, va="top", linespacing=1.6)

    ax = axs[0, 0]
    steps = np.array([r["step"] for r in log_rows]) / 1000
    ax.plot(steps, [r["weighted_prior"] for r in log_rows], color=orange, lw=2, label="Weighted prior loss")
    ax.plot(steps, [r["weighted_mse"] for r in log_rows], color=blue, lw=2.7, label="Denoiser MSE")
    ax.plot(steps, [r["variance_avg"] for r in log_rows], color="#273240", lw=1.5,
            linestyle=(0, (4, 3)), label="Aggregate variance (mean-predictor baseline)")
    ax.set(title="A   10M legacy run: the prior improves; denoising stalls",
           xlabel="Logged optimizer step (thousands)", ylabel="Loss / variance per coordinate",
           xlim=(0, 9.5), ylim=(0, 2.6))
    ax.legend(loc="upper right", fontsize=8.6)
    ax.annotate("At step 9,250:\nMSE = variance = 0.9071", xy=(9.25, .9071), xytext=(4.5, 1.4),
                fontsize=9.5, arrowprops={"arrowstyle": "->", "color": gray}, color=blue)
    ax.grid(axis="y", alpha=.17)

    ax = axs[0, 1]
    ts = np.array([r["t"] for r in model["timesteps"]])
    trained = np.array([r["mse"] for r in model["timesteps"]])
    analytic = np.array([r["analytic_tokenwise_mse"] for r in model["timesteps"]])
    ax.loglog(ts, trained, "o-", color=blue, lw=2, markersize=5, label="5M denoiser")
    ax.loglog(ts, analytic, "s--", color=teal, lw=2, markersize=4.5,
              label="Analytic independent-token estimator")
    ax.set(title="B   5M model trails a baseline that uses no context",
           xlabel="Diffusion timestep (log scale)", ylabel="Reconstruction MSE (log scale)",
           xlim=(.8, 1300), ylim=(6e-5, 1.5))
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(which="major", alpha=.17)
    ax.legend(loc="lower right", fontsize=8.7)
    ax.annotate(f"t = 1: {trained[0] / analytic[0]:.0f}× baseline MSE\n(both decode tokens at 100%)",
                xy=(1, trained[0]), xytext=(3.3, .008), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": gray})

    ax = axs[1, 0]
    between = np.array([.95, model["between_token_variance"]])
    within = np.array([.05, model["within_token_variance"]])
    xs = np.array([0, 1])
    ax.bar(xs, between, color=blue, width=.54, label="Variation between token means")
    ax.bar(xs, within, bottom=between, color=orange, width=.54, label="Noise within a token")
    for i in range(2):
        ax.text(xs[i], between[i] / 2, f"{between[i]:.3f}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(xs[i], between[i] + within[i] / 2, f"{within[i]:.3f}", ha="center", va="center",
                color="white" if i else "#202B38", fontsize=9, fontweight="bold")
    ax.set(title="C   5M embeddings allocate over half their variance to noise",
           ylabel="Variance per coordinate", ylim=(0, 1.19), xlim=(-.7, 1.9))
    ax.set_xticks(xs, ["Nominal initialization*", "Saved 5M checkpoint"])
    frac = within[1] / (between[1] + within[1])
    ax.text(1, 1.025, f"{frac:.1%} within-token noise", ha="center", fontsize=9.5, color=orange, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-.01, -.21), fontsize=8.9, ncol=1)
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)

    ax = axs[1, 1]
    context_ts = sorted({r["t"] for r in context["rows"]})
    deltas = [np.array([100 * (r["accuracy"] - r["shuffled_accuracy"])
                        for r in context["rows"] if r["t"] == t]) for t in context_ts]
    means = np.array([x.mean() for x in deltas])
    for t, values in zip(context_ts, deltas):
        ax.scatter(np.full(len(values), t), values, color=blue, alpha=.45, s=28, zorder=3)
    ax.plot(context_ts, means, "o-", color=blue, markersize=6, lw=2, label="Mean of 3 paired noise seeds")
    ax.axhline(0, color=gray, linewidth=1.2)
    ax.set(title="D   5M context helps only modestly in this diagnostic",
           xlabel="Diffusion timestep", ylabel="Accuracy: original − shuffled context (pp)",
           xlim=(60, 640), ylim=(-.35, 1.85))
    ax.set_xticks(context_ts)
    ax.grid(axis="y", alpha=.17)
    ax.legend(loc="upper left", fontsize=8.7)
    middle = context_ts.index(500)
    ax.annotate(f"t = 500: +{means[middle]:.2f} percentage points", xy=(500, means[middle]),
                xytext=(175, 1.43), fontsize=9.2,
                arrowprops={"arrowstyle": "->", "color": gray})
    ax.text(.02, .025, "Each token keeps its own noisy vector and position;\nneighboring context is randomized across sequences.",
            transform=ax.transAxes, fontsize=8.5, color=gray, va="bottom")

    fig.text(.075, .037,
             "* Nominal initialization uses mean-coordinate variance 0.95 and within-token variance 0.05; the initial corpus-weighted\n"
             "   between-token variance was not measured. Checkpoint variance is weighted by corpus token frequencies.\n"
             "B: one 2,048-token diagnostic batch. D: same 4,096 tokens across 3 noise seeds; dots are seed results, not confidence intervals.",
             fontsize=8.4, color=gray, va="bottom", linespacing=1.45)
    for suffix in ("png", "pdf"):
        output = args.data_dir / f"dlm_diagnostics.{suffix}"
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
