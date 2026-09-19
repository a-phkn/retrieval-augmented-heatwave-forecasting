"""
Generates PNG charts for exactly the 4 metrics this project's evaluation
should be judged on (see training/evaluate_lstm.py's docstring and
context/ml_notes.md for the reasoning behind narrowing to these 4 --
R^2, MAPE, per-horizon breakdown, plain accuracy, F1, and the raw
confusion matrix were all deliberately dropped as either redundant or not
decision-relevant here).

Reads: evaluation/baseline_lstm/val_extended_metrics.json (all 4 metrics
now live in this single file, produced by training/evaluate_lstm.py).
Does NOT recompute anything -- pure visualization.

Run from repo root:
    python -m training.visualize_results

Outputs (PNG, 150 dpi) to evaluation/figures/:
    01_global_accuracy.png    -- MAE/RMSE, persistence vs climatology vs LSTM
    02_stratified_rmse.png    -- RMSE by normal/unusual/extreme, all 3 methods
                                  (THE headline chart -- this is the metric
                                  RAG has to beat later)
    03_detection.png          -- precision/recall/F2, all 3 methods
    04_bias_diagnostic.png    -- LSTM bias on hot vs normal days
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"
FIG_DIR = EVAL_DIR / "figures"

COLOR_LSTM = "#2B6CB0"
COLOR_PERSIST = "#2F9E8F"
COLOR_CLIM = "#8562C9"
COLOR_HOT = "#C0392B"
COLOR_NORMAL = "#2F9E8F"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#4A5268", "axes.labelcolor": "#171B26", "text.color": "#171B26",
    "xtick.color": "#4A5268", "ytick.color": "#4A5268",
    "axes.grid": True, "grid.color": "#ECEEF3", "grid.linewidth": 0.8,
    "font.size": 11, "axes.axisbelow": True,
})


def bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)


def plot_global_accuracy(m: dict) -> None:
    """Metric 1: credibility floor."""
    metrics = ["MAE", "RMSE"]
    persistence = [m["persistence"]["mae"], m["persistence"]["rmse"]]
    climatology = [m["climatology"]["mae"], m["climatology"]["rmse"]]
    lstm = [m["lstm"]["mae"]["mean"], m["lstm"]["rmse"]["mean"]]
    lstm_err = [m["lstm"]["mae"]["std"], m["lstm"]["rmse"]["std"]]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width, persistence, width, label="Persistence", color=COLOR_PERSIST)
    b2 = ax.bar(x, climatology, width, label="Climatology", color=COLOR_CLIM)
    b3 = ax.bar(x + width, lstm, width, yerr=lstm_err, capsize=4, label="LSTM", color=COLOR_LSTM)
    bar_labels(ax, b1); bar_labels(ax, b2); bar_labels(ax, b3)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Degrees C")
    ax.set_title("1. Global accuracy — is the baseline sane at all?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_global_accuracy.png", dpi=150)
    plt.close(fig)


def plot_stratified_rmse(m: dict) -> None:
    """Metric 2: the headline metric this project's conclusions rest on."""
    strata = ["normal", "unusual", "extreme"]
    persistence = [m["persistence"][s]["rmse"] for s in strata]
    climatology = [m["climatology"][s]["rmse"] for s in strata]
    lstm = [m["lstm"][s]["rmse_mean"] for s in strata]
    lstm_err = [m["lstm"][s]["rmse_std"] for s in strata]
    n_per_stratum = [m["lstm"][s]["n"] for s in strata]

    x = np.arange(len(strata))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - width, persistence, width, label="Persistence", color=COLOR_PERSIST)
    b2 = ax.bar(x, climatology, width, label="Climatology", color=COLOR_CLIM)
    b3 = ax.bar(x + width, lstm, width, yerr=lstm_err, capsize=4, label="LSTM", color=COLOR_LSTM)
    bar_labels(ax, b1); bar_labels(ax, b2); bar_labels(ax, b3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(["Normal", "Unusual", "Extreme"], n_per_stratum)])
    ax.set_ylabel("RMSE (degrees C)")
    ax.set_title("2. Stratified RMSE — the metric RAG has to beat")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_stratified_rmse.png", dpi=150)
    plt.close(fig)


def plot_detection(m: dict) -> None:
    """Metric 3: supporting diagnostic, not the optimization target."""
    metrics = ["Precision", "Recall", "F2"]
    persistence = [m["persistence"]["precision"], m["persistence"]["recall"], m["persistence"]["f2"]]
    climatology = [np.nan, m["climatology"]["recall"], np.nan]
    lstm = [m["lstm"]["precision"], m["lstm"]["recall"], m["lstm"]["f2"]]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width, persistence, width, label="Persistence", color=COLOR_PERSIST)
    b2 = ax.bar(x, climatology, width, label="Climatology", color="#CBD2E0")
    b3 = ax.bar(x + width, lstm, width, label="LSTM", color=COLOR_LSTM)
    bar_labels(ax, b1); bar_labels(ax, b2); bar_labels(ax, b3)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.7)
    ax.set_title("3. Heatwave detection — diagnostic, not the target metric")
    ax.legend(frameon=False)
    ax.annotate("F2 weights recall 2x precision (matches this project's stated priority)\n"
                "Climatology's recall is mechanically 0 by construction",
                xy=(0.5, 0.02), xycoords="axes fraction", fontsize=8, color="#4A5268", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_detection.png", dpi=150)
    plt.close(fig)


def plot_bias_diagnostic(bias: dict) -> None:
    """Metric 4: explains why #3 looks the way it does."""
    categories = [f"Actual hot days\n(n={bias['n_hot']:,})", f"Normal days\n(n={bias['n_normal']:,})"]
    values = [bias["mean_bias_on_hot_days"], bias["mean_bias_on_normal_days"]]
    colors = [COLOR_HOT, COLOR_NORMAL]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(categories, values, color=colors)
    ax.axvline(0, color="#4A5268", linewidth=1)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:+.2f}°C", (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(6 if v >= 0 else -6, 0),
                    ha="left" if v >= 0 else "right", va="center", fontsize=10)
    ax.set_xlabel("Mean bias, predicted - actual (degrees C)")
    ax.set_title("4. Bias diagnostic — why detection looks the way it does")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_bias_diagnostic.png", dpi=150)
    plt.close(fig)


def main():
    with open(EVAL_DIR / "baseline_lstm" / "val_extended_metrics.json") as f:
        result = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_global_accuracy(result["1_global_accuracy"])
    plot_stratified_rmse(result["2_stratified_rmse"])
    plot_detection(result["3_detection"])
    plot_bias_diagnostic(result["4_bias_diagnostic"])

    print(f"Saved 4 figures to {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()