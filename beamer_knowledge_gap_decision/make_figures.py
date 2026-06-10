from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "sufficient_information": "sufficient",
    "ambiguous_question": "ambiguous",
    "user_info_missing": "user-info\nmissing",
    "evidence_missing": "evidence\nmissing",
    "time_sensitive": "time\nsensitive",
    "false_premise": "false\npremise",
    "high_risk_or_expert_needed": "high-risk",
    "answer": "answer",
    "ask": "ask",
    "retrieve": "retrieve",
    "abstain": "abstain",
    "challenge_premise": "challenge\npremise",
}

COLORS = {
    "teal": "#0f766e",
    "cyan": "#0e7490",
    "green": "#2f855a",
    "amber": "#b45309",
    "red": "#b91c1c",
    "violet": "#6d5bd0",
    "gray": "#718096",
    "dark": "#1f2937",
}

CANDIDATE_LABELS = {
    "all_modules": "all modules",
    "no_evidence_sufficiency_verifier": "no sufficiency\nverifier",
    "no_evidence_side_features": "no evidence-side\nfeatures",
    "no_gap_type_classifier": "no gap-type\nclassifier",
}

ACTION_LABELS = {
    "answer": "answer",
    "ask": "ask",
    "retrieve": "retrieve",
    "abstain": "abstain",
    "challenge_premise": "challenge",
}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
    }
)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hbar(ax, labels, values, color, title, xlabel):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    xmax = max(values) if len(values) else 1
    for i, v in enumerate(values):
        text = f"{v:.0f}" if abs(v) >= 2 else f"{v:.3f}"
        ax.text(v + xmax * 0.015, i, text, va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.16)


def copy_existing_confusions() -> None:
    for name in ["confusion_matrix_action.png", "confusion_matrix_gap_type.png"]:
        src = ROOT / "results" / name
        if src.exists():
            shutil.copy2(src, OUT / name)


def plot_label_distribution() -> None:
    df = read_csv("data/processed/label_distribution.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, kind, title, color in [
        (axs[0], "gap_type", "Gap-type distribution", COLORS["teal"]),
        (axs[1], "gold_action", "Gold-action distribution", COLORS["green"]),
    ]:
        sub = df[df["label_type"] == kind].copy()
        sub["pretty"] = sub["label"].map(LABELS)
        hbar(ax, sub["pretty"], sub["count"], color, title, "Count")
    save(fig, "label_distribution.png")


def plot_method_comparison() -> None:
    metrics = read_csv("results/metrics_summary.csv")
    ok = metrics[metrics["status"] == "ok"].sort_values("action_macro_f1")
    colors = [
        COLORS["green"]
        if m == "Logistic Regression"
        else COLORS["amber"]
        if m == "Full Method"
        else COLORS["teal"]
        if m in {"Random Forest", "GBDT"}
        else COLORS["gray"]
        for m in ok["method"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    hbar(ax, ok["method"], ok["action_macro_f1"], colors, "Action macro-F1 by method", "Action macro-F1")
    ax.set_xlim(0, 1.03)
    save(fig, "method_action_f1.png")


def plot_utility_wrong_answer() -> None:
    metrics = read_csv("results/metrics_summary.csv")
    ok = metrics[metrics["status"] == "ok"].copy().sort_values("score")
    colors = [
        COLORS["green"]
        if m == "Logistic Regression"
        else COLORS["amber"]
        if m == "Full Method"
        else COLORS["red"]
        if m == "Always Answer"
        else COLORS["teal"]
        for m in ok["method"]
    ]
    labels = [
        m.replace(" Similarity Threshold", "").replace(" Classifier", "").replace(" Baseline", "")
        for m in ok["method"]
    ]
    y = np.arange(len(ok))
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axs[0].barh(y, ok["score"], color=colors, alpha=0.92)
    axs[0].axvline(0, color=COLORS["dark"], lw=1)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels)
    axs[0].set_xlabel("Utility score")
    axs[0].set_title("Utility")
    for i, v in enumerate(ok["score"]):
        if v < -1:
            axs[0].text(v + 0.08, i, f"{v:.3f}", va="center", ha="left", fontsize=8, color="white")
        else:
            ha = "left" if v >= 0 else "right"
            dx = 0.025 if v >= 0 else -0.025
            axs[0].text(v + dx, i, f"{v:.3f}", va="center", ha=ha, fontsize=8)
    axs[0].grid(axis="x", alpha=0.16)

    axs[1].barh(y, ok["wrong_answer_rate"], color=colors, alpha=0.92)
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([])
    axs[1].set_xlim(0, max(0.20, float(ok["wrong_answer_rate"].max()) + 0.08))
    axs[1].set_xlabel("Wrong-answer rate")
    axs[1].set_title("Premature answer risk")
    for i, v in enumerate(ok["wrong_answer_rate"]):
        axs[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    axs[1].grid(axis="x", alpha=0.16)
    fig.suptitle("Utility and wrong-answer risk", fontweight="bold", fontsize=14)
    save(fig, "utility_wrong_answer.png")


def plot_per_action_f1() -> None:
    per = read_csv("results/per_class_metrics.csv")
    methods = ["Prompted LLM Baseline", "Logistic Regression", "Random Forest", "GBDT", "Full Method"]
    labels = ["answer", "ask", "retrieve", "abstain", "challenge_premise"]
    sub = per[(per["task"] == "action") & (per["method"].isin(methods)) & (per["label"].isin(labels))]
    pivot = sub.pivot(index="label", columns="method", values="f1").loc[labels, methods]
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    x = np.arange(len(labels))
    width = 0.15
    palette = [COLORS["gray"], COLORS["green"], COLORS["teal"], COLORS["violet"], COLORS["amber"]]
    for i, method in enumerate(methods):
        ax.bar(x + (i - 2) * width, pivot[method], width=width, label=method, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[l] for l in labels])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("Per-action F1")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.16)
    save(fig, "per_action_f1.png")


def plot_ablation() -> None:
    df = read_csv("results/ablation_summary.csv").sort_values("delta_score")
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    colors = [COLORS["red"] if v < -0.05 else COLORS["gray"] for v in df["delta_score"]]
    hbar(ax, df["variant"].str.replace("Full - ", "", regex=False), df["delta_score"], colors, "Ablation: utility delta vs Full", "Delta utility score")
    ax.axvline(0, color=COLORS["dark"], lw=1)
    ax.set_xlim(min(-0.65, df["delta_score"].min() - 0.05), 0.08)
    save(fig, "ablation_delta.png")


def plot_significance() -> None:
    df = read_csv("results/significance_tests.csv")
    df["baseline"] = df["comparison"].str.replace("Full Method vs ", "", regex=False)
    df = df.sort_values("macro_f1_diff")
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    y = np.arange(len(df))
    ax.hlines(y, df["ci_low"], df["ci_high"], color=COLORS["gray"], lw=2)
    ax.scatter(df["macro_f1_diff"], y, color=[COLORS["green"] if lo > 0 else COLORS["amber"] for lo in df["ci_low"]], s=72, zorder=3)
    ax.axvline(0, color=COLORS["dark"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["baseline"])
    ax.set_xlabel("Action macro-F1 difference: Full - baseline")
    ax.set_title("Paired bootstrap 95% intervals")
    ax.grid(axis="x", alpha=0.16)
    save(fig, "significance_forest.png")


def plot_stability() -> None:
    df = read_csv("results/repeated_seed_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(df["seed"].astype(str), df["action_macro_f1"], marker="o", lw=2.5, color=COLORS["teal"], label="action macro-F1")
    ax.plot(df["seed"].astype(str), df["score"], marker="s", lw=2.5, color=COLORS["green"], label="utility")
    ax.set_ylim(0.75, 0.88)
    ax.set_xlabel("Split seed")
    ax.set_title("Repeated split stability")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False)
    save(fig, "stability.png")


def plot_error_categories() -> None:
    path = ROOT / "results/predictions_test.csv"
    if path.exists():
        pred = pd.read_csv(path)
        errors = pred[pred["gold_action"] != pred["pred_action"]].copy()
        labels = [
            f"{ACTION_LABELS.get(g, g)} -> {ACTION_LABELS.get(p, p)}"
            for g, p in zip(errors["gold_action"], errors["pred_action"])
        ]
    else:
        fallback = ROOT / "results/error_analysis.json"
        raw = json.loads(fallback.read_text(encoding="utf-8")) if fallback.exists() else []
        labels = [
            f"{ACTION_LABELS.get(e.get('gold_action', ''), e.get('gold_action', 'gold'))}"
            f" -> {ACTION_LABELS.get(e.get('pred_action', ''), e.get('pred_action', 'pred'))}"
            for e in raw
        ]
    counts = pd.Series(labels).value_counts()
    if counts.empty:
        counts = pd.Series({"No action errors": 1})
    counts = counts.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = [
        COLORS["red"] if "-> answer" in label and not label.startswith("answer") else COLORS["amber"]
        for label in counts.index
    ]
    hbar(ax, counts.index, counts.values, colors, "Full Method residual action transitions", "Count")
    save(fig, "error_categories.png")


def plot_validation_selection() -> None:
    df = read_csv("results/full_method_selection.csv")
    selected = df.sort_values(["action_macro_f1", "score"], ascending=False).iloc[0]["candidate"]
    labels = [CANDIDATE_LABELS.get(c, c.replace("_", "\n")) for c in df["candidate"]]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    height = 0.34
    action_colors = [COLORS["green"] if c == selected else COLORS["teal"] for c in df["candidate"]]
    score_colors = [COLORS["amber"] if c == selected else COLORS["gray"] for c in df["candidate"]]
    ax.barh(y - height / 2, df["action_macro_f1"], height=height, color=action_colors, label="action macro-F1")
    ax.barh(y + height / 2, df["score"], height=height, color=score_colors, label="utility score")
    for i, (action, score) in enumerate(zip(df["action_macro_f1"], df["score"])):
        ax.text(action + 0.012, i - height / 2, f"{action:.3f}", va="center", fontsize=8)
        ax.text(score + 0.012, i + height / 2, f"{score:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.72, float(df[["action_macro_f1", "score"]].to_numpy().max()) + 0.08))
    ax.set_xlabel("Validation metric")
    ax.set_title("Validation selection: highest action macro-F1 wins")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.16)
    save(fig, "validation_selection.png")


def main() -> None:
    copy_existing_confusions()
    plot_label_distribution()
    plot_method_comparison()
    plot_utility_wrong_answer()
    plot_per_action_f1()
    plot_ablation()
    plot_significance()
    plot_stability()
    plot_error_categories()
    plot_validation_selection()


if __name__ == "__main__":
    main()
