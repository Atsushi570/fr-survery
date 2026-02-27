"""ROC curve plotting for face recognition benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models.base import ModelResult

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def plot_roc_curves(results: list[ModelResult], output_dir: Path | None = None) -> None:
    """Plot ROC curves for all models (linear + log scale)."""
    if output_dir is None:
        output_dir = _RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Linear scale ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for r in results:
        if len(r.fpr) == 0:
            continue
        ax.plot(r.fpr, r.tpr, label=f"{r.model_name} (AUC={r.auc:.4f}, EER={r.eer:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Face Recognition Models on LFW")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'roc_curves.png'}")

    # --- Log scale (FAR axis) ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for r in results:
        if len(r.fpr) == 0:
            continue
        # Filter out zero FPR for log scale
        mask = r.fpr > 0
        ax.plot(
            r.fpr[mask],
            r.tpr[mask],
            label=f"{r.model_name} (AUC={r.auc:.4f}, EER={r.eer:.4f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate (log scale)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Log Scale) — Face Recognition Models on LFW")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(left=1e-4)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_log.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'roc_curves_log.png'}")


def plot_speed_comparison(
    results: list[ModelResult], output_dir: Path | None = None
) -> None:
    """Plot embedding speed comparison (mean / median) as grouped bar chart."""
    if output_dir is None:
        output_dir = _RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to results that have timing data
    timed = [r for r in results if r.timing is not None and r.timing.num_timed_embeddings > 0]
    if not timed:
        print("  No timing data available — skipping speed comparison plot.")
        return

    names = [r.model_name for r in timed]
    means = [r.timing.avg_embedding_time_ms for r in timed]
    medians = [r.timing.median_embedding_time_ms for r in timed]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_mean = ax.bar(x - width / 2, means, width, label="Mean")
    bars_median = ax.bar(x + width / 2, medians, width, label="Median")

    ax.set_ylabel("Embedding Time (ms)")
    ax.set_title("Embedding Speed Comparison — Face Recognition Models")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars_mean:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars_median:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / "speed_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_roc_curves_by_race(
    race_results: dict[str, list[ModelResult]],
    output_dir: Path | None = None,
) -> None:
    """Plot ROC curves per race, with all models overlaid in each subplot.

    Args:
        race_results: Mapping from race name to list of ModelResult (one per model).
    """
    if output_dir is None:
        output_dir = _RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    races = list(race_results.keys())
    n = len(races)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, race in enumerate(races):
        ax = axes[i]
        for r in race_results[race]:
            if len(r.fpr) == 0:
                continue
            ax.plot(r.fpr, r.tpr, label=f"{r.model_name} (AUC={r.auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {race}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "roc_curves_by_race.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_race_metric_comparison(
    race_results: dict[str, list[ModelResult]],
    output_dir: Path | None = None,
) -> None:
    """Plot grouped bar charts comparing AUC and EER across races and models."""
    if output_dir is None:
        output_dir = _RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    races = list(race_results.keys())
    # Get model names from the first race's results
    model_names = [r.model_name for r in race_results[races[0]]]

    x = np.arange(len(races))
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, (ax_auc, ax_eer) = plt.subplots(1, 2, figsize=(14, 6))

    # AUC comparison
    for i, model_name in enumerate(model_names):
        aucs = []
        for race in races:
            match = [r for r in race_results[race] if r.model_name == model_name]
            aucs.append(match[0].auc if match else 0.0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax_auc.bar(x + offset, aucs, width, label=model_name)
        for bar in bars:
            h = bar.get_height()
            ax_auc.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=90,
            )

    ax_auc.set_ylabel("AUC")
    ax_auc.set_title("AUC by Race")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(races)
    ax_auc.legend(fontsize=8)
    ax_auc.grid(True, alpha=0.3, axis="y")
    ax_auc.set_ylim(bottom=0.9)

    # EER comparison
    for i, model_name in enumerate(model_names):
        eers = []
        for race in races:
            match = [r for r in race_results[race] if r.model_name == model_name]
            eers.append(match[0].eer if match else 0.0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax_eer.bar(x + offset, eers, width, label=model_name)
        for bar in bars:
            h = bar.get_height()
            ax_eer.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=90,
            )

    ax_eer.set_ylabel("EER")
    ax_eer.set_title("EER by Race")
    ax_eer.set_xticks(x)
    ax_eer.set_xticklabels(races)
    ax_eer.legend(fontsize=8)
    ax_eer.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = output_dir / "race_metric_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
