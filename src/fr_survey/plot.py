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
