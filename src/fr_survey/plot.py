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
