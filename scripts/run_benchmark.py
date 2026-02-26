"""Main benchmark script: load LFW, evaluate all models, save results, plot ROC."""

import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fr_survey.dataset import load_lfw_pairs
from fr_survey.evaluate import evaluate_model
from fr_survey.models.base import FaceRecognitionModel, ModelResult
from fr_survey.models.opencv_sface import OpenCVSFace
from fr_survey.models.auraface import AuraFace
from fr_survey.models.dlib_resnet import DlibResNet, DLIB_AVAILABLE
from fr_survey.models.deepface_ghost import DeepFaceGhostFaceNet
from fr_survey.plot import plot_roc_curves

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _build_models() -> list[FaceRecognitionModel]:
    models: list[FaceRecognitionModel] = [
        OpenCVSFace(),
        AuraFace(),
        DeepFaceGhostFaceNet(),
    ]
    if DLIB_AVAILABLE:
        models.append(DlibResNet())
    else:
        print("⚠ dlib not available — skipping Dlib ResNet-34\n")
    return models


def _result_path(model_name: str) -> Path:
    """Per-model result JSON path."""
    safe_name = model_name.replace(" ", "_").lower()
    return RESULTS_DIR / f"{safe_name}.json"


def _save_model_result(result: ModelResult) -> None:
    """Save a single model's result to its own JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "model_name": result.model_name,
        "auc": result.auc,
        "eer": result.eer,
        "tar_at_far_001": result.tar_at_far_001,
        "tar_at_far_0001": result.tar_at_far_0001,
        "num_pairs": result.num_pairs,
        "num_skipped": result.num_skipped,
        "detection_failure_rate": result.detection_failure_rate,
        "fpr": result.fpr.tolist(),
        "tpr": result.tpr.tolist(),
    }
    path = _result_path(result.model_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


def _load_model_result(model_name: str) -> ModelResult | None:
    """Load a previously saved model result, or return None."""
    path = _result_path(model_name)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return ModelResult(
        model_name=data["model_name"],
        fpr=np.array(data["fpr"]),
        tpr=np.array(data["tpr"]),
        thresholds=np.array([]),
        auc=data["auc"],
        eer=data["eer"],
        tar_at_far_001=data["tar_at_far_001"],
        tar_at_far_0001=data["tar_at_far_0001"],
        num_pairs=data["num_pairs"],
        num_skipped=data["num_skipped"],
    )


def _save_combined_results(results: list[ModelResult]) -> None:
    """Save combined benchmark_results.json for the notebook."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for r in results:
        data[r.model_name] = {
            "auc": r.auc,
            "eer": r.eer,
            "tar_at_far_001": r.tar_at_far_001,
            "tar_at_far_0001": r.tar_at_far_0001,
            "num_pairs": r.num_pairs,
            "num_skipped": r.num_skipped,
            "detection_failure_rate": r.detection_failure_rate,
            "fpr": r.fpr.tolist(),
            "tpr": r.tpr.tolist(),
        }
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {out_path}")


def _print_summary(results: list[ModelResult]) -> None:
    header = f"{'Model':<25} {'AUC':>8} {'EER':>8} {'TAR@FAR=1%':>12} {'TAR@FAR=0.1%':>14} {'Pairs':>7} {'Skip':>6} {'Fail%':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model_name:<25} "
            f"{r.auc:>8.4f} "
            f"{r.eer:>8.4f} "
            f"{r.tar_at_far_001:>12.4f} "
            f"{r.tar_at_far_0001:>14.4f} "
            f"{r.num_pairs:>7d} "
            f"{r.num_skipped:>6d} "
            f"{r.detection_failure_rate:>6.2%}"
        )
    print("=" * len(header))


def main() -> None:
    models = _build_models()
    results: list[ModelResult] = []
    need_eval = False

    # Check which models already have saved results
    for model in models:
        cached = _load_model_result(model.name)
        if cached is not None:
            print(f"[cached] {model.name} — loaded from {_result_path(model.name)}")
            results.append(cached)
        else:
            need_eval = True
            results.append(model)  # placeholder — will be replaced

    if not need_eval:
        print("\nAll models already evaluated.\n")
    else:
        print("\nLoading LFW pairs dataset ...")
        pairs = load_lfw_pairs()
        print(f"  Loaded {len(pairs)} pairs\n")

        for i, entry in enumerate(results):
            if isinstance(entry, ModelResult):
                continue  # already cached
            model = entry
            print(f"Setting up {model.name} ...")
            try:
                model.setup()
            except RuntimeError as e:
                print(f"  Skipping {model.name}: {e}\n")
                results[i] = None
                continue
            print(f"Evaluating {model.name} ...")
            result = evaluate_model(model, pairs)
            _save_model_result(result)
            results[i] = result
            print(f"  AUC={result.auc:.4f}  EER={result.eer:.4f}  "
                  f"Skipped={result.num_skipped}/{result.num_pairs + result.num_skipped}\n")
            del model
            gc.collect()

    # Filter out skipped models
    results = [r for r in results if isinstance(r, ModelResult)]

    print("Saving combined results ...")
    _save_combined_results(results)

    print("Plotting ROC curves ...")
    plot_roc_curves(results)

    _print_summary(results)


if __name__ == "__main__":
    main()
