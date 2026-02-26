"""Main benchmark script: load LFW, evaluate all models, save results, plot ROC."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fr_survey.dataset import load_lfw_pairs
from fr_survey.evaluate import evaluate_model
from fr_survey.models.base import FaceRecognitionModel, ModelResult, TimingResult
from fr_survey.models.opencv_sface import OpenCVSFace
from fr_survey.models.auraface import AuraFace
from fr_survey.models.dlib_resnet import DlibResNet, DLIB_AVAILABLE
from fr_survey.models.deepface_ghost import DeepFaceGhostFaceNet
from fr_survey.plot import plot_roc_curves, plot_speed_comparison

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Recognition Benchmark")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit number of evaluation pairs (useful for Raspberry Pi)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and re-evaluate all models",
    )
    return parser.parse_args()


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
    if result.timing is not None:
        data["timing"] = {
            "setup_time_s": result.timing.setup_time_s,
            "total_eval_time_s": result.timing.total_eval_time_s,
            "avg_embedding_time_ms": result.timing.avg_embedding_time_ms,
            "median_embedding_time_ms": result.timing.median_embedding_time_ms,
            "std_embedding_time_ms": result.timing.std_embedding_time_ms,
            "avg_pair_time_ms": result.timing.avg_pair_time_ms,
            "num_timed_embeddings": result.timing.num_timed_embeddings,
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

    timing = None
    if "timing" in data:
        t = data["timing"]
        timing = TimingResult(
            setup_time_s=t.get("setup_time_s", 0.0),
            total_eval_time_s=t.get("total_eval_time_s", 0.0),
            avg_embedding_time_ms=t.get("avg_embedding_time_ms", 0.0),
            median_embedding_time_ms=t.get("median_embedding_time_ms", 0.0),
            std_embedding_time_ms=t.get("std_embedding_time_ms", 0.0),
            avg_pair_time_ms=t.get("avg_pair_time_ms", 0.0),
            num_timed_embeddings=t.get("num_timed_embeddings", 0),
        )

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
        timing=timing,
    )


def _save_combined_results(results: list[ModelResult]) -> None:
    """Save combined benchmark_results.json for the notebook."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for r in results:
        entry = {
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
        if r.timing is not None:
            entry["timing"] = {
                "setup_time_s": r.timing.setup_time_s,
                "total_eval_time_s": r.timing.total_eval_time_s,
                "avg_embedding_time_ms": r.timing.avg_embedding_time_ms,
                "median_embedding_time_ms": r.timing.median_embedding_time_ms,
                "std_embedding_time_ms": r.timing.std_embedding_time_ms,
                "avg_pair_time_ms": r.timing.avg_pair_time_ms,
                "num_timed_embeddings": r.timing.num_timed_embeddings,
            }
        data[r.model_name] = entry
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {out_path}")


def _print_summary(results: list[ModelResult]) -> None:
    header = (
        f"{'Model':<25} {'AUC':>8} {'EER':>8} {'TAR@FAR=1%':>12} {'TAR@FAR=0.1%':>14}"
        f" {'Pairs':>7} {'Skip':>6} {'Fail%':>7}"
        f" {'Avg(ms)':>9} {'Med(ms)':>9} {'Total(s)':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        t = r.timing
        avg_ms = f"{t.avg_embedding_time_ms:>9.1f}" if t else f"{'—':>9}"
        med_ms = f"{t.median_embedding_time_ms:>9.1f}" if t else f"{'—':>9}"
        total_s = f"{t.total_eval_time_s:>9.1f}" if t else f"{'—':>9}"
        print(
            f"{r.model_name:<25} "
            f"{r.auc:>8.4f} "
            f"{r.eer:>8.4f} "
            f"{r.tar_at_far_001:>12.4f} "
            f"{r.tar_at_far_0001:>14.4f} "
            f"{r.num_pairs:>7d} "
            f"{r.num_skipped:>6d} "
            f"{r.detection_failure_rate:>6.2%}"
            f"{avg_ms}{med_ms}{total_s}"
        )
    print("=" * len(header))


def main() -> None:
    args = _parse_args()
    use_cache = not args.force and args.max_pairs is None

    models = _build_models()
    results: list[ModelResult] = []
    need_eval = False

    # Check which models already have saved results
    for model in models:
        if use_cache:
            cached = _load_model_result(model.name)
            if cached is not None:
                print(f"[cached] {model.name} — loaded from {_result_path(model.name)}")
                results.append(cached)
                continue
        need_eval = True
        results.append(model)  # placeholder — will be replaced

    if not need_eval:
        print("\nAll models already evaluated.\n")
    else:
        print("\nLoading LFW pairs dataset ...")
        pairs = load_lfw_pairs()
        if args.max_pairs is not None and args.max_pairs < len(pairs):
            pairs = pairs[: args.max_pairs]
            print(f"  Limited to {len(pairs)} pairs (--max-pairs)")
        print(f"  Loaded {len(pairs)} pairs\n")

        for i, entry in enumerate(results):
            if isinstance(entry, ModelResult):
                continue  # already cached
            model = entry
            print(f"Setting up {model.name} ...")
            try:
                setup_start = time.perf_counter()
                model.setup()
                setup_time = time.perf_counter() - setup_start
            except RuntimeError as e:
                print(f"  Skipping {model.name}: {e}\n")
                results[i] = None
                continue
            print(f"Evaluating {model.name} ...")
            result = evaluate_model(model, pairs)
            # Record setup time in the timing result
            if result.timing is not None:
                result.timing.setup_time_s = setup_time
            if use_cache:
                _save_model_result(result)
            results[i] = result
            print(f"  AUC={result.auc:.4f}  EER={result.eer:.4f}  "
                  f"Skipped={result.num_skipped}/{result.num_pairs + result.num_skipped}\n")
            del model
            gc.collect()

    # Filter out skipped models
    results = [r for r in results if isinstance(r, ModelResult)]

    if use_cache:
        print("Saving combined results ...")
        _save_combined_results(results)

    print("Plotting ROC curves ...")
    plot_roc_curves(results)

    print("Plotting speed comparison ...")
    plot_speed_comparison(results)

    _print_summary(results)


if __name__ == "__main__":
    main()
