"""Evaluation pipeline: compute ROC, AUC, EER, TAR@FAR for face recognition models."""

import time

import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from .dataset import FacePair
from .models.base import FaceRecognitionModel, ModelResult, TimingResult


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute Equal Error Rate from ROC curve."""
    fnr = 1 - tpr
    # Find the point where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """Find TAR (TPR) at a given FAR (FPR) threshold."""
    # Find the largest FPR that is <= target_far
    valid = fpr <= target_far
    if not np.any(valid):
        return 0.0
    return float(tpr[valid][-1])


def evaluate_model(
    model: FaceRecognitionModel,
    pairs: list[FacePair],
) -> ModelResult:
    """Run a face recognition model on all pairs and compute metrics."""
    similarities: list[float] = []
    labels: list[int] = []
    num_skipped = 0
    embedding_times: list[float] = []

    # Warmup: run one embedding to trigger JIT / lazy initialization
    if pairs:
        model.get_embedding(pairs[0].img1)

    eval_start = time.perf_counter()

    for pair in tqdm(pairs, desc=f"  {model.name}", unit="pair"):
        t0 = time.perf_counter()
        emb1 = model.get_embedding(pair.img1)
        t1 = time.perf_counter()
        emb2 = model.get_embedding(pair.img2)
        t2 = time.perf_counter()

        if emb1 is None or emb2 is None:
            num_skipped += 1
            continue

        # Only record times for successful embeddings
        embedding_times.append(t1 - t0)
        embedding_times.append(t2 - t1)

        sim = _cosine_similarity(emb1, emb2)
        similarities.append(sim)
        labels.append(int(pair.is_same))

    eval_end = time.perf_counter()

    # Build TimingResult
    timing = TimingResult(
        total_eval_time_s=eval_end - eval_start,
        num_timed_embeddings=len(embedding_times),
    )
    if embedding_times:
        times_ms = np.array(embedding_times) * 1000
        timing.avg_embedding_time_ms = float(np.mean(times_ms))
        timing.median_embedding_time_ms = float(np.median(times_ms))
        timing.std_embedding_time_ms = float(np.std(times_ms))
        num_successful_pairs = len(embedding_times) // 2
        if num_successful_pairs > 0:
            timing.avg_pair_time_ms = float(
                timing.total_eval_time_s * 1000 / num_successful_pairs
            )

    if len(similarities) == 0:
        empty = np.array([])
        return ModelResult(
            model_name=model.name,
            fpr=empty,
            tpr=empty,
            thresholds=empty,
            num_pairs=0,
            num_skipped=num_skipped,
            timing=timing,
        )

    y_true = np.array(labels)
    y_score = np.array(similarities)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    eer = _compute_eer(fpr, tpr)
    tar_001 = _tar_at_far(fpr, tpr, 0.01)
    tar_0001 = _tar_at_far(fpr, tpr, 0.001)

    return ModelResult(
        model_name=model.name,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=auc_val,
        eer=eer,
        tar_at_far_001=tar_001,
        tar_at_far_0001=tar_0001,
        num_pairs=len(similarities),
        num_skipped=num_skipped,
        timing=timing,
    )
