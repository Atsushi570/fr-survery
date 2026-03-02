"""Abstract base class for face recognition models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimingResult:
    """Timing measurements for a single model evaluation."""

    setup_time_s: float = 0.0
    total_eval_time_s: float = 0.0
    avg_embedding_time_ms: float = 0.0
    median_embedding_time_ms: float = 0.0
    std_embedding_time_ms: float = 0.0
    avg_pair_time_ms: float = 0.0
    num_timed_embeddings: int = 0


class FaceRecognitionModel(ABC):
    """Base class for face recognition model wrappers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of face embeddings."""

    @property
    @abstractmethod
    def aligned_input_size(self) -> int:
        """Expected aligned face crop size (112 or 150)."""

    @abstractmethod
    def setup(self) -> None:
        """Download and initialize model. Called once before evaluation."""

    @abstractmethod
    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray | None:
        """Extract face embedding from an aligned BGR face crop.

        Args:
            aligned_face_bgr: Pre-aligned face image in BGR format,
                              sized according to aligned_input_size.

        Returns:
            Embedding vector, or None on failure.
        """


@dataclass
class ModelResult:
    """Evaluation results for a single model."""

    model_name: str
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    thresholds: np.ndarray = field(repr=False)
    auc: float = 0.0
    eer: float = 0.0
    tar_at_far_001: float = 0.0  # TAR @ FAR=0.01
    tar_at_far_0001: float = 0.0  # TAR @ FAR=0.001
    num_pairs: int = 0
    num_skipped: int = 0
    timing: TimingResult | None = None

    @property
    def detection_failure_rate(self) -> float:
        total = self.num_pairs + self.num_skipped
        if total == 0:
            return 0.0
        return self.num_skipped / total
