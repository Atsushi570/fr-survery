"""DeepFace GhostFaceNet face recognition model (MIT)."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .base import FaceRecognitionModel


class DeepFaceGhostFaceNet(FaceRecognitionModel):
    """GhostFaceNet via the DeepFace library, recognition only."""

    @property
    def name(self) -> str:
        return "DeepFace GhostFaceNet"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def aligned_input_size(self) -> int:
        return 112

    def setup(self) -> None:
        # Suppress noisy TF/DeepFace logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        from deepface import DeepFace

        # Trigger model download by running on a dummy image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="GhostFaceNet",
                detector_backend="skip",
                enforce_detection=False,
            )
        except Exception:
            pass  # May fail on blank image, but model gets cached

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray | None:
        from deepface import DeepFace

        # DeepFace.represent expects RGB
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)

        try:
            results = DeepFace.represent(
                rgb,
                model_name="GhostFaceNet",
                detector_backend="skip",
                enforce_detection=False,
            )
            if results and len(results) > 0:
                return np.array(results[0]["embedding"])
        except Exception:
            pass
        return None
