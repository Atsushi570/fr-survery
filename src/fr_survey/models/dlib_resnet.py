"""Dlib ResNet-34 face recognition model (Public Domain, 128d)."""

from __future__ import annotations

import cv2
import numpy as np

from .base import FaceRecognitionModel

try:
    import face_recognition

    DLIB_AVAILABLE = True
except (ImportError, OSError):
    DLIB_AVAILABLE = False


class DlibResNet(FaceRecognitionModel):
    """Dlib's ResNet-34 via the face_recognition library."""

    @property
    def name(self) -> str:
        return "Dlib ResNet-34"

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def aligned_input_size(self) -> int:
        return 150

    def setup(self) -> None:
        if not DLIB_AVAILABLE:
            raise RuntimeError(
                "dlib/face_recognition is not available on this platform. "
                "This model will be skipped."
            )
        # face_recognition loads models lazily; nothing extra to do here.

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray | None:
        if not DLIB_AVAILABLE:
            return None
        # Convert BGR to RGB for face_recognition
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Treat the entire aligned crop as a single face location
        # face_recognition format: (top, right, bottom, left)
        face_location = [(0, w, h, 0)]
        encodings = face_recognition.face_encodings(rgb, face_location)
        if len(encodings) == 0:
            return None
        return encodings[0]
