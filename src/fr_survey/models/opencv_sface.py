"""OpenCV SFace face recognition model (Apache 2.0, 128d)."""

from pathlib import Path

import cv2
import numpy as np

from ..detector import _download_if_missing
from .base import FaceRecognitionModel

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


class OpenCVSFace(FaceRecognitionModel):
    """SFace recognition on pre-aligned 112x112 crops."""

    def __init__(self) -> None:
        self._recognizer: cv2.FaceRecognizerSF | None = None

    @property
    def name(self) -> str:
        return "OpenCV SFace"

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def aligned_input_size(self) -> int:
        return 112

    def setup(self) -> None:
        sface_path = _download_if_missing(
            _SFACE_URL, _MODELS_DIR / "face_recognition_sface_2021dec.onnx"
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(sface_path),
            "",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray | None:
        assert self._recognizer is not None
        feat = self._recognizer.feature(aligned_face_bgr)
        return feat.flatten()
