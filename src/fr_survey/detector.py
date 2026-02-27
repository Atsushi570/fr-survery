"""Common YuNet face detector + ArcFace-standard alignment."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ArcFace standard reference landmarks for 112x112 crop
# (same as insightface/utils/face_align.py `arcface_dst`)
ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# YuNet landmark columns: [right_eye, left_eye, nose, right_mouth, left_mouth]
# ArcFace reference:      [left_eye,  right_eye, nose, left_mouth,  right_mouth]
# "right_eye" in YuNet = person's right = image left (small x),
# which matches "left_eye" in ArcFace reference (x=38.3, also image left).
# Both systems already agree spatially — no reordering needed.


def _download_if_missing(url: str, dest: Path) -> Path:
    """Download a file if it does not already exist."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def _align_crop(
    bgr: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Align and crop a face using a similarity transform to ArcFace reference points.

    Args:
        bgr: Source image in BGR format.
        landmarks_5: 5-point landmarks in insightface order,
                      shape (5, 2).
        output_size: Output square size (112 or 150).

    Returns:
        Aligned BGR image of shape (output_size, output_size, 3).
    """
    ref = ARCFACE_REF_112.copy()
    if output_size != 112:
        scale = output_size / 112.0
        ref = ref * scale

    tform = SimilarityTransform.from_estimate(landmarks_5, ref)
    M = tform.params[:2]
    aligned = cv2.warpAffine(bgr, M, (output_size, output_size), borderValue=0)
    return aligned


@dataclass
class DetectionResult:
    """Result of face detection + alignment."""

    bbox: np.ndarray  # (x, y, w, h)
    confidence: float
    landmarks_5: np.ndarray  # shape (5, 2), insightface order
    aligned_112: np.ndarray  # BGR, 112x112
    aligned_150: np.ndarray  # BGR, 150x150


class YuNetDetector:
    """YuNet face detector with ArcFace-standard alignment."""

    def __init__(self) -> None:
        self._detector: cv2.FaceDetectorYN | None = None

    def setup(self) -> None:
        """Download YuNet ONNX model and initialize the detector."""
        yunet_path = _download_if_missing(
            _YUNET_URL, _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        )
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, image_rgb: np.ndarray) -> DetectionResult | None:
        """Detect the best face in an RGB image and return aligned crops.

        Args:
            image_rgb: Input image in RGB uint8 format.

        Returns:
            DetectionResult with aligned 112x112 and 150x150 crops,
            or None if no face is detected.
        """
        assert self._detector is not None, "Call setup() first"

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)

        if faces is None or len(faces) == 0:
            return None

        # Pick the highest-confidence detection
        best = faces[faces[:, -1].argmax()]

        # Parse bbox
        bbox = best[:4].astype(np.float32)  # x, y, w, h

        # Parse confidence
        confidence = float(best[-1])

        # Parse 5-point landmarks from YuNet output columns 4..13
        # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nose, y_nose,
        #                x_rm, y_rm, x_lm, y_lm, confidence]
        raw_lm = best[4:14].reshape(5, 2).astype(np.float32)

        landmarks = raw_lm

        # Generate aligned crops
        aligned_112 = _align_crop(bgr, landmarks, output_size=112)
        aligned_150 = _align_crop(bgr, landmarks, output_size=150)

        return DetectionResult(
            bbox=bbox,
            confidence=confidence,
            landmarks_5=landmarks,
            aligned_112=aligned_112,
            aligned_150=aligned_150,
        )
