"""LFW dataset loading for face recognition benchmark."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


@dataclass
class FacePair:
    """A pair of face images with a label indicating if they are the same person."""

    img1: np.ndarray  # RGB uint8, shape (H, W, 3)
    img2: np.ndarray  # RGB uint8, shape (H, W, 3)
    is_same: bool


def load_lfw_pairs() -> list[FacePair]:
    """Load LFW pairs dataset (10_folds subset, 6000 pairs).

    Returns RGB uint8 numpy arrays.
    """
    dataset = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0)
    pairs = dataset.pairs  # (6000, 2, 125, 94, 3) float32 [0, 1]
    labels = dataset.target  # 1 = same, 0 = different

    result = []
    for i in range(len(labels)):
        img1 = (pairs[i, 0] * 255).astype(np.uint8)
        img2 = (pairs[i, 1] * 255).astype(np.uint8)
        result.append(FacePair(img1=img1, img2=img2, is_same=bool(labels[i])))

    return result
