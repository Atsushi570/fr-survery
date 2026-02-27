"""LFW / RFW dataset loading for face recognition benchmark."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

RFW_RACES = ["Asian", "Caucasian", "Indian", "African"]


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


def load_rfw_pairs(rfw_dir: Path, race: str) -> list[FacePair]:
    """Load RFW pairs for a specific race.

    RFW directory structure:
        rfw_dir/txts/<race>/<race>_pairs.txt
        rfw_dir/data/<race>/<person_name>/<person_name>_NNNN.jpg

    pairs.txt format (tab-separated, LFW standard):
        Line 1: <num_folds>\t<pairs_per_fold>
        Same person:     Name\t1\t4
        Different person: Name1\t1\tName2\t3
    """
    if race not in RFW_RACES:
        raise ValueError(f"Unknown race '{race}'. Must be one of {RFW_RACES}")

    rfw_dir = Path(rfw_dir)
    pairs_file = rfw_dir / "txts" / race / f"{race}_pairs.txt"
    images_dir = rfw_dir / "data" / race

    if not pairs_file.exists():
        raise FileNotFoundError(f"pairs.txt not found: {pairs_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    def _img_path(name: str, idx: int) -> Path:
        return images_dir / name / f"{name}_{idx:04d}.jpg"

    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result: list[FacePair] = []
    with open(pairs_file) as f:
        # Skip header line (e.g. "10\t300")
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                # Same person: Name\tidx1\tidx2
                name, idx1, idx2 = parts
                img1 = _load_image(_img_path(name, int(idx1)))
                img2 = _load_image(_img_path(name, int(idx2)))
                result.append(FacePair(img1=img1, img2=img2, is_same=True))
            elif len(parts) == 4:
                # Different person: Name1\tidx1\tName2\tidx2
                name1, idx1, name2, idx2 = parts
                img1 = _load_image(_img_path(name1, int(idx1)))
                img2 = _load_image(_img_path(name2, int(idx2)))
                result.append(FacePair(img1=img1, img2=img2, is_same=False))

    return result
