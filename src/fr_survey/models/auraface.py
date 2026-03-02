"""AuraFace face recognition model (Apache 2.0, 512d)."""

from __future__ import annotations

import numpy as np

from .base import FaceRecognitionModel


class AuraFace(FaceRecognitionModel):
    """AuraFace-v1 recognition on pre-aligned 112x112 crops."""

    def __init__(self) -> None:
        self._rec_model = None

    @property
    def name(self) -> str:
        return "AuraFace"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def aligned_input_size(self) -> int:
        return 112

    def setup(self) -> None:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from insightface.model_zoo import get_model

        models_dir = Path(__file__).resolve().parents[3] / "models"
        auraface_dir = models_dir / "auraface"

        if not auraface_dir.exists() or not list(auraface_dir.glob("*.onnx")):
            print("  Downloading AuraFace-v1 from HuggingFace ...")
            snapshot_download(
                repo_id="fal/AuraFace-v1",
                local_dir=str(auraface_dir),
            )

        # Load the recognition ONNX model (glintr100.onnx)
        rec_path = auraface_dir / "glintr100.onnx"
        if not rec_path.exists():
            raise RuntimeError(f"Recognition model not found: {rec_path}")

        self._rec_model = get_model(
            str(rec_path),
            providers=["CPUExecutionProvider"],
        )
        self._rec_model.prepare(ctx_id=-1)

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray | None:
        assert self._rec_model is not None
        # insightface recognition model expects 112x112 BGR
        feat = self._rec_model.get_feat(aligned_face_bgr)
        emb = feat.flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm
