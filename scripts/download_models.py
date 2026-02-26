"""Download all model files before running the benchmark."""

import sys
from pathlib import Path

# Add project root to path so we can import fr_survey
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fr_survey.models.opencv_sface import OpenCVSFace
from fr_survey.models.auraface import AuraFace
from fr_survey.models.dlib_resnet import DlibResNet, DLIB_AVAILABLE
from fr_survey.models.deepface_ghost import DeepFaceGhostFaceNet


def main() -> None:
    models = [
        OpenCVSFace(),
        AuraFace(),
        DeepFaceGhostFaceNet(),
    ]
    if DLIB_AVAILABLE:
        models.append(DlibResNet())
    else:
        print("⚠ dlib not available — skipping DlibResNet download")

    for model in models:
        print(f"Setting up {model.name} ...")
        try:
            model.setup()
            print(f"  ✓ {model.name} ready")
        except Exception as e:
            print(f"  ✗ {model.name} failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
