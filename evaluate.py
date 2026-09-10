import sys
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio_utils import load_audio
from app.deepfake_detector import VoiceCloneDetector
from app.config import MODEL_PATH


DATASET_DIR = PROJECT_ROOT / "dataset"

REAL_DIR = DATASET_DIR / "real"
CLONED_DIR = DATASET_DIR / "cloned"


def collect_audio_files(directory):
    extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        ]
    )


def build_evaluation_dataset():
    detector = VoiceCloneDetector()

    X = []
    y = []

    real_files = collect_audio_files(
        REAL_DIR
    )

    cloned_files = collect_audio_files(
        CLONED_DIR
    )

    print(
        f"Real files: {len(real_files)}"
    )

    print(
        f"Cloned files: {len(cloned_files)}"
    )

    for file_path in real_files:

        try:
            waveform, sample_rate = load_audio(
                file_path
            )

            features = detector.extract_features(
                waveform,
                sample_rate,
            )

            X.append(features)
            y.append(0)

        except Exception as exc:
            print(
                f"[WARNING] {file_path}: {exc}"
            )

    for file_path in cloned_files:

        try:
            waveform, sample_rate = load_audio(
                file_path
            )

            features = detector.extract_features(
                waveform,
                sample_rate,
            )

            X.append(features)
            y.append(1)

        except Exception as exc:
            print(
                f"[WARNING] {file_path}: {exc}"
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )


def main():
    print("=" * 60)
    print("VOICE CLONE DETECTION - EVALUATION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    X, y = build_evaluation_dataset()

    predictions = model.predict(X)

    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print(
        f"Accuracy : {accuracy_score(y, predictions):.4f}"
    )

    print(
        f"Precision: {precision_score(y, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y, predictions, zero_division=0):.4f}"
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[:, 1]

        try:
            auc = roc_auc_score(
                y,
                probabilities,
            )

            print(
                f"ROC-AUC  : {auc:.4f}"
            )

        except ValueError:
            pass

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "REAL",
                "CLONED",
            ],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()