import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

# Allow importing from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio_utils import load_audio
from app.deepfake_detector import VoiceCloneDetector
from app.config import MODEL_PATH


# ============================================================
# DATASET
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

REAL_DIR = DATASET_DIR / "real"
CLONED_DIR = DATASET_DIR / "cloned"


def collect_audio_files(directory: Path):
    """Collect supported audio files recursively."""

    extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    files = []

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files)


def extract_dataset_features():
    """Extract features from all training audio."""

    if not REAL_DIR.exists():
        raise FileNotFoundError(
            f"Real dataset directory not found: {REAL_DIR}"
        )

    if not CLONED_DIR.exists():
        raise FileNotFoundError(
            f"Cloned dataset directory not found: {CLONED_DIR}"
        )

    real_files = collect_audio_files(REAL_DIR)
    cloned_files = collect_audio_files(CLONED_DIR)

    print(f"Real audio files: {len(real_files)}")
    print(f"Cloned audio files: {len(cloned_files)}")

    if not real_files:
        raise RuntimeError("No real audio files found.")

    if not cloned_files:
        raise RuntimeError("No cloned audio files found.")

    detector = VoiceCloneDetector()

    X = []
    y = []

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    for index, file_path in enumerate(real_files, start=1):

        try:
            waveform, sample_rate = load_audio(file_path)

            features = detector.extract_features(
                waveform,
                sample_rate,
            )

            X.append(features)
            y.append(0)

        except Exception as exc:
            print(
                f"[WARNING] Skipping {file_path}: {exc}"
            )

        if index % 100 == 0:
            print(
                f"Processed real: {index}/{len(real_files)}"
            )

    # --------------------------------------------------------
    # CLONED
    # --------------------------------------------------------

    for index, file_path in enumerate(cloned_files, start=1):

        try:
            waveform, sample_rate = load_audio(file_path)

            features = detector.extract_features(
                waveform,
                sample_rate,
            )

            X.append(features)
            y.append(1)

        except Exception as exc:
            print(
                f"[WARNING] Skipping {file_path}: {exc}"
            )

        if index % 100 == 0:
            print(
                f"Processed cloned: "
                f"{index}/{len(cloned_files)}"
            )

    if not X:
        raise RuntimeError(
            "No usable audio samples were processed."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y


def train_model(X, y):
    """Train the Random Forest classifier."""

    print("\nDataset shape:", X.shape)
    print("Labels:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "REAL",
                "CLONED",
            ],
        )
    )

    return model


def main():
    print("=" * 60)
    print("VOICE CLONE DETECTION - TRAINING")
    print("=" * 60)

    X, y = extract_dataset_features()

    model = train_model(
        X,
        y,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()