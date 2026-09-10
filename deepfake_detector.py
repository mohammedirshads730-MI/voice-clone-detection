from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np

from app.audio_utils import load_audio
from app.config import (
    HOP_LENGTH,
    LABEL_NAMES,
    MODEL_PATH,
    N_FFT,
    N_MFCC,
)


class VoiceCloneDetector:
    """
    Voice clone detection inference engine.

    The trained model is expected to be a scikit-learn
    classifier saved as a joblib/pickle artifact.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None

    def load_model(self) -> None:
        """Load the trained classifier from disk."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}. "
                "Run scripts/train.py first."
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load trained model: {exc}"
            ) from exc

    @staticmethod
    def extract_features(
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract acoustic features from an audio waveform.

        Feature groups:
        1. MFCC mean
        2. MFCC standard deviation
        3. Delta MFCC mean
        4. Delta MFCC standard deviation
        5. Delta-delta MFCC mean
        6. Delta-delta MFCC standard deviation
        """

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        delta = librosa.feature.delta(mfcc)

        delta_delta = librosa.feature.delta(
            mfcc,
            order=2,
        )

        features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.std(delta, axis=1),
                np.mean(delta_delta, axis=1),
                np.std(delta_delta, axis=1),
            ]
        )

        return features.astype(np.float32)

    def predict(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        Analyze one audio file.

        Returns a structured prediction dictionary.
        """

        if self.model is None:
            self.load_model()

        waveform, sample_rate = load_audio(file_path)

        features = self.extract_features(
            waveform,
            sample_rate,
        )

        feature_vector = features.reshape(1, -1)

        prediction = int(
            self.model.predict(feature_vector)[0]
        )

        # Probability is available for classifiers
        # such as RandomForestClassifier.
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                feature_vector
            )[0]

            confidence = float(
                np.max(probabilities)
            )

            cloned_probability = float(
                probabilities[1]
            )
        else:
            confidence = 1.0
            cloned_probability = (
                1.0 if prediction == 1 else 0.0
            )

        label = LABEL_NAMES.get(
            prediction,
            "UNKNOWN",
        )

        risk_score = self._calculate_risk_score(
            cloned_probability
        )

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "cloned_probability": round(
                cloned_probability,
                4,
            ),
            "risk_score": risk_score,
            "feature_count": int(len(features)),
        }

    @staticmethod
    def _calculate_risk_score(
        cloned_probability: float,
    ) -> int:
        """
        Convert cloned probability into a 0-100 risk score.

        This is a presentation-oriented risk score,
        not a calibrated cybersecurity risk metric.
        """

        score = int(
            round(cloned_probability * 100)
        )

        return max(
            0,
            min(100, score),
        )