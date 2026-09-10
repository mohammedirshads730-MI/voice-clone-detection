from pathlib import Path

import librosa
import numpy as np

from app.config import (
    SAMPLE_RATE,
    MAX_AUDIO_DURATION,
    MIN_AUDIO_DURATION,
)


def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file and convert it to:
    - mono
    - 16 kHz
    - floating-point waveform

    Returns:
        waveform: numpy array
        sample_rate: integer
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        waveform, sample_rate = librosa.load(
            str(file_path),
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to read audio file: {exc}"
        ) from exc

    if waveform is None or len(waveform) == 0:
        raise ValueError("Audio file contains no usable samples.")

    duration = len(waveform) / sample_rate

    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio is too short. Minimum duration is "
            f"{MIN_AUDIO_DURATION} seconds."
        )

    # Limit maximum processing duration.
    max_samples = int(MAX_AUDIO_DURATION * sample_rate)

    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    # Remove DC offset.
    waveform = waveform - np.mean(waveform)

    # Normalize safely.
    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    return waveform.astype(np.float32), sample_rate


def validate_audio_file(file_path: str | Path) -> dict:
    """
    Validate and inspect an audio file.

    Returns basic metadata.
    """

    waveform, sample_rate = load_audio(file_path)

    duration = len(waveform) / sample_rate

    return {
        "sample_rate": sample_rate,
        "duration_seconds": round(float(duration), 3),
        "samples": int(len(waveform)),
        "channels": 1,
    }


def get_audio_quality(waveform: np.ndarray) -> dict:
    """
    Estimate basic audio quality indicators.

    This is not a forensic-quality measurement.
    It is intended for prototype-level analysis and UI feedback.
    """

    if len(waveform) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "is_silent": True,
        }

    rms = float(np.sqrt(np.mean(waveform ** 2)))
    peak = float(np.max(np.abs(waveform)))

    is_silent = rms < 0.005

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "is_silent": is_silent,
    }