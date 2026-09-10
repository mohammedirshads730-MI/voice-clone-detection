from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = MODELS_DIR / "detector.pkl"


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

MONO = True

# Maximum audio duration used by the detector.
# Longer recordings are trimmed for predictable inference time.
MAX_AUDIO_DURATION = 15

# Minimum useful audio duration.
MIN_AUDIO_DURATION = 0.5


# ============================================================
# MFCC FEATURE SETTINGS
# ============================================================

N_MFCC = 40

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# CLASS LABELS
# ============================================================

REAL_LABEL = 0
CLONED_LABEL = 1

LABEL_NAMES = {
    0: "REAL",
    1: "CLONED",
}


# ============================================================
# API SETTINGS
# ============================================================

API_TITLE = "Voice Clone Detection API"

API_DESCRIPTION = """
AI-powered voice authenticity analysis API.

Accepts an audio file and returns:
- REAL / CLONED prediction
- confidence
- risk score
- analysis metadata
"""

API_VERSION = "1.0.0"