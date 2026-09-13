import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.audio_utils import get_audio_quality
from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    FRONTEND_DIR,
)
from app.deepfake_detector import VoiceCloneDetector


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

detector = VoiceCloneDetector()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    model_available = detector.model_path.exists()

    return {
        "status": "healthy",
        "model_available": model_available,
    }


# ============================================================
# AUDIO ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded audio file.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, FLAC, OGG, or M4A."
            ),
        )

    temp_path = None

    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        # Basic upload size protection.
        max_size = 25 * 1024 * 1024

        if len(audio_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail="Audio file is too large. Maximum is 25 MB.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_file.write(audio_bytes)

            temp_path = temp_file.name

        start_time = time.perf_counter()

        result = detector.predict(temp_path)

        processing_time = (
            time.perf_counter() - start_time
        )

        # Audio metadata.
        from app.audio_utils import load_audio

        waveform, sample_rate = load_audio(
            temp_path
        )

        quality = get_audio_quality(
            waveform
        )

        result.update(
            {
                "filename": file.filename,
                "audio": {
                    "sample_rate": sample_rate,
                    "duration_seconds": round(
                        len(waveform) / sample_rate,
                        3,
                    ),
                    "quality": quality,
                },
                "processing_time_ms": round(
                    processing_time * 1000,
                    2,
                ),
            }
        )

        return result

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)