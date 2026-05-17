#!/usr/bin/env python3
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

"""
Web service for SAM-Audio separation.
"""

import os
import shutil
import subprocess
import sys
import uuid
import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mago.separator import SAMAudioSeparator
from sam.utils.logs import get_logger

# Define
logger = get_logger(__name__, level='INFO')

SERVICE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVICE_DIR / "static"
UPLOAD_DIR = SERVICE_DIR / "uploads"
OUT_DIR = Path(os.getenv("SAM_AUDIO_OUT_DIR", "exp/sam-audio"))
MODEL_PATH = os.getenv(
    "SAM_AUDIO_MODEL",
    "/data/models/mstudio/sam-audio/facebook/sam-audio-small",
)
DEVICE = os.getenv("SAM_AUDIO_DEVICE", "cuda:0")
PORT = int(os.getenv("SAM_AUDIO_PORT", "8304"))
INPUT_SAMPLE_RATE = int(os.getenv("SAM_AUDIO_INPUT_SAMPLE_RATE", "16000"))
INPUT_SAMPLE_WIDTH = int(os.getenv("SAM_AUDIO_INPUT_SAMPLE_WIDTH", "16"))
INPUT_CHANNELS = int(os.getenv("SAM_AUDIO_INPUT_CHANNELS", "1"))

app = FastAPI(
    title="MAGO SAM-Audio",
    description="Upload audio and separate target sound using SAM-Audio.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

separator: SAMAudioSeparator | None = None


def convert_to_wav(
    input_path: Path,
    content_id: str,
) -> Path:
    """
    Convert uploaded audio to 16 kHz 16-bit mono WAV for SAM-Audio processing.

    Args:
        input_path (Path): Uploaded audio file path.
        content_id (str): Request id.

    Returns:
        Path: Converted WAV file path.
    """
    wav_path = UPLOAD_DIR / f"{content_id}.wav"
    if input_path == wav_path:
        wav_path = UPLOAD_DIR / f"{content_id}.converted.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(INPUT_CHANNELS),
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path


@app.on_event("startup")
def startup() -> None:
    """
    Load SAM-Audio separator once when web service starts.
    """
    global separator

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SAM-Audio separator: {MODEL_PATH}")
    separator = SAMAudioSeparator(
        model_path=MODEL_PATH,
        device=DEVICE,
        out_dir=str(OUT_DIR),
    )
    logger.info("SAM-Audio web service is ready.")


@app.get("/")
def index() -> FileResponse:
    """
    Serve upload web page.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """
    Check web service status.
    """
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": DEVICE,
        "port": PORT,
    }


@app.post("/api/separate")
async def separate_audio(
    audio: UploadFile = File(...),
    description: str = Form(...),
    already_wav: bool = Form(False),
    stream_mode: bool = Form(False),
    chunk_index: int | None = Form(None),
    context_seconds: float | None = Form(None),
) -> dict:
    """
    Upload audio file and separate target sound.

    Args:
        audio (UploadFile): Uploaded audio file.
        description (str): Target sound prompt.
        already_wav (bool): If True, the upload is already 16 kHz 16-bit mono WAV.
        stream_mode (bool): If True, treat upload as a streaming window.
        chunk_index (int | None): Streaming window index.
        context_seconds (float | None): Seconds included in this streaming window.

    Returns:
        dict: Separation result and download URLs.
    """
    if separator is None:
        raise HTTPException(status_code=503, detail="SAM-Audio separator is not ready.")
    if not description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")

    content_id = str(uuid.uuid4())
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    upload_path = UPLOAD_DIR / f"{content_id}{suffix}"

    try:
        with upload_path.open("wb") as fout:
            shutil.copyfileobj(audio.file, fout)

        wav_path = upload_path if already_wav else convert_to_wav(upload_path, content_id)
        result = separator(
            audio=str(wav_path),
            description=description,
            content_id=content_id,
        )
    except Exception as e:
        logger.exception("Failed to separate uploaded audio")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        audio.file.close()

    return {
        "content_id": result["content_id"],
        "description": result["description"],
        "sample_rate": result["sample_rate"],
        "detection": result["detection"],
        "stream": {
            "enabled": stream_mode,
            "chunk_index": chunk_index,
            "context_seconds": context_seconds,
        },
        "original_url": f"api/uploads/{result['content_id']}",
        "target_url": f"api/files/{result['content_id']}/target.wav",
        "residual_url": f"api/files/{result['content_id']}/residual.wav",
    }


@app.get("/api/uploads/{content_id}")
def get_uploaded_file(
    content_id: str,
) -> FileResponse:
    """
    Download uploaded original audio file.
    """
    matches = list(UPLOAD_DIR.glob(f"{content_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = matches[0]
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@app.get("/api/files/{content_id}/{filename}")
def get_file(
    content_id: str,
    filename: str,
) -> FileResponse:
    """
    Download separated audio file.
    """
    if filename not in {"target.wav", "residual.wav"}:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = OUT_DIR / content_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service.app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
