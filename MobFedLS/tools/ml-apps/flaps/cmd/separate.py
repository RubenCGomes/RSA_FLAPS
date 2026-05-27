from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from clientML import APP

router = APIRouter(tags=["separation"])

# ---------------------------------------------------------------------------
# Playback state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_buffered_stem_path: str | None = None   # path written by the last /buffer call
_buffered_stem_name: str | None = None   # e.g. "vocals"
_playback_proc: subprocess.Popen | None = None


def _is_playing() -> bool:
    # poll() returns None only while the process is still running; any other
    # return value (including non-zero exit codes) means it has stopped.
    if _playback_proc is None:
        return False
    return _playback_proc.poll() is None


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Separation helpers
# ---------------------------------------------------------------------------
def _run_predict(input_path: str) -> dict[str, str]:
    """Run APP.predict and return {stem_name: file_path}."""
    outputs = APP.predict(None, [input_path])
    if not outputs:
        raise RuntimeError("Prediction returned no outputs")
    return outputs[0]["stems"]


def _stem_response(stems: dict[str, str], target: str | None) -> StreamingResponse:
    """Build a WAV or ZIP streaming response from predict output."""
    if target and target in stems:
        wav_bytes = Path(stems[target]).read_bytes()
        return StreamingResponse(
            BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{target}.wav"'},
        )
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem_name, stem_path in stems.items():
            zf.write(stem_path, f"{stem_name}.wav")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="stems.zip"'},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/external/separate/info")
async def separate_info() -> dict:
    """Return this client's stem specialisation and model configuration."""
    return {
        "target_stem": APP.target_stem,
        "model_name": APP.model_name,
        "sample_rate": APP.sr,
        "available_stems": ["drums", "bass", "vocals", "other"],
        "busy": APP.lock.locked(),
    }


@router.post("/external/separate")
async def separate(file: UploadFile, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Accept a full audio mix; return the client's assigned stem as WAV (or all stems as ZIP).

    Returns 503 while an FL round is in progress.
    """
    if APP.lock.locked():
        raise HTTPException(status_code=503, detail="ML app is busy — FL round in progress")

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=f"_{uuid.uuid4().hex}{suffix}", delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        stems = _run_predict(input_path)
    finally:
        os.unlink(input_path)

    background_tasks.add_task(_cleanup, list(stems.values()))
    return _stem_response(stems, APP.target_stem)


@router.post("/external/separate/buffer")
async def separate_buffer(file: UploadFile) -> dict:
    """Separate a song and buffer the result for synchronized playback.

    Unlike POST /external/separate, this endpoint does not stream the audio
    back — it stores the separated stem locally so that a subsequent call to
    POST /external/playback can start all nodes at the same instant.

    Returns 503 while an FL round is in progress.
    """
    global _buffered_stem_path, _buffered_stem_name

    if APP.lock.locked():
        raise HTTPException(status_code=503, detail="ML app is busy — FL round in progress")

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=f"_{uuid.uuid4().hex}{suffix}", delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        stems = _run_predict(input_path)
    finally:
        os.unlink(input_path)

    target = APP.target_stem
    stem_path = stems.get(target) if target else None

    if stem_path is None:
        # No target stem — buffer the first available stem as fallback
        stem_path = next(iter(stems.values()))
        target = next(iter(stems.keys()))

    with _state_lock:
        # Discard any previously buffered file
        if _buffered_stem_path and _buffered_stem_path != stem_path:
            _cleanup([_buffered_stem_path])
        # Clean up the other stems we won't use
        _cleanup([p for s, p in stems.items() if s != target])
        _buffered_stem_path = stem_path
        _buffered_stem_name = target

    return {"ready": True, "stem": target}


@router.post("/external/playback")
async def start_playback() -> dict:
    """Play the stem buffered by POST /external/separate/buffer.

    Playback is non-blocking — the endpoint returns immediately while audio
    plays in the background via ffplay.  Returns 409 if already playing and
    404 if no audio has been buffered yet.
    """
    global _playback_proc

    with _state_lock:
        if _is_playing():
            raise HTTPException(status_code=409, detail="Already playing")
        if not _buffered_stem_path or not Path(_buffered_stem_path).exists():
            raise HTTPException(status_code=404, detail="No buffered audio — call /external/separate/buffer first")

        try:
            _playback_proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", _buffered_stem_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="ffplay not found — install ffmpeg")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to start playback: {exc}")

    return {"playing": True, "stem": _buffered_stem_name}


@router.get("/external/playback/status")
async def playback_status() -> dict:
    """Return current playback and buffer state."""
    with _state_lock:
        return {
            "playing": _is_playing(),
            "stem": _buffered_stem_name,
            "buffered": bool(_buffered_stem_path and Path(_buffered_stem_path).exists()),
        }


@router.delete("/external/playback")
async def stop_playback() -> dict:
    """Stop playback if running."""
    global _playback_proc
    with _state_lock:
        if _playback_proc and _playback_proc.poll() is None:
            _playback_proc.terminate()
            _playback_proc.wait()
    return {"stopped": True}


@router.get("/external/separate/stream")
async def stream_buffered_stem() -> StreamingResponse:
    """Stream the currently buffered stem as a WAV file to the caller.

    Used by the dashboard to play separated audio in the browser.
    Returns 404 if no stem has been buffered yet.
    """
    with _state_lock:
        path = _buffered_stem_path
        name = _buffered_stem_name or "stem"
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No buffered audio — call /external/separate/buffer first")
    file_size = Path(path).stat().st_size

    def _iter_file():
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        _iter_file(),
        media_type="audio/wav",
        headers={
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{name}.wav"',
        },
    )


@router.get("/training/status")
async def training_status() -> dict:
    """Return FL training history accumulated since startup."""
    return APP.get_training_status()


@router.post("/external/model/load")
async def load_model_checkpoint(file: UploadFile) -> dict:
    """Replace the running model weights with an uploaded .pt checkpoint.

    Returns 503 while an FL round is in progress.
    """
    if APP.lock.locked():
        raise HTTPException(status_code=503, detail="ML app is busy — FL round in progress")
    data = await file.read()
    try:
        APP.load_checkpoint_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")
    return {"loaded": True, "model_name": APP.model_name}


@router.get("/external/model/download")
async def download_model_checkpoint() -> StreamingResponse:
    """Stream the current model state dict as a .pt file."""
    buf = io.BytesIO()
    torch.save(APP.model.state_dict(), buf)
    buf.seek(0)
    size = buf.getbuffer().nbytes
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(size),
            "Content-Disposition": f'attachment; filename="{APP.model_name}.pt"',
        },
    )


@router.post("/external/classify")
async def classify_audio(file: UploadFile) -> dict:
    """Classify whether this client's assigned stem is present in the uploaded audio.

    Returns 503 while an FL round is in progress.
    The response is always scoped to this client's TARGET_STEM:
      {"stem": "vocals", "present": true, "confidence": 0.87}
    If no TARGET_STEM is configured, all four stems are returned.
    """
    if APP.lock.locked():
        raise HTTPException(status_code=503, detail="ML app is busy — FL round in progress")

    audio_bytes = await file.read()
    return APP.classify(audio_bytes)
