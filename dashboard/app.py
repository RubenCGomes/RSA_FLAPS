import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flaps-dashboard")

app = FastAPI(title="FLAPS Dashboard Backend")

# Enable CORS for local testing/development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths — DATA_DIR can be overridden via env var for containerised deployments
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
UPLOADS_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
NODES_FILE = DATA_DIR / "nodes.json"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class NodeRequest(BaseModel):
    url: str

class BufferRequest(BaseModel):
    filename: str
    nodes: List[str]

class PlaybackRequest(BaseModel):
    nodes: List[str]

class ClassifyRequest(BaseModel):
    filename: str
    node_url: str

# ---------------------------------------------------------------------------
# Persistent Storage Helpers
# ---------------------------------------------------------------------------
def load_nodes() -> List[str]:
    if not NODES_FILE.exists():
        default_nodes = ["http://localhost:5001", "http://localhost:5002"]
        with open(NODES_FILE, "w") as f:
            json.dump(default_nodes, f)
        return default_nodes
    try:
        with open(NODES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading nodes: {e}")
        return ["http://localhost:5001", "http://localhost:5002"]

def save_nodes(nodes: List[str]):
    try:
        with open(NODES_FILE, "w") as f:
            json.dump(nodes, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving nodes: {e}")

# Initialize files
load_nodes()

# ---------------------------------------------------------------------------
# Node Communication Helpers
# ---------------------------------------------------------------------------
def check_single_node_status(url: str) -> dict:
    status_data = {
        "url": url,
        "online": False,
        "busy": False,
        "target_stem": None,
        "model_name": None,
        "sample_rate": None,
        "playing": False,
        "buffered": False,
        "buffered_stem": None,
        "error": None
    }
    try:
        # Get Info
        info_resp = requests.get(f"{url}/external/separate/info", timeout=2)
        if info_resp.status_code == 200:
            status_data["online"] = True
            info_json = info_resp.json()
            status_data["target_stem"] = info_json.get("target_stem")
            status_data["model_name"] = info_json.get("model_name")
            status_data["sample_rate"] = info_json.get("sample_rate")
            status_data["busy"] = info_json.get("busy", False)
        elif info_resp.status_code == 503:
            status_data["online"] = True
            status_data["busy"] = True
            status_data["error"] = "FL training in progress"
            return status_data
        else:
            status_data["error"] = f"Info status: {info_resp.status_code}"
            return status_data

        # Get Playback Status (only if info succeeded)
        play_resp = requests.get(f"{url}/external/playback/status", timeout=2)
        if play_resp.status_code == 200:
            play_json = play_resp.json()
            status_data["playing"] = play_json.get("playing", False)
            status_data["buffered"] = play_json.get("buffered", False)
            status_data["buffered_stem"] = play_json.get("stem")
    except requests.exceptions.RequestException as e:
        status_data["error"] = str(e)
    return status_data

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

# --- Nodes Management ---
@app.get("/api/nodes")
def get_nodes():
    return load_nodes()

@app.post("/api/nodes")
def add_node(req: NodeRequest):
    url = req.url.strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL format. Must start with http:// or https://")
    nodes = load_nodes()
    if url in nodes:
        raise HTTPException(status_code=409, detail="Node already registered")
    nodes.append(url)
    save_nodes(nodes)
    return {"message": "Node added successfully", "nodes": nodes}

@app.delete("/api/nodes")
def delete_node(req: NodeRequest):
    url = req.url.strip().rstrip("/")
    nodes = load_nodes()
    if url not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    nodes.remove(url)
    save_nodes(nodes)
    return {"message": "Node removed successfully", "nodes": nodes}

@app.get("/api/nodes/status")
def get_nodes_status():
    nodes = load_nodes()
    if not nodes:
        return []
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        results = list(executor.map(check_single_node_status, nodes))
    return results

# --- Audio Library ---
@app.get("/api/audio")
def list_audio_files():
    files = []
    for f in UPLOADS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            stats = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stats.st_size,
                "size_mb": round(stats.st_size / (1024 * 1024), 2),
                "created_at": stats.st_mtime
            })
    # Sort by created time descending
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return files

@app.post("/api/audio/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    orig_filename = file.filename
    suffix = Path(orig_filename).suffix.lower()
    if suffix not in [".wav", ".mp3", ".flac", ".ogg"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, MP3, FLAC or OGG.")
    
    # Save the uploaded file temporarily
    temp_file_path = UPLOADS_DIR / f"temp_{orig_filename}"
    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Failed to save temporary upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    
    # Determine target filename (always convert to .wav if not already)
    if suffix != ".wav":
        target_filename = Path(orig_filename).stem + ".wav"
    else:
        target_filename = orig_filename
        
    file_path = UPLOADS_DIR / target_filename
    
    try:
        if suffix != ".wav":
            logger.info(f"Converting {orig_filename} to WAV format using ffmpeg...")
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_file_path),
                "-acodec", "pcm_s16le",
                str(file_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {result.stderr}")
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        else:
            # Just rename temp file to final file
            if file_path.exists():
                file_path.unlink()
            temp_file_path.rename(file_path)
    except Exception as e:
        logger.error(f"Failed to process audio file: {e}")
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()
            
    return {"filename": target_filename, "message": "File uploaded and processed successfully"}

@app.delete("/api/audio/{filename}")
def delete_audio_file(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.unlink()
        return {"message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

# --- Playback & Orchestration ---
def buffer_on_node(node_url: str, file_path: Path, filename: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            resp = requests.post(f"{node_url}/external/separate/buffer", files=files, timeout=300)
            if resp.status_code == 200:
                return {"url": node_url, "success": True, "data": resp.json()}
            else:
                return {"url": node_url, "success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"url": node_url, "success": False, "error": str(e)}

@app.post("/api/audio/buffer")
def buffer_audio(req: BufferRequest):
    file_path = UPLOADS_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found in library")
    
    nodes = req.nodes
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes selected")

    results = {}
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(buffer_on_node, url, file_path, req.filename): url for url in nodes}
        for fut in futures:
            node_url = futures[fut]
            try:
                res = fut.result()
                results[node_url] = res
            except Exception as e:
                results[node_url] = {"url": node_url, "success": False, "error": str(e)}
    
    return {"message": "Buffer process completed", "results": results}

def start_playback_on_node(node_url: str) -> dict:
    try:
        resp = requests.post(f"{node_url}/external/playback", timeout=5)
        if resp.status_code == 200:
            return {"url": node_url, "success": True, "data": resp.json()}
        else:
            return {"url": node_url, "success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"url": node_url, "success": False, "error": str(e)}

@app.post("/api/audio/playback/start")
def start_playback(req: PlaybackRequest):
    nodes = req.nodes
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes selected")

    results = {}
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(start_playback_on_node, url): url for url in nodes}
        for fut in futures:
            node_url = futures[fut]
            try:
                res = fut.result()
                results[node_url] = res
            except Exception as e:
                results[node_url] = {"url": node_url, "success": False, "error": str(e)}
    
    return {"message": "Synchronized playback triggered", "results": results}

def stop_playback_on_node(node_url: str) -> dict:
    try:
        resp = requests.delete(f"{node_url}/external/playback", timeout=5)
        if resp.status_code == 200:
            return {"url": node_url, "success": True, "data": resp.json()}
        else:
            return {"url": node_url, "success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"url": node_url, "success": False, "error": str(e)}

@app.post("/api/audio/playback/stop")
def stop_playback(req: PlaybackRequest):
    nodes = req.nodes
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes selected")

    results = {}
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(stop_playback_on_node, url): url for url in nodes}
        for fut in futures:
            node_url = futures[fut]
            try:
                res = fut.result()
                results[node_url] = res
            except Exception as e:
                results[node_url] = {"url": node_url, "success": False, "error": str(e)}
    
    return {"message": "Stop command sent", "results": results}

# --- Audio Classification Proxy ---
@app.post("/api/audio/classify")
def classify_audio(req: ClassifyRequest):
    file_path = UPLOADS_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found in library")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (req.filename, f, "audio/wav")}
            resp = requests.post(f"{req.node_url}/external/classify", files=files, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            else:
                raise HTTPException(status_code=resp.status_code, detail=f"Client classification failed: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification request failed: {str(e)}")



# --- Browser Audio Streaming ---
@app.get("/api/audio/stream")
def stream_node_audio(node_url: str):
    """Proxy the buffered stem WAV from a node to the browser for in-browser playback."""
    try:
        upstream = requests.get(f"{node_url}/external/separate/stream", stream=True, timeout=10)
        if upstream.status_code == 404:
            raise HTTPException(status_code=404, detail="No buffered audio on this node")
        if upstream.status_code != 200:
            raise HTTPException(status_code=upstream.status_code, detail="Node error")

        def _iter_chunks():
            for chunk in upstream.iter_content(chunk_size=65536):
                if chunk:
                    yield chunk

        headers = {}
        if "Content-Length" in upstream.headers:
            headers["Content-Length"] = upstream.headers["Content-Length"]

        return StreamingResponse(_iter_chunks(), media_type="audio/wav", headers=headers)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Node unreachable")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Node timed out")


# --- Training Status ---
def fetch_training_status_from_node(url: str) -> dict:
    try:
        resp = requests.get(f"{url}/training/status", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            data["url"] = url
            data["ok"] = True
            return data
        return {"url": url, "ok": False, "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"url": url, "ok": False, "error": str(e)}

@app.get("/api/training/status")
def get_training_status():
    nodes = load_nodes()
    if not nodes:
        return []
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        results = list(executor.map(fetch_training_status_from_node, nodes))
    return results


# --- Static Frontend Serving ---
@app.get("/")
def read_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Fallback to creating a quick template if it doesn't exist yet
        return {"message": "Frontend files not found under static/ directory yet."}
    return FileResponse(index_path)

# Mount /static for JS, CSS, images, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
