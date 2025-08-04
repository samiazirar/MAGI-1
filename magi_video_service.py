"""
FastAPI service that mimics OpenAI’s `/v1/chat/completions` endpoint but
returns a URL to a MAGI-1 video instead of text.

Start with:
    uvicorn magi_video_service:app --host 0.0.0.0 --port 8002

Based on the Cosmos reference server .
"""

import os, io, time, uuid, base64
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from magi_video_generator import generate_magi_video, check_dependencies

# --------------------------------------------------------------------- #
# Additional models for direct API
# --------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    model_size: Optional[str] = None
    gpus: Optional[int] = None

# --------------------------------------------------------------------- #
# Configuration (env vars make it docker-friendly)
# --------------------------------------------------------------------- #
OUT_DIR          = os.getenv("OUT_DIR", "/tmp/magi_outputs")
MAGI_MODEL_SIZE  = os.getenv("MAGI_MODEL_SIZE", "4.5B")
MAGI_GPUS        = int(os.getenv("MAGI_GPUS", "1"))
MAGI_CONFIG_FILE = os.getenv("MAGI_CONFIG_FILE")  # optional
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------- #
# Utility helpers (decode data: URIs, fetch remote images, save temp)
# --------------------------------------------------------------------- #

def _decode_data_uri(uri: str) -> bytes:
    header, _, b64 = uri.partition(",")
    if not header.startswith("data:"): raise ValueError("Bad data URI")
    return base64.b64decode(b64)

def _fetch_image(url: str) -> Image.Image:
    try:
        data = _decode_data_uri(url) if url.startswith("data:") else requests.get(url, timeout=10).content
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Cannot load image: {e}") from e

def _save_temp(img: Image.Image) -> str:
    path = os.path.join(OUT_DIR, f"inp_{uuid.uuid4().hex}.jpg")
    img.save(path, "JPEG", quality=95)
    return path

def _generate(prompt: str, img: Optional[Image.Image]) -> str:
    img_path = _save_temp(img) if img else None
    try:
        out = generate_magi_video(
            prompt=prompt,
            mode="i2v" if img else "t2v",
            image_path=img_path,
            model_size=MAGI_MODEL_SIZE,
            gpus=MAGI_GPUS,
            config_file=MAGI_CONFIG_FILE,
        )
        if not out["success"]:
            error_msg = out.get("error") or out.get("stderr") or "Unknown MAGI error"
            print(f"MAGI generation failed: {error_msg}")
            if out.get("stdout"):
                print(f"STDOUT: {out['stdout']}")
            if out.get("stderr"):
                print(f"STDERR: {out['stderr']}")
            raise RuntimeError(error_msg)
        return out["output_path"]
    finally:
        # Clean up temporary image file
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {img_path}: {e}")

# --------------------------------------------------------------------- #
# OpenAI-style schema (unchanged from Cosmos) 
# --------------------------------------------------------------------- #
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    class Config: extra = "allow"

class ChatMessage(BaseModel):
    role: str
    content: List[MessageContent]
    class Config: extra = "allow"

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="magi-video-001")
    messages: List[ChatMessage]
    class Config: extra = "allow"

class Choice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]

# --------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------- #
app = FastAPI(title="MAGI-1 Video Service", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/ping")
def ping(): return {"status": "ok", "model": MAGI_MODEL_SIZE, "gpus": MAGI_GPUS}

@app.get("/health")
def health(): 
    deps = check_dependencies()
    return {
        "status": "healthy" if deps["ready"] else "unhealthy",
        "dependencies": deps,
        "magi_config": {
            "model_size": MAGI_MODEL_SIZE,
            "gpus": MAGI_GPUS,
            "config_file": MAGI_CONFIG_FILE
        },
        "output_dir": OUT_DIR
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def completions(req: ChatCompletionRequest, http_request: Request):
    # Extract last user message
    try: last = next(m for m in reversed(req.messages) if m.role == "user")
    except StopIteration: raise HTTPException(400, "Need at least one user message")

    prompt_parts, img = [], None
    for part in last.content:
        if part.type == "text" and part.text: prompt_parts.append(part.text)
        if part.type == "image_url" and not img: img = _fetch_image(part.image_url["url"])
    prompt = " ".join(prompt_parts) or "(empty prompt)"

    try:
        video_path = _generate(prompt, img)
        url = str(http_request.url_for("download", file_id=os.path.basename(video_path)))
        choice = Choice(index=0, message={"role": "assistant", "content": url,
                                          "metadata": {"generated_with": "magi-1",
                                                       "model_size": MAGI_MODEL_SIZE,
                                                       "prompt": prompt}})
        return ChatCompletionResponse(id=f"chatcmpl-{uuid.uuid4().hex}",
                                      created=int(time.time()),
                                      model=req.model,
                                      choices=[choice])
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {e}") from e

@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(OUT_DIR, file_id)
    if not os.path.exists(path): raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")

@app.post("/generate")
def generate(request: GenerateRequest):
    img = _fetch_image(request.image_url) if request.image_url else None
    
    # Use provided parameters or fall back to defaults
    actual_model_size = request.model_size or MAGI_MODEL_SIZE
    actual_gpus = request.gpus or MAGI_GPUS
    
    img_path = _save_temp(img) if img else None
    try:
        out = generate_magi_video(
            prompt=request.prompt,
            mode="i2v" if img else "t2v",
            image_path=img_path,
            model_size=actual_model_size,
            gpus=actual_gpus,
            config_file=MAGI_CONFIG_FILE,
        )
        if not out["success"]:
            error_msg = out.get("error") or out.get("stderr") or "Unknown MAGI error"
            raise HTTPException(500, f"Video generation failed: {error_msg}")
            
        path = out["output_path"]
        return {"success": True,
                "video_path": path,
                "download_url": f"/download/{os.path.basename(path)}",
                "prompt": request.prompt,
                "model_size": actual_model_size,
                "gpus": actual_gpus}
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {e}")
    finally:
        # Clean up temporary image file
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass  # Ignore cleanup errors
