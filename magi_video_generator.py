#!/usr/bin/env python3
"""
MAGI-1 Video Generator – thin wrapper around the MAGI examples shipped in
`sandai/magi:latest`.

The helper spawns the same `example/<SIZE>/run.sh` scripts that the README
describes :contentReference[oaicite:0]{index=0}, but lets you control them from regular Python code
or a REST service (see `magi_video_service.py`).

Usage
-----
from magi_video_generator import generate_magi_video
result = generate_magi_video(prompt="A red fox in the snow")
print(result["output_path"])
"""

import os, subprocess, uuid
from pathlib import Path
from typing import Optional, Dict, Any


def _resolve_script(model_size: str) -> str:
    size = model_size.strip().upper()
    if size == "24B":
        return "example/24B/run.sh"
    if size in {"4.5B", "4_5B", "4.5B_BASE"}:
        return "example/4.5B/run.sh"
    raise ValueError(f"Unknown MAGI-1 model size: {model_size}")


def generate_magi_video(
    *,
    prompt: str,
    mode: str = "t2v",
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    model_size: str = "4.5B",
    gpus: int = 1,
    config_file: Optional[str] = None,
    save_path: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parameters mirror the CLI flags documented in the MAGI README :contentReference[oaicite:1]{index=1}.
    All args are validated and forwarded to the official run script.
    """
    prompt = prompt.strip()
    if not prompt:
        return {"success": False, "error": "Prompt must not be empty"}

    script = _resolve_script(model_size)
    if not Path(script).exists():
        return {"success": False, "error": f"Script not found: {script}"}

    out = Path(save_path or f"/tmp/magi_{uuid.uuid4().hex}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["bash", script,
           "--mode", mode,
           "--prompt", prompt,
           "--output_path", str(out),
           "--gpus", str(gpus)]
    if config_file:
        cmd += ["--config_file", config_file]
    if image_path:
        cmd += ["--image_path", image_path]
    if prefix_video_path:
        cmd += ["--prefix_video_path", prefix_video_path]
    for k, v in (extra_args or {}).items():
        cmd += [f"--{k}", str(v)]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    ok = result.returncode == 0 and out.exists()

    return {
        "success": ok,
        "output_path": str(out) if ok else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(cmd),
        "returncode": result.returncode,
    }
