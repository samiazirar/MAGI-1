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
    Parameters mirror the CLI flags documented in the MAGI README.
    All args are validated and forwarded to the official entry.py script.
    """
    prompt = prompt.strip()
    if not prompt:
        return {"success": False, "error": "Prompt must not be empty"}

    # Find the MAGI root directory
    magi_root = Path(__file__).parent.absolute()
    
    # Set up config file path
    if not config_file:
        # Map model sizes to their directory names and config file names
        if model_size == "24B":
            config_file = str(magi_root / "example/24B/24B_base_config.json")
        elif model_size in {"4.5B", "4_5B", "4.5B_BASE"}:
            config_file = str(magi_root / "example/4.5B/4.5B_base_config.json")
        else:
            return {"success": False, "error": f"Unknown MAGI-1 model size: {model_size}"}
    
    if not Path(config_file).exists():
        return {"success": False, "error": f"Config file not found: {config_file}"}

    # Set up output path
    out = Path(save_path or f"/tmp/magi_{uuid.uuid4().hex}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build command to call entry.py directly
    entry_script = magi_root / "inference/pipeline/entry.py"
    if not entry_script.exists():
        return {"success": False, "error": f"Entry script not found: {entry_script}"}

    cmd = ["python3", str(entry_script),
           "--config_file", config_file,
           "--mode", mode,
           "--prompt", prompt,
           "--output_path", str(out)]
    
    if image_path and mode == "i2v":
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}
        cmd += ["--image_path", image_path]
    
    if prefix_video_path and mode == "v2v":
        if not Path(prefix_video_path).exists():
            return {"success": False, "error": f"Prefix video file not found: {prefix_video_path}"}
        cmd += ["--prefix_video_path", prefix_video_path]

    # Add any extra arguments
    for k, v in (extra_args or {}).items():
        cmd += [f"--{k}", str(v)]

    # Set up environment
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{magi_root}:{env.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "6009",
        "GPUS_PER_NODE": str(gpus),
        "NNODES": "1",
        "WORLD_SIZE": str(gpus),
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus)),
        "PAD_HQ": "1",
        "PAD_DURATION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OFFLOAD_T5_CACHE": "true",
        "OFFLOAD_VAE_CACHE": "true",
        "TORCH_CUDA_ARCH_LIST": "8.9;9.0"
    })

    # Change to MAGI root directory for execution
    original_cwd = os.getcwd()
    try:
        os.chdir(magi_root)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)  # 30 min timeout
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Video generation timed out after 30 minutes",
            "command": " ".join(cmd),
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {e}",
            "command": " ".join(cmd),
            "returncode": -1,
        }
    finally:
        os.chdir(original_cwd)

    ok = result.returncode == 0 and out.exists()

    return {
        "success": ok,
        "output_path": str(out) if ok else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(cmd),
        "returncode": result.returncode,
    }
