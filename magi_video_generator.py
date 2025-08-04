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

import os, subprocess, uuid, sys, threading, time
from pathlib import Path
from typing import Optional, Dict, Any


def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are available."""
    issues = []
    
    # Check if PyTorch is available
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        torch_available = False
        cuda_available = False
        gpu_count = 0
        issues.append("PyTorch not installed")
    
    # Check if MAGI entry script exists
    magi_root = Path(__file__).parent.absolute()
    entry_script = magi_root / "inference/pipeline/entry.py"
    entry_exists = entry_script.exists()
    if not entry_exists:
        issues.append(f"MAGI entry script not found: {entry_script}")
    
    return {
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "entry_script_exists": entry_exists,
        "issues": issues,
        "ready": len(issues) == 0
    }


def _stream_output(process, prefix="", show_progress=True):
    """Stream subprocess output in real-time."""
    stdout_lines = []
    stderr_lines = []
    
    def read_stdout():
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            stdout_lines.append(line)
            if show_progress and line:
                print(f"{prefix}[STDOUT] {line}", flush=True)
    
    def read_stderr():
        while True:
            line = process.stderr.readline()
            if not line:
                break
            line = line.strip()
            stderr_lines.append(line)
            if show_progress and line:
                print(f"{prefix}[STDERR] {line}", flush=True)
    
    # Start threads to read both stdout and stderr
    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process to complete
    process.wait()
    
    # Wait for threads to finish reading
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    
    return "\n".join(stdout_lines), "\n".join(stderr_lines)


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
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Parameters mirror the CLI flags documented in the MAGI README.
    All args are validated and forwarded to the official entry.py script.
    
    Args:
        show_progress: If True, stream the actual process output in real-time.
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
    start_time = time.time()
    
    try:
        os.chdir(magi_root)
        
        # Start process with streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time or capture silently
        try:
            stdout, stderr = _stream_output(
                process, 
                prefix="" if show_progress else "",  # No prefix, just raw output
                show_progress=show_progress
            )
            returncode = process.returncode
        except Exception as e:
            process.kill()
            return {
                "success": False,
                "error": f"Error during execution: {e}",
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {e}",
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }
    finally:
        os.chdir(original_cwd)

    duration = time.time() - start_time
        
    # Check for common dependency errors and provide helpful messages
    if returncode != 0:
        stderr_lower = stderr.lower()
        if "no module named 'torch'" in stderr_lower:
            error_msg = "PyTorch not installed. Please install PyTorch with CUDA support for GPU inference."
        elif "cuda" in stderr_lower and ("not available" in stderr_lower or "not found" in stderr_lower):
            error_msg = "CUDA not available. Please ensure CUDA is properly installed and GPUs are accessible."
        else:
            error_msg = f"Generation failed with return code {returncode}"
            
        return {
            "success": False,
            "error": error_msg,
            "command": " ".join(cmd),
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": duration
        }

    ok = returncode == 0 and out.exists()

    return {
        "success": ok,
        "output_path": str(out) if ok else None,
        "stdout": stdout,
        "stderr": stderr,
        "command": " ".join(cmd),
        "returncode": returncode,
        "duration": duration
    }
