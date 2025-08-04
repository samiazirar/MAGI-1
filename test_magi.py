#!/usr/bin/env python3
"""Test script to debug MAGI issues"""

import sys
import os
from pathlib import Path

# Add the MAGI root to Python path
magi_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(magi_root))

print(f"MAGI root: {magi_root}")
print(f"Python path: {sys.path[:3]}")

# Test basic imports
try:
    from magi_video_generator import generate_magi_video
    print("✓ Successfully imported magi_video_generator")
except Exception as e:
    print(f"✗ Failed to import magi_video_generator: {e}")
    sys.exit(1)

try:
    from inference.pipeline import MagiPipeline
    print("✓ Successfully imported MagiPipeline")
except Exception as e:
    print(f"✗ Failed to import MagiPipeline: {e}")
    print("This might be expected if dependencies are not installed")

# Test config file paths
config_file = magi_root / "example/4.5B/4.5B_base_config.json"
print(f"Config file exists: {config_file.exists()}")
print(f"Config file path: {config_file}")

entry_script = magi_root / "inference/pipeline/entry.py"
print(f"Entry script exists: {entry_script.exists()}")
print(f"Entry script path: {entry_script}")

# Test a simple generation call
print("\nTesting video generation...")
try:
    result = generate_magi_video(
        prompt="Test prompt",
        mode="t2v",
        model_size="4.5B",
        gpus=1,
        save_path="/tmp/test_magi_output.mp4"
    )
    print(f"Generation result: {result}")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
