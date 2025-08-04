# MAGI-1 Video Generation Service

## Setup Instructions for HPC

### 1. Environment Setup
```bash
# Create a virtual environment with Python 3.8+
python3 -m venv magi_env
source magi_env/bin/activate

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Running the Service
```bash
# Activate environment
source magi_env/bin/activate

# Start the FastAPI service
uvicorn magi_video_service:app --host 0.0.0.0 --port 8002

# Or run in background
nohup uvicorn magi_video_service:app --host 0.0.0.0 --port 8002 > magi_service.log 2>&1 &
```

### 3. Using the Client
```bash
# Test with text-to-video (OpenAI-compatible API)
python3 magi_client.py --prompt "Sunset over ocean"

# Test with direct API
python3 magi_client.py --prompt "Sunset over ocean" --method direct

# Test with image-to-video
python3 magi_client.py --prompt "A video of this image" --image path/to/image.jpg --method direct
```

### 4. Health Check
```bash
# Check if service is running and dependencies are available
curl http://localhost:8002/health

# Simple ping
curl http://localhost:8002/ping
```

## Environment Variables

You can configure the service using environment variables:

```bash
export OUT_DIR="/path/to/output/directory"     # Default: /tmp/magi_outputs
export MAGI_MODEL_SIZE="4.5B"                 # Options: 4.5B, 24B
export MAGI_GPUS="1"                          # Number of GPUs to use
export MAGI_CONFIG_FILE="/path/to/config.json" # Optional custom config
```

## Troubleshooting

### Common Issues

1. **PyTorch not found**: Make sure PyTorch with CUDA support is installed
2. **CUDA not available**: Ensure CUDA drivers and toolkit are properly installed
3. **Out of memory**: Reduce batch size or use model offloading (already enabled)
4. **Permission errors**: Make sure output directory is writable

### Error Messages

- `PyTorch not installed`: Install PyTorch with CUDA support
- `CUDA not available`: Check CUDA installation and GPU accessibility
- `Config file not found`: Verify model files are present in the example/ directory
- `Entry script not found`: Ensure the MAGI inference code is properly installed

### Logs

Check the service logs for detailed error information:
```bash
# If running in foreground, errors will be displayed
# If running in background, check the log file
tail -f magi_service.log
```
