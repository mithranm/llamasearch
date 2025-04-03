#!/bin/bash
echo "Container started successfully"
echo "GPU Information:"
nvidia-smi || echo "No NVIDIA GPU detected"
echo "CUDA Version: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -c2- || echo "Unknown")"
echo "To install llamasearch: cd /workspace && pip install -e ."
exec "$@"
