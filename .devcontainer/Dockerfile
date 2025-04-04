# Use NVIDIA's official CUDA 11.8 base image with cuDNN8 on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Noninteractive to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1) Upgrade to a newer cmake from Kitware apt repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg apt-transport-https ca-certificates software-properties-common \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main' \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# 2) Install Python 3, pip, other build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git \
    ocl-icd-libopencl1 libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# 3) Pre-install some GPU-accelerated libraries that you want
RUN pip3 install --no-cache-dir lightgbm --config-settings=cmake.define.USE_CUDA=ON \
    && pip3 install --no-cache-dir xgboost \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir cudf-cu11 cuml-cu11 cugraph-cu11 --extra-index-url=https://pypi.nvidia.com

# 4) Let’s set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# We'll also ensure our scikit-build sees sm_80 or sm_86
# (If Ampere, typically 80 or 86 is valid. We'll do both.)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86"

# Default shell command
CMD ["/bin/bash"]