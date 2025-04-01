FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies early
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    sudo \
    && rm -rf /var/lib/apt/lists/*
# Download Miniconda installer early for caching
ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
# Install Miniconda in its own layer
RUN bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH
# Use bash shell
SHELL ["/bin/bash", "-c"]
# Create conda environment in its own layer
RUN conda create -n rapids python=3.10 -y
# Create environment.yml for better caching
COPY environment.yml /tmp/environment.yml
# Install conda packages from environment.yml
RUN conda env update -n rapids -f /tmp/environment.yml && \
    conda clean -afy
# Install PyTorch in its own layer
RUN conda run -n rapids pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install additional packages in their own layer
RUN conda run -n rapids pip install --no-cache-dir xgboost && \
    conda run -n rapids pip install --no-cache-dir --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm && \
    conda run -n rapids pip install git+https://github.com/huggingface/transformers@v4.50.0
# Set working directory to match the mount point
WORKDIR /workspace
RUN useradd -m -s /bin/bash vscode \
    && echo "vscode:vscode" | chpasswd \
    && adduser vscode sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode
# Update ownership path
RUN mkdir -p /workspace && chown -R vscode:vscode /workspace
# Copy project files
COPY pyproject.toml ./
COPY llamasearch/setup_utils.py ./llamasearch/setup_utils.py
COPY llamasearch/__init__.py ./llamasearch/__init__.py
# Install project dependencies
RUN CMAKE_ARGS="-DGGML_OPENMP=OFF" conda run -n rapids pip install -e '.[dev]'

# Properly initialize conda
RUN conda init bash && \
    mkdir -p /home/vscode/.config && \
    chown -R vscode:vscode /home/vscode/.config && \
    touch /home/vscode/.sudo_as_admin_successful && \
    chown vscode:vscode /home/vscode/.sudo_as_admin_successful

# Set up environment variables for conda activation
ENV CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PREFIX=/opt/conda/envs/rapids \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_PROMPT_MODIFIER=(rapids) \
    CONDA_DEFAULT_ENV=rapids \
    PATH=/opt/conda/envs/rapids/bin:/opt/conda/bin:$PATH

# Create .bashrc with conda initialization
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/vscode/.bashrc && \
    echo 'conda activate rapids' >> /home/vscode/.bashrc && \
    chown vscode:vscode /home/vscode/.bashrc

# Switch to vscode user
USER vscode