@echo off
echo Windows llama-cpp-python CUDA Installer
echo =====================================

echo Setting environment variables...
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
set "FORCE_CMAKE=1"
set "CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET=\"cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\" -DCMAKE_CXX_STANDARD=17"

echo Uninstalling previous llama-cpp-python...
pip uninstall -y llama-cpp-python

echo Installing llama-cpp-python with CUDA support...
pip install --no-cache-dir --force-reinstall llama-cpp-python==0.3.8

echo Done!
pause
