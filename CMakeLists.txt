# /workspace/CMakeLists.txt

cmake_minimum_required(VERSION 3.26)
project(llamasearch LANGUAGES C CXX)

# Enable position-independent code if needed
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# If you have custom logic or want to build any local C++ sources, place it here.
# For example, if you want to include or build llama.cpp, you'd do something like:
# add_subdirectory(path/to/llama.cpp) 
# Or link to external libraries, etc.

message(STATUS "CMake project for LlamaSearch is configured.")
