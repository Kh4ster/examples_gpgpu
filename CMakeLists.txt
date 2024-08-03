cmake_minimum_required(VERSION 3.30)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES "75")

# Find the RMM library
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)

# Add the executable
add_executable(test_nvtx_cub_rmm test_nvtx_cub_rmm.cu)
target_compile_options(test_nvtx_cub_rmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm PUBLIC OpenMP::OpenMP_CXX rmm::rmm dl)