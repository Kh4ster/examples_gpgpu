cmake_minimum_required(VERSION 3.25)
project(example_gpgpua LANGUAGES CXX CUDA)

# Get the latest CCCL version
# Necessary to use CPMAddPackage
include(cmake/CPM.cmake)

# Get the latest CCCL version
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY "nvidia/cccl"
  GIT_TAG "main"
)

CPMAddPackage(NAME rmm
              GITHUB_REPOSITORY rapidsai/rmm
              VERSION 25.12.00a
              SOURCE_SUBDIR cpp
              SYSTEM Off)


CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.3.0-c-cpp
    GIT_SHALLOW TRUE)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
# Set the C++ standard
set(CXX_STANDARD 17)
# Set the target CUDA architecture to the current one (native)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(test_raw_error test_raw_error.cu)
target_compile_options(test_raw_error PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_raw_error PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_span test_span.cu)
target_compile_options(test_span PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span PUBLIC OpenMP::OpenMP_CXX CCCL::CCCL)

add_executable(test_span_nvtx test_span_nvtx.cu)
target_compile_options(test_span_nvtx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx PUBLIC OpenMP::OpenMP_CXX CCCL::CCCL nvtx3-cpp)

add_executable(test_span_nvtx_cub test_span_nvtx_cub.cu)
target_compile_options(test_span_nvtx_cub PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx_cub PUBLIC OpenMP::OpenMP_CXX CCCL::CCCL nvtx3-cpp)

add_executable(test_span_nvtx_cub_rmm test_span_nvtx_cub_rmm.cu)
target_compile_options(test_span_nvtx_cub_rmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx_cub_rmm PUBLIC OpenMP::OpenMP_CXX rmm::rmm CCCL::CCCL nvtx3-cpp)

add_executable(test_span_nvtx_cub_rmm_pool test_span_nvtx_cub_rmm_pool.cu)
target_compile_options(test_span_nvtx_cub_rmm_pool PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx_cub_rmm_pool PUBLIC OpenMP::OpenMP_CXX rmm::rmm CCCL::CCCL nvtx3-cpp)

add_executable(test_span_nvtx_cub_rmm_pool_stream test_span_nvtx_cub_rmm_pool_stream.cu)
target_compile_options(test_span_nvtx_cub_rmm_pool_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx_cub_rmm_pool_stream PUBLIC OpenMP::OpenMP_CXX rmm::rmm CCCL::CCCL nvtx3-cpp)


add_executable(test_span_nvtx_cub_rmm_pool_stream_pinned test_span_nvtx_cub_rmm_pool_stream_pinned.cu)
target_compile_options(test_span_nvtx_cub_rmm_pool_stream_pinned PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_span_nvtx_cub_rmm_pool_stream_pinned PUBLIC OpenMP::OpenMP_CXX rmm::rmm CCCL::CCCL nvtx3-cpp)
