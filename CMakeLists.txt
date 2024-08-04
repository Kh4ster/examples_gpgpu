cmake_minimum_required(VERSION 3.25)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES "90")

# Find the RMM library
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)

add_executable(test_raw_error test_raw_error.cu)
target_compile_options(test_raw_error PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_raw_error PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_raft test_raft.cu)
target_compile_options(test_raft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS} --expt-relaxed-constexpr>)
target_link_libraries(test_raft PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_nvtx test_nvtx.cu)
target_compile_options(test_nvtx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx PUBLIC OpenMP::OpenMP_CXX raft::raft)

add_executable(test_nvtx_cub test_nvtx_cub.cu)
target_compile_options(test_nvtx_cub PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub PUBLIC OpenMP::OpenMP_CXX raft::raft)

add_executable(test_nvtx_cub_rmm test_nvtx_cub_rmm.cu)
target_compile_options(test_nvtx_cub_rmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

add_executable(test_nvtx_cub_rmm_pool test_nvtx_cub_rmm_pool.cu)
target_compile_options(test_nvtx_cub_rmm_pool PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm_pool PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

add_executable(test_nvtx_cub_rmm_pool_stream test_nvtx_cub_rmm_pool_stream.cu)
target_compile_options(test_nvtx_cub_rmm_pool_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm_pool_stream PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

add_executable(test_nvtx_cub_rmm_pool_stream_pinned test_nvtx_cub_rmm_pool_stream_pinned.cu)
target_compile_options(test_nvtx_cub_rmm_pool_stream_pinned PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm_pool_stream_pinned PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

add_executable(test_nvtx_cub_rmm_pool_stream_pinned_direct test_nvtx_cub_rmm_pool_stream_pinned_direct.cu)
target_compile_options(test_nvtx_cub_rmm_pool_stream_pinned_direct PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(test_nvtx_cub_rmm_pool_stream_pinned_direct PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)