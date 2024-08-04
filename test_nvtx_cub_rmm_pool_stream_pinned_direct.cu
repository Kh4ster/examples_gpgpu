#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <raft/core/nvtx.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <cub/cub.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>


#define CUDA_CHECK_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }
inline auto make_pool()
{
  size_t free_mem, total_mem;
  CUDA_CHECK_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
  size_t rmm_alloc_gran = 256;
  double alloc_ratio    = 0.4;
  // allocate 40%
  size_t initial_pool_size = (size_t(free_mem * alloc_ratio) / rmm_alloc_gran) * rmm_alloc_gran;
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_async(),
                                                                     initial_pool_size);
}

template <int TILE_WIDTH, int HISTO_SIZE>
__global__ void computeMedian(raft::device_span<const int> d_matrix, raft::device_span<int> d_median, int width, int height) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        const int index = x + y * width;

        // Declare storage for CUB BlockRadixSort
        typedef cub::BlockRadixSort<int,
                                    TILE_WIDTH,
                                    1,
                                    cub::NullType,
                                    4,
                                    true,
                                    cub::BLOCK_SCAN_WARP_SCANS,
                                    cudaSharedMemBankSizeFourByte,
                                    TILE_WIDTH> BlockRadixSort;

        __shared__ typename BlockRadixSort::TempStorage temp_storage;

        int thread_keys[1];
        thread_keys[0] = d_matrix[index];

        // Perform block-level radix sort
        BlockRadixSort(temp_storage).Sort(thread_keys);

        if (threadIdx.x == TILE_WIDTH / 2 && threadIdx.x == TILE_WIDTH / 2) {
            d_median[blockIdx.x + blockIdx.y * gridDim.x] = thread_keys[0];
        }
    }
}

int main() {
    constexpr auto TILE_WIDTH = 32;
    constexpr auto HISTO_SIZE = 256;
    constexpr auto NB_TILE_X = 250;
    constexpr auto NB_TILE_Y = NB_TILE_X;
    constexpr auto MATRIX_LEGNTH = TILE_WIDTH * NB_TILE_X;
    constexpr auto MATRIX_SIZE = MATRIX_LEGNTH * MATRIX_LEGNTH;
    constexpr auto NB_IMAGES = 3;
    constexpr auto INIT_VALUE = 4;

    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    using host_pinned_vector = thrust::host_vector<int,
                      thrust::mr::stateless_resource_allocator<
                        int,
                        thrust::system::cuda::universal_host_pinned_memory_resource>>;

    std::vector<host_pinned_vector> h_matrices(NB_IMAGES, host_pinned_vector(MATRIX_SIZE, 4));
    std::vector<host_pinned_vector> h_medians(NB_IMAGES, host_pinned_vector(NB_TILE_X * NB_TILE_Y));

    std::vector<std::thread> threads;

    raft::common::nvtx::push_range("Images compute");

#pragma omp parallel for
    for (int i = 0; i < NB_IMAGES; ++i)
    {
        raft::common::nvtx::range fun_scope("Image compute");

        const raft::handle_t handle{};

        int thread_id = omp_get_thread_num();

        const host_pinned_vector& d_matrix = h_matrices[thread_id];
        host_pinned_vector& d_median = h_medians[thread_id];

        raft::common::nvtx::push_range("Kernel");

        // Launch kernel
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize((MATRIX_LEGNTH + blockSize.x - 1) / blockSize.x, (MATRIX_LEGNTH + blockSize.y - 1) / blockSize.y);
        computeMedian<TILE_WIDTH, HISTO_SIZE><<<gridSize, blockSize, 0, handle.get_stream()>>>(raft::device_span<const int>{thrust::raw_pointer_cast(d_matrix.data()), d_matrix.size()}, raft::device_span<int>{thrust::raw_pointer_cast(d_median.data()), d_median.size()}, MATRIX_LEGNTH, MATRIX_LEGNTH);
        CUDA_CHECK_ERROR(cudaGetLastError());

        raft::common::nvtx::pop_range();


        raft::common::nvtx::pop_range();

        CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    raft::common::nvtx::pop_range();

    for (int image = 0; image < NB_IMAGES; ++image)
    {
        if (!std::all_of(h_medians[image].cbegin(), h_medians[image].cend(), [INIT_VALUE](int i){ return i == INIT_VALUE; }))
        {
            std::cout << "Value should be " << INIT_VALUE << std::endl;
            for (int i = 0; i <= 6; ++i)
                std::cout << h_medians[image][i] << " ";
            std::cout << std::endl;
            return -1;
        }
    }

    std::cout << "All good" << std::endl;

    return 0;
}