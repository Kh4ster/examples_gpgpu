#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <raft/core/nvtx.hpp>
#include <raft/core/device_span.hpp>
#include <cub/cub.cuh>
#include <rmm/device_uvector.hpp>

#define CUDA_CHECK_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

template <int TILE_WIDTH, int HISTO_SIZE>
__global__ void computeMedian(raft::device_span<int> d_matrix, raft::device_span<int> d_median, int width, int height) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        const int index = x + y * width;

        // Declare storage for CUB BlockRadixSort
        typedef cub::BlockRadixSort<int, TILE_WIDTH, 1> BlockRadixSort;

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

    std::vector<std::vector<int>> h_matrices(NB_IMAGES, std::vector<int>(MATRIX_SIZE, 4));
    std::vector<std::vector<int>> h_medians(NB_IMAGES, std::vector<int>(NB_TILE_X * NB_TILE_Y));

    raft::common::nvtx::push_range("Images compute");

#pragma omp parallel for
    for (int i = 0; i < NB_IMAGES; ++i)
    {
        raft::common::nvtx::range fun_scope("Image compute");

        raft::common::nvtx::push_range("Memory Allocation");

        // Allocate GPU memory
        rmm::device_uvector<int> d_matrix(MATRIX_SIZE, rmm::cuda_stream_default);
        rmm::device_uvector<int> d_median(NB_TILE_X * NB_TILE_Y, rmm::cuda_stream_default);

        raft::common::nvtx::pop_range();

        raft::common::nvtx::push_range("Memory Copy In");

        // Copy memory to GPU
        CUDA_CHECK_ERROR(cudaMemcpy(d_matrix.data(), h_matrices[i].data(), MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice));

        raft::common::nvtx::pop_range();

        raft::common::nvtx::push_range("Kernel");

        // Launch kernel
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize((MATRIX_LEGNTH + blockSize.x - 1) / blockSize.x, (MATRIX_LEGNTH + blockSize.y - 1) / blockSize.y);
        computeMedian<TILE_WIDTH, HISTO_SIZE><<<gridSize, blockSize>>>(raft::device_span<int>{d_matrix.data(), d_matrix.size()}, raft::device_span<int>{d_median.data(), d_median.size()}, MATRIX_LEGNTH, MATRIX_LEGNTH);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        raft::common::nvtx::pop_range();

        raft::common::nvtx::push_range("Memory Copy Out");

        // Copy results back to host
        CUDA_CHECK_ERROR(cudaMemcpy(h_medians[i].data(), d_median.data(), (NB_TILE_X * NB_TILE_Y) * sizeof(int), cudaMemcpyDeviceToHost));

        raft::common::nvtx::pop_range();

        raft::common::nvtx::pop_range();
    }

    raft::common::nvtx::pop_range();

    for (int image = 0; image < NB_IMAGES; ++image)
    {
        if (!std::all_of(h_medians[image].cbegin(), h_medians[image].cend(), [INIT_VALUE](int i){ return i == INIT_VALUE; }))
        {
            std::cout << "Value should be " << INIT_VALUE << std::endl;
            for (auto e : h_medians[image])
                std::cout << e << " ";
            std::cout << std::endl;
            return -1;
        }
    }

    std::cout << "All good" << std::endl;

    return 0;
}