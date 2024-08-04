#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <raft/core/device_span.hpp>

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

    // Shared memory to hold the tile
    __shared__ int tile[TILE_WIDTH * TILE_WIDTH];

    if (x < width && y < height) {
        const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
        const int index = x + y * width;


        // Load the value into shared memory
        tile[threadId] = d_matrix[index];

        // Synchronize to make sure all threads have loaded their data
        __syncthreads();

        // Sort the tile array using a single thread
        if (threadId == 0) {
            // Simple bubble sort, replace with a more efficient sort if needed
            for (int i = 0; i < TILE_WIDTH * TILE_WIDTH; ++i) {
                for (int j = i + 1; j < TILE_WIDTH * TILE_WIDTH; ++j) {
                    if (tile[i] > tile[j]) {
                        int temp = tile[i];
                        tile[i] = tile[j];
                        tile[j] = temp;
                    }
                }
            }

            // Store the median in the tile memory
            const int medianIndex = (TILE_WIDTH * TILE_WIDTH) / 2;
            d_median[blockIdx.x + blockIdx.y * gridDim.x] = tile[medianIndex];
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

#pragma omp parallel for
    for (int i = 0; i < NB_IMAGES; ++i)
    {
        int *d_matrix, *d_median;

        // Allocate GPU memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_matrix, MATRIX_SIZE * sizeof(int)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_median, (NB_TILE_X * NB_TILE_Y) * sizeof(int)));

        // Copy memory to GPU
        CUDA_CHECK_ERROR(cudaMemcpy(d_matrix, h_matrices[i].data(), MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize((MATRIX_LEGNTH + blockSize.x - 1) / blockSize.x, (MATRIX_LEGNTH + blockSize.y - 1) / blockSize.y);
        computeMedian<TILE_WIDTH, HISTO_SIZE><<<gridSize, blockSize>>>(raft::device_span<int>{d_matrix, MATRIX_SIZE}, raft::device_span<int>{d_median, NB_TILE_X * NB_TILE_Y}, MATRIX_LEGNTH, MATRIX_LEGNTH);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Copy results back to host
        CUDA_CHECK_ERROR(cudaMemcpy(h_medians[i].data(), d_median, (NB_TILE_X * NB_TILE_Y) * sizeof(int), cudaMemcpyDeviceToHost));

        // Free GPU memory
        CUDA_CHECK_ERROR(cudaFree(d_matrix));
        CUDA_CHECK_ERROR(cudaFree(d_median));
    }

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