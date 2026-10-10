#ifndef TILED_MATMUL_CUH
#define TILED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

/*
    ==================================================
        TILED MATRIX MULTIPLICATION KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for a tiled matrix multiplcation kernel.
    The primary purpose of this code is to apply the CUDA kernel
    creation capabilities of the GPU. The choice behind running
    a tiled matrix mutliplication kernel instead of a standard
    naive one was to showcase the use of shared memory and
    __syncthreads().

*/

template <int TILE_WIDTH>
__global__ void tiledMatmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    int tx{static_cast<int>(threadIdx.x)};
    int ty{static_cast<int>(threadIdx.y)};
    int bx{static_cast<int>(blockIdx.x)};
    int by{static_cast<int>(blockIdx.y)};

    int row{TILE_WIDTH * by + ty};
    int col{TILE_WIDTH * bx + tx};

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    float sum{};

    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {

        if((row < M) && ((tile_idx * TILE_WIDTH + tx) < K))
            Ads[ty][tx] = A[row * K + tile_idx * TILE_WIDTH + tx];
        else
            Ads[ty][tx] = 0.0f;

        if((col < N) && ((tile_idx * TILE_WIDTH + ty) < K))
            Bds[ty][tx] = B[(N * (tile_idx * TILE_WIDTH + ty)) + col];
        else
            Bds[ty][tx] = 0.0f;

        __syncthreads();

        for(int i{}; i < TILE_WIDTH; ++i) {
            sum += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();

    }

    if((row < M) && (col < N))
        C[row * N + col] = sum;
}

template <int TILE_WIDTH>
void tiledMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    dim3 dimBlock(
        TILE_WIDTH,
        TILE_WIDTH,
        1
    );

    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x,
        (M + dimBlock.y - 1) / dimBlock.y,
        1
    ); 

    tiledMatmulKernel<TILE_WIDTH><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Tiled Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';
}

#endif