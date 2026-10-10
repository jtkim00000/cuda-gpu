#include <iostream>
#include <cuda_runtime.h>

#include "kernel/tiled_matmul.cuh"

int main() {

    constexpr size_t M{10};
    constexpr size_t K{10};
    constexpr size_t N{10};

    float* A_h;
    float* B_h;
    float* C_h;

    float* A_d;
    float* B_d;
    float* C_d;

    constexpr size_t A_size = M * K * sizeof(float);
    constexpr size_t B_size = K * N * sizeof(float);
    constexpr size_t C_size = M * N * sizeof(float);

    cudaMalloc((void **), &A_d, A_size);
    cudaMalloc((void **), &B_d, B_size);
    cudaMalloc((void **), &C_d, C_size);

    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

    tiledMatmulGPU(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
