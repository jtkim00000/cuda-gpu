#include <iostream>

#include "kernel/tiled_matmul.cuh"

# define M 10
# define K 10
# define N 10

int main() {

    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc();
    cudaMalloc();
    cudaMalloc();

    cudaMemcpy();
    cudaMemcpy();
    cudaMemcpy();

    tiledMatmulGPU();

    cudaMemcpy();
    cudaMemcpy();
    cudaMemcpy();

    cudaDeviceSynchronize();

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
