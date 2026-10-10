#include <iostream>
#include <cuda_runtime.h>

#include "kernel/tiled_matmul.cuh"

#define SIZE 10 //Square for simplicity and nice printing

int main() {

    constexpr int iterations{100000};

    constexpr size_t M{SIZE};
    constexpr size_t K{SIZE};
    constexpr size_t N{SIZE};

    float* A_h;
    float* B_h;
    float* C_h;

    float* A_d;
    float* B_d;
    float* C_d;

    constexpr size_t A_size = M * K * sizeof(float);
    constexpr size_t B_size = K * N * sizeof(float);
    constexpr size_t C_size = M * N * sizeof(float);

    for(int current_iter{1}; current_iter <= iterations; ++current_iter) {

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


        //Temporary Printing Code
        if(current_iter % 1000 == 0) {
            std::cout << current_iter << "/" << iterations << '\n';
            for(size_t row{}; row < SIZE; ++row) {
                for(size_t a_col{}; a_col < SIZE; ++a_col) {
                    if(a_col == 0)
                        std::cout << "[";

                    std::cout << " " << A_h[row * SIZE + a_col];

                    if(a_col == (SIZE - 1))
                        std::cout << "]\n";
                    else
                        std::cout << ",";
                }
                for(int i{}; i < 3; ++i) {
                    if(row == (SIZE / 2))
                        std::cout << " * ";
                    else
                        std::cout << "   ";
                }
                for(size_t b_col{}; b_col < SIZE; ++b_col) {
                    if(a_col == 0)
                        std::cout << "[";

                    std::cout << " " << B_h[row * SIZE + b_col];

                    if(a_col == (SIZE - 1))
                        std::cout << "]\n";
                    else
                        std::cout << ",";
                }
                for(int i{}; i < 3; ++i) {
                    if(row == (SIZE / 2))
                        std::cout << " = ";
                    else
                        std::cout << "   ";
                }
                for(size_t c_col{}; c_col < SIZE: ++c_col) {
                    if(a_col == 0)
                        std::cout << "[";

                    std::cout << " " << C_h[row * SIZE + c_col];

                    if(a_col == (SIZE - 1))
                        std::cout << "]\n";
                    else
                        std::cout << ",";
                }

                std::cout << "\n\n\n";
            }
        }
    }

    return 0;
}
