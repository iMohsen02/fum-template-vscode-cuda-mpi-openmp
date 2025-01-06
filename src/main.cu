#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void helloFromCUDA()
{
    printf("Hello World from CUDA thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char *argv[])
{

    // Launch the CUDA kernel
    helloFromCUDA<<<2, 4>>>(); // 2 blocks, 4 threads per block
    cudaDeviceSynchronize();   // Wait for the kernel to complete

    return 0;
}
