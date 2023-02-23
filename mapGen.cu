#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include "mapGen.cuh"
#include "PerlinNoiseGPU.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int* permutation;

void generatePermutation(int seed) {
    std::iota(&permutation[0], &permutation[256], 0);
    std::default_random_engine engine(seed);
    std::shuffle(&permutation[0], &permutation[256], engine);
    for (int i = 0; i < 256; i++) {
        permutation[256+i] = permutation[i];
    }
}

void mapGen(float* result, unsigned int size, int seed, float scale) {
    cudaMallocManaged(&permutation, 512*sizeof(int));
    generatePermutation(seed);
    dim3 blockSize = {size/32, size/32};
    dim3 gridSize = {size/blockSize.x, size/blockSize.y};
    getNoiseArray<<<blockSize,gridSize>>>(result, size, permutation, scale);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}