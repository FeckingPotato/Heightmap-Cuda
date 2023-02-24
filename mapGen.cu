#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include "mapGen.cuh"
#include "noise.cuh"


__constant__
int* permutation;

void generatePermutation(int seed) {
    std::iota(&permutation[0], &permutation[256], 0);
    std::default_random_engine engine(seed);
    std::shuffle(&permutation[0], &permutation[256], engine);
    for (int i = 0; i < 256; i++) {
        permutation[256+i] = permutation[i];
    }
}

void mapGen(float* result, unsigned int size, unsigned int offsetX, unsigned int offsetY, int seed, float scale) {
    cudaMallocManaged(&permutation, 512*sizeof(int));
    generatePermutation(seed);
    dim3 blockSize = {size/32, size/32};
    dim3 gridSize = {size/blockSize.x, size/blockSize.y};
    getNoiseArray<<<blockSize,gridSize>>>(result, size, offsetX, offsetY, permutation, scale);
}