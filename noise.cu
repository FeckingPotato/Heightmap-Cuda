#include "PerlinNoiseGPU.cuh"

__device__
float PerlinNoiseGPU::fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
__device__
float PerlinNoiseGPU::lerp(float t, float a, float b) {
    return a + t * (b - a);
}
__device__
float PerlinNoiseGPU::grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y,
            v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__device__
float PerlinNoiseGPU::getNoise(float x, float y, float z, const int* permutation) {
    int X = (int) floor(x) & 255,
            Y = (int) floor(y) & 255,
            Z = (int) floor(z) & 255;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    float u,v,w;
    u = fade(x);
    v = fade(y);
    w = fade(z);
    int A = permutation[X] + Y;
    int AA = permutation[A] + Z;
    int AB = permutation[A + 1] + Z;
    int B = permutation[X + 1] + Y;
    int BA = permutation[B] + Z;
    int BB = permutation[B + 1] + Z;
    return lerp(w,
                lerp(v,
                     lerp(u,
                          grad(permutation[AA], x, y, z),
                          grad(permutation[BA], x - 1, y, z)),
                     lerp(u,
                          grad(permutation[AB], x, y - 1, z),
                          grad(permutation[BB], x - 1, y - 1, z))),
                lerp(v,
                     lerp(u,
                          grad(permutation[AA + 1], x, y, z - 1),
                          grad(permutation[BA + 1], x - 1, y, z - 1)),
                     lerp(u,
                          grad(permutation[AB + 1], x, y - 1, z - 1),
                          grad(permutation[BB + 1], x - 1, y - 1, z - 1))));
}

__global__
void getNoiseArray(float* result, unsigned int size, const int* permutation, float scale) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        for (unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; j < size; j += blockDim.y * gridDim.y)
        result[i*size+j] = PerlinNoiseGPU::getNoise(
                scale * ((float) i) / ((float) size),
                scale * ((float) j) / ((float) size),
                0.0, permutation);
    }
};
