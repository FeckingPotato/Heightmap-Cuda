
struct PerlinNoiseGPU {
private:
    __device__
    float static fade(float t);
    __device__
    float static lerp(float t, float a, float b);
    __device__
    float static grad(int hash, float x, float y, float z);
public:
    __device__
    float static getNoise(float x, float y, float z, const int* permutation);

};

__global__
void getNoiseArray(float* result, unsigned int size, const int* permutation, float scale = 1.0f);
