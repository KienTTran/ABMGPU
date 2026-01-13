//
// Created by kient on 6/17/2023.
//

#include "GPURandom.cuh"
#include "../utils/GPUUtils.cuh"

GPURandom::GPURandom() {
    d_states = nullptr;
    n_threads = 1024;
    rng_n = 0;
}
__global__ void setup(curandState *state, int n, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;
    curand_init(seed, id, 0, &state[id]);
}

void GPURandom::init(int n, long seed) {
    cudaMalloc((void **) &d_states, sizeof(curandState) * n);
    rng_n = n;
    n_blocks = (n + n_threads - 1) / n_threads;   // correct
    setup<<<n_blocks,n_threads>>>(d_states, n, (unsigned long long)seed);
    checkCudaErr(cudaPeekAtLastError());
    checkCudaErr(cudaDeviceSynchronize());
}


void GPURandom::free() {
    if (d_states) { cudaFree(d_states); d_states = nullptr; }
}

GPURandom::~GPURandom() { free(); }


