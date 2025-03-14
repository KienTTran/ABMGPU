//
// Created by kient on 6/17/2023.
//

#include "GPURandom.cuh"
#include "../utils/GPUUtils.cuh"

GPURandom::GPURandom() {
    d_states = nullptr;
    n_threads = 1024;
}

GPURandom::~GPURandom() {
    cudaFree(d_states);
}

__global__ void setup(curandState *state, long seed)
{
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

void GPURandom::init(int n, long seed) {
    cudaMalloc((void **) &d_states, sizeof(curandState) * n);
    n_blocks = (n + n_threads + 1) / n_threads;
    setup<<<n_blocks,n_threads>>>(d_states, seed);
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaPeekAtLastError());
}

void GPURandom::free() {
    cudaFree(d_states);
}

