//
// Created by kient on 6/17/2023.
//

#ifndef MASS_GPUBUFFER_CUH
#define MASS_GPUBUFFER_CUH

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "GPUPerson.cuh"
#include "../utils/GPURandom.cuh"
#include "Population.cuh"
#include "GPUEntity.cuh"
#include <thread>

class GPUBuffer {
public:
    thrust::device_vector<GPUPerson> buffer_person_;
    thrust::device_vector<glm::mat4> buffer_person_model_;
    thrust::device_vector<glm::vec4> buffer_person_color_;
    thrust::device_vector<float> buffer_person_velocity_;
    std::thread buffer_thread_;
    Population *population_;
public:
    __host__ __device__ GPUBuffer();
    __host__ __device__ ~GPUBuffer();
    __host__ __device__ void initPopOnGPU(Population *population);
    __host__ size_t GetFreeVRam(int gpu_id);
    __host__ size_t GetUsedVRam(int gpu_id);
    __host__ __device__ void update();
    void updateThread();
    void join();
};


#endif //MASS_GPUBUFFER_CUH
