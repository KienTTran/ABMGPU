//
// Created by kient on 6/17/2023.
//

#ifndef ABMGPU_GPUBUFFER_CUH
#define ABMGPU_GPUBUFFER_CUH

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "GPUPerson.cuh"
#include "../utils/GPURandom.cuh"
#include "Population.cuh"
#include "GPUEntity.cuh"
#include "../cpu/Renderer.h"
#include <thread>

class GPUBuffer {
public:
    struct remove_func
    {
        template <typename T>
        __host__ __device__
        bool operator()(T &t){
            return (thrust::get<1>(t) < 0); // could change to other kinds of tests
        }
    };

    struct is_minus_one
    {
        __host__ __device__
        bool operator()(const int x)
        {
            return (x == -1);
        }
    };
    thrust::device_vector<GPUPerson> buffer_person_;
    thrust::device_vector<bool> buffer_person_render_flag_;
    thrust::device_vector<int> buffer_person_render_index_;

    std::thread buffer_thread_;
    Population *population_;
    Renderer *renderer_;

public:
    __host__ __device__ GPUBuffer();
    __host__ __device__ ~GPUBuffer();
    __host__ void initPopOnGPU(Population *population);
    __host__ size_t GetFreeVRam(int gpu_id);
    __host__ size_t GetUsedVRam(int gpu_id);
    __host__ void updateRender();
    __host__ void update();
    void start();
    void startThread();
};


#endif //ABMGPU_GPUBUFFER_CUH
