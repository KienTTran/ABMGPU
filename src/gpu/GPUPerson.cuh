//
// Created by kient on 6/17/2023.
//

#ifndef MASS_GPUPERSON_CUH
#define MASS_GPUPERSON_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <glm/ext/matrix_float4x4.hpp>

class GPUPerson{
public:
    enum State{
        ALIVE = 0,
        DEAD = -1
    };
    uint64_t id;
    State state;
public:
    __host__ __device__ GPUPerson();
    __host__ __device__ ~GPUPerson();
};


#endif //MASS_GPUPERSON_H
