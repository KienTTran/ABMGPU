//
// Created by kient on 6/17/2023.
//

#ifndef ABMGPU_GPUPERSON_CUH
#define ABMGPU_GPUPERSON_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <glm/ext/matrix_float4x4.hpp>

class GPUPerson{
public:
    __host__ __device__ enum State{
        ALIVE = 0,
        DEAD = -1
    };
    int id;
    int pop_index;
    int cell_id;
    int cell_all_id;
    int cell_has_people_id;
    int cell_col;
    int cell_row;
    double velocity;
    State state;
    bool is_render;
    glm::mat4 last_model;
    glm::mat4 render_model;

    glm::vec4 last_color;
    glm::vec4 cell_color;
    glm::vec4 pop_color;
    glm::vec4 render_color;
public:
    __host__ __device__ GPUPerson();
    __host__ __device__ ~GPUPerson();
};


#endif //ABMGPU_GPUPERSON_H
