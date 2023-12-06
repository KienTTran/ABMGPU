//
// Created by kient on 6/17/2023.
//

#include "GPUPerson.cuh"

__host__ __device__ GPUPerson::GPUPerson(){
    id = -1;
    pop_index = -1;
    cell_id = -1;
    cell_all_id = -1;
    cell_has_people_id = -1;
    cell_col = -1;
    cell_row = -1;
    velocity = 0.0;
    state = ALIVE;
};
__host__ __device__ GPUPerson::~GPUPerson(){
    id = -1;
    pop_index = -1;
    cell_id = -1;
    cell_all_id = -1;
    cell_has_people_id = -1;
    cell_col = -1;
    cell_row = -1;
    velocity = 0.0;
    state = ALIVE;
};