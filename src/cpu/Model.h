//
// Created by kient on 6/17/2023.
//

#ifndef ABMGPU_MODEL_H
#define ABMGPU_MODEL_H


#include "../utils/GPURandom.cuh"
#include "../gpu/GPUBuffer.cuh"
#include "Renderer.h"

class Model {
public:
    Population *population;
    Renderer *renderer;
    GPUBuffer *gpu_buffer;
    GPUEntity *gpu_entity;
public:
    Model();
    ~Model();
    void init();
    void run();
};


#endif //ABMGPU_MODEL_H
