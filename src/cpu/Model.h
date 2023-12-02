//
// Created by kient on 6/17/2023.
//

#ifndef MASS_MODEL_H
#define MASS_MODEL_H


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


#endif //MASS_MODEL_H
