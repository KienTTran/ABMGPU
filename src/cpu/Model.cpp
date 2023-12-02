//
// Created by kient on 6/17/2023.
//

#include "Model.h"

Model::Model() {
    population = new Population();
    gpu_buffer = new GPUBuffer();
    gpu_entity = new GPUEntity();
    renderer = new Renderer();
}
Model::~Model() {
}

void Model::init() {
    population->init();//init h_population
    gpu_buffer->initPopOnGPU(population);//init h_population using gpu
    gpu_entity->initPopEntity(population);//send h_population to render
    renderer->init(gpu_entity,Config::getInstance().window_width, Config::getInstance().window_height);
}

void Model::run() {
    renderer->detach();//render h_population detached
    gpu_buffer->join();//compute on h_population using gpu buffer
}