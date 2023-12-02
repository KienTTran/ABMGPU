//
// Created by kient on 6/17/2023.
//

#ifndef MASS_POPULATION_CUH
#define MASS_POPULATION_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../gpu/GPUPerson.cuh"
#include "../cpu/Config.h"


class Population {
public:
    //on HOST
    thrust::host_vector<glm::vec4> h_population_colors;
    thrust::host_vector<glm::mat4> *h_people_models;
    thrust::host_vector<glm::vec4> *h_people_colors;
    thrust::host_vector<GPUPerson> *h_population;
    thrust::host_vector<float> *h_people_velocities;

    //on DEVICE
    thrust::device_vector<glm::vec4> d_population_colors; // for init pop color on device
    thrust::device_vector<glm::mat4> *d_people_models; // for copy to entity->ogl_buffer_model_ptr
    thrust::device_vector<glm::vec4> *d_people_colors; // for copy to entity->ogl_buffer_color_ptr
public:
    Population();
    ~Population();
    void init();
};


#endif //MASS_POPULATION_CUH
