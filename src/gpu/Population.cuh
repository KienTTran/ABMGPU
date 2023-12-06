//
// Created by kient on 6/17/2023.
//

#ifndef ABMGPU_POPULATION_CUH
#define ABMGPU_POPULATION_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../gpu/GPUPerson.cuh"
#include "../utils/GPUUtils.cuh"
#include "../cpu/Config.h"


class Population {
public:

    //on HOST
    int *n_people_1_pop_current;
    int *n_new_people;
    int *n_dead_people;
    bool add_person;
    bool remove_person;
    std::mutex add_person_mtx;
    std::mutex remove_person_mtx;
    glm::mat4 h_view_mat;
    glm::mat4 h_projection_mat;
    thrust::host_vector<thrust::tuple<int,int,int,int,int,int>> *asc_cell_all_current;
    thrust::host_vector<thrust::tuple<int,int,int,int,int,int>> *asc_cell_people_current;
    thrust::host_vector<glm::mat4> *h_people_models;
    thrust::host_vector<glm::vec4> *h_people_colors;
    thrust::host_vector<GPUPerson> *h_population;
    thrust::host_vector<int> h_cell_pop_map;
    thrust::host_vector<glm::vec4> h_base_person_pos;
    thrust::host_vector<int> *h_people_render_indices;

    //on DEVICE CUDA
    thrust::device_vector<glm::vec4> d_population_colors; // for init pop color on device
    thrust::device_vector<glm::vec4> *d_cell_colors; // for init cell color on device
    glm::mat4 *d_view_mat;
    glm::mat4 *d_projection_mat;
    thrust::device_vector<glm::vec4> d_base_person_pos;
    thrust::device_vector<bool> *d_people_render_flags;
    thrust::device_vector<int> *d_people_render_indices;

    //On DEVICE OGL
    struct cudaGraphicsResource **d_cuda_buffer_model;
    size_t *d_ogl_buffer_model_num_bytes; // to get models data from gpu_buffer
    struct cudaGraphicsResource **d_cuda_buffer_color;
    size_t *d_ogl_buffer_color_num_bytes;// to get colors data from gpu_buffer
    glm::mat4 **d_ogl_buffer_model_ptr;
    glm::vec4 **d_ogl_buffer_color_ptr;

    int counter = 0;



public:
    Population();
    ~Population();
    void init();
};


#endif //ABMGPU_POPULATION_CUH
