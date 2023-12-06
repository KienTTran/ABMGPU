//
// Created by kient on 6/17/2023.
//

#include "Population.cuh"

Population::Population() {
}

Population::~Population() {
}

void Population::init() {
    asc_cell_all_current = new thrust::host_vector<thrust::tuple<int,int,int,int,int,int>>[Config::getInstance().n_pops];
    asc_cell_people_current = new thrust::host_vector<thrust::tuple<int,int,int,int,int,int>>[Config::getInstance().n_pops];

    n_people_1_pop_current = new int[Config::getInstance().n_pops];
    n_new_people = new int[Config::getInstance().n_pops];
    n_dead_people = new int[Config::getInstance().n_pops];
    h_base_person_pos = thrust::host_vector<glm::vec4>(Config::getInstance().n_pops);
    h_population = new thrust::host_vector<GPUPerson>[Config::getInstance().n_pops];
    h_people_models = new thrust::host_vector<glm::mat4>[Config::getInstance().n_pops];
    h_people_colors = new thrust::host_vector<glm::vec4>[Config::getInstance().n_pops];
    h_people_render_indices = new thrust::host_vector<int>[Config::getInstance().n_pops];
    add_person = false;
    remove_person = false;

    d_population_colors = thrust::device_vector<glm::vec4>(Config::getInstance().n_pops);
    d_cell_colors = new thrust::device_vector<glm::vec4>[Config::getInstance().n_pops];
    d_base_person_pos = thrust::device_vector<glm::vec4>(Config::getInstance().n_pops);
    d_people_render_flags = new thrust::device_vector<bool>[Config::getInstance().n_pops];
    d_people_render_indices = new thrust::device_vector<int>[Config::getInstance().n_pops];

    checkCudaErr(cudaMalloc(&d_view_mat, sizeof(glm::mat4)));
    checkCudaErr(cudaMalloc(&d_projection_mat, sizeof(glm::mat4)));

    d_cuda_buffer_model = new cudaGraphicsResource*[Config::getInstance().n_pops];
    d_ogl_buffer_model_num_bytes = new size_t[Config::getInstance().n_pops];
    d_cuda_buffer_color = new cudaGraphicsResource*[Config::getInstance().n_pops];
    d_ogl_buffer_color_num_bytes = new size_t[Config::getInstance().n_pops];
    d_ogl_buffer_model_ptr = new glm::mat4*[Config::getInstance().n_pops];
    d_ogl_buffer_color_ptr = new glm::vec4*[Config::getInstance().n_pops];

    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++){
        for(int c_index = 0; c_index < Config::getInstance().asc_cell_all[p_index].size(); c_index++) {
            asc_cell_all_current[p_index].push_back(
                    thrust::make_tuple(std::get<0>(Config::getInstance().asc_cell_all[p_index][c_index]),
                                       std::get<1>(Config::getInstance().asc_cell_all[p_index][c_index]),
                                       std::get<2>(Config::getInstance().asc_cell_all[p_index][c_index]),
                                       std::get<3>(Config::getInstance().asc_cell_all[p_index][c_index]),
                                       std::get<4>(Config::getInstance().asc_cell_all[p_index][c_index]),
                                       std::get<5>(Config::getInstance().asc_cell_all[p_index][c_index])));
        }
        for(int c_index = 0; c_index < Config::getInstance().asc_cell_people[p_index].size(); c_index++) {
            asc_cell_people_current[p_index].push_back(
                    thrust::make_tuple(std::get<0>(Config::getInstance().asc_cell_people[p_index][c_index]),
                                       std::get<1>(Config::getInstance().asc_cell_people[p_index][c_index]),
                                       std::get<2>(Config::getInstance().asc_cell_people[p_index][c_index]),
                                       std::get<3>(Config::getInstance().asc_cell_people[p_index][c_index]),
                                       std::get<4>(Config::getInstance().asc_cell_people[p_index][c_index]),
                                       std::get<5>(Config::getInstance().asc_cell_people[p_index][c_index])));
        }

        n_people_1_pop_current[p_index] = Config::getInstance().n_people_1_pop_base[p_index];
        n_new_people[p_index] = 0;
        n_dead_people[p_index] = 0;

        h_population[p_index] = thrust::host_vector<GPUPerson>(Config::getInstance().n_people_1_pop_max[p_index],GPUPerson());
        h_people_models[p_index] = thrust::host_vector<glm::mat4>(Config::getInstance().n_people_1_pop_max[p_index],glm::mat4(1.0f));
        h_people_colors[p_index] = thrust::host_vector<glm::vec4>(Config::getInstance().n_people_1_pop_max[p_index],glm::vec4(1.0f));
        h_people_render_indices[p_index] = thrust::host_vector<int>(Config::getInstance().n_people_1_pop_max[p_index],0);

        d_cell_colors[p_index] = thrust::device_vector<glm::vec4>(Config::getInstance().asc_cell_people[p_index].size());
        d_people_render_flags[p_index] = thrust::device_vector<bool>(Config::getInstance().n_people_1_pop_max[p_index],false);
        d_people_render_indices[p_index] = thrust::device_vector<int>(1,0);

        checkCudaErr(cudaMalloc(&d_ogl_buffer_model_ptr[p_index], Config::getInstance().n_people_1_pop_max[p_index] * sizeof(glm::mat4)));
        checkCudaErr(cudaMalloc(&d_ogl_buffer_color_ptr[p_index], Config::getInstance().n_people_1_pop_max[p_index] * sizeof(glm::vec4)));
    }
}
