//
// Created by kient on 6/17/2023.
//

#include "Population.cuh"

Population::Population() {
}

Population::~Population() {
}

void Population::init() {
    h_population_colors = thrust::host_vector<glm::vec4>(Config::getInstance().n_pops);
    h_population = new thrust::host_vector<GPUPerson>[Config::getInstance().n_pops];
    h_people_models = new thrust::host_vector<glm::mat4>[Config::getInstance().n_pops];
    h_people_colors = new thrust::host_vector<glm::vec4>[Config::getInstance().n_pops];
    h_people_velocities = new thrust::host_vector<float>[Config::getInstance().n_pops];

    d_population_colors = thrust::device_vector<glm::vec4>(Config::getInstance().n_pops);
    d_people_models = new thrust::device_vector<glm::mat4>[Config::getInstance().n_pops];
    d_people_colors = new thrust::device_vector<glm::vec4>[Config::getInstance().n_pops];
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++){
        h_population[p_index] = thrust::host_vector<GPUPerson>(Config::getInstance().n_people_1_pop[p_index]);
        h_people_models[p_index] = thrust::host_vector<glm::mat4>(Config::getInstance().n_people_1_pop[p_index]);
        h_people_colors[p_index] = thrust::host_vector<glm::vec4>(Config::getInstance().n_people_1_pop[p_index]);
        h_people_velocities[p_index] = thrust::host_vector<float>(Config::getInstance().n_people_1_pop[p_index]);

        d_people_models[p_index] = thrust::device_vector<glm::mat4>(Config::getInstance().n_people_1_pop[p_index],glm::mat4(1.0f));
        d_people_colors[p_index] = thrust::device_vector<glm::vec4>(Config::getInstance().n_people_1_pop[p_index],glm::vec4(1.0f));
    }
}
