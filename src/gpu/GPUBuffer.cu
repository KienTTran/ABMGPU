//
// Created by kient on 6/17/2023.
//

#include "GPUBuffer.cuh"
#include "../utils/GPUUtils.cuh"
#include "../cpu/Config.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <random>

__host__ __device__ GPUBuffer::GPUBuffer(){
};

__host__ __device__ GPUBuffer::~GPUBuffer(){
};


__host__ size_t GPUBuffer::GetFreeVRam(int gpu_id)
{
    cudaSetDevice(gpu_id);

    size_t l_free = 0;
    size_t l_Total = 0;
    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);

    return l_free;
}
__host__ size_t GPUBuffer::GetUsedVRam(int gpu_id)
{
    cudaSetDevice(gpu_id);

    size_t l_free = 0;
    size_t l_Total = 0;
    cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);

    return l_Total - l_free;
}

__global__ void add_person(uint64_t work_from, uint64_t work_to, GPUPerson *people,curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[thread_index];
    for (uint64_t index = thread_index; index < (work_to-work_from); index += stride) {
        people[index] = GPUPerson();
        people[index].id = work_from + index;
        people[index].state = GPUPerson::ALIVE;
    __syncthreads();
    }
    state[thread_index] = localState;
}

__global__ void add_person_entity(uint64_t work_from, uint64_t work_to, int p_index, float width, float height, glm::vec4 *pop_colors,
                                  glm::mat4 *people_models, glm::vec4 *people_colors,  float *people_velocities,
                                  curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[thread_index];
    for (uint64_t index = thread_index; index < (work_to-work_from); index += stride) {
        glm::mat4 model = people_models[index];
        float x_n1_1 = (curand_uniform(&localState)-0.5f)*2.0f;
        float y_n1_1 = (curand_uniform(&localState)-0.5f)*2.0f;
        float x = x_n1_1*width;
        float y = y_n1_1*height;
        float rot = curand_uniform(&localState)*360.0f;
        model = translate(model, glm::vec3(x, y, 0.0f));
        model = translate(model, glm::vec3(x, y, 0.0f));
        model = translate(model, glm::vec3(x, y, 0.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, 1.0f));
        model = rotate(model, rot, glm::vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, -1.0f));
        people_models[index] = model;
        people_colors[index] = pop_colors[p_index];
        people_velocities[index] = 0.001f;
        if(index == (work_to-work_from) - 1){
            printf("  [device][addPersonPos] from %lld to %lld index: %lld [0][0]: %f [1][1]: %f color:( %f %f %f)\n",
                   work_from, work_to, index, people_models[index][0][0], people_models[index][1][1],
                   people_colors[index][0], people_colors[index][1],people_colors[index][2]);
        }
        __syncthreads();
    }
    state[thread_index] = localState;
}

__global__ void adjust_person_entity(uint64_t work_batch, double i, float width, float height, GPUPerson *people, glm::mat4 *people_models, glm::vec4 *people_colors,curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[thread_index];
    for (uint64_t index = thread_index; index < work_batch; index += stride) {
        people[index].id = people[index].id + (int)i;
        glm::mat4 model = people_models[index];
        float velocity = 0.000005;
        float x_n1_1 = (curand_uniform(&localState)-0.5f)*2.0f;
        float y_n1_1 = (curand_uniform(&localState)-0.5f)*2.0f;
        float x = x_n1_1*velocity;
        float y = y_n1_1*velocity;
        float rot = curand_uniform(&localState)*360.0f*velocity;
        model = translate(model, glm::vec3(x, y, 0.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, 1.0f));
        model = rotate(model, rot, glm::vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, -1.0f));
        people_models[index] = model;
        __syncthreads();
    }
    state[thread_index] = localState;
}

__host__ __device__ void GPUBuffer::initPopOnGPU(Population *population){
    population_ = population;
    float width = (float)100;
    float height = (float)100;

    //Generate pop colors
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist_color(0.0,1.0);
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        population_->h_population_colors[p_index] = glm::vec4(dist_color(gen), dist_color(gen), dist_color(gen), 1.0f);
    }
    population_->d_population_colors = population_->h_population_colors;

    size_t mem_avai = GetFreeVRam(0);
    printf("Memory available: %zd\n", mem_avai);
    Config::getInstance().max_people_1_pop = Config::getInstance().min_people_1_pop;
    while(true){
        buffer_person_ = thrust::device_vector<GPUPerson>(Config::getInstance().max_people_1_pop);
        if(GetUsedVRam(0) >= mem_avai*Config::getInstance().p_max_mem_allow) break;
        Config::getInstance().max_people_1_pop += Config::getInstance().min_people_1_pop;
        buffer_person_.clear();
        thrust::device_vector<GPUPerson>().swap(buffer_person_);
    }
    printf("Max people 1 pop: %lld\n", Config::getInstance().max_people_1_pop);

    buffer_person_model_ = thrust::device_vector<glm::mat4>(1);
    buffer_person_color_ = thrust::device_vector<glm::vec4>(1);
    buffer_person_velocity_ = thrust::device_vector<float>(1);

    int n_threads = 1024;
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++){
        printf("h_population[%d].size(): %lld\n", p_index, population_->h_population[p_index].size());
        GPURandom::getInstance().init(population_->h_population[p_index].size());
        uint64_t batch_size = (Config::getInstance().max_people_1_pop < population_->h_population[p_index].size())
                ? Config::getInstance().max_people_1_pop : population_->h_population[p_index].size();
        //This is to make sure threads fit all people in population
        n_threads = (population_->h_population[p_index].size() < n_threads) ? population_->h_population[p_index].size() : n_threads;
        for(uint64_t remain = population_->h_population[p_index].size(); remain > 0; remain -= batch_size){
            batch_size = (remain < batch_size) ? remain : batch_size;
            uint64_t batch_from = population_->h_population[p_index].size() - remain;
            uint64_t batch_to = batch_from + batch_size;
            printf("Pop %d work batch size %lld remain %lld, from %lld to %lld\n", p_index,batch_size,remain,batch_from,batch_to);
            buffer_person_.resize(batch_size);
            buffer_person_model_.resize(batch_size);
            buffer_person_color_.resize(batch_size);
            buffer_person_velocity_.resize(batch_size);
            thrust::copy(population_->d_people_models[p_index].begin() + batch_from, population_->d_people_models[p_index].begin() + batch_to,buffer_person_model_.begin());
            thrust::copy(population_->d_people_colors[p_index].begin() + batch_from, population_->d_people_colors[p_index].begin() + batch_to,buffer_person_color_.begin());
            add_person<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from,batch_to,
                                                                thrust::raw_pointer_cast(buffer_person_.data()),
                                                                GPURandom::getInstance().d_states);
            add_person_entity<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from,batch_to,p_index,width,height,
                                                                        thrust::raw_pointer_cast(population_->d_population_colors.data()),
                                                                        thrust::raw_pointer_cast(buffer_person_model_.data()),
                                                                        thrust::raw_pointer_cast(buffer_person_color_.data()),
                                                                        thrust::raw_pointer_cast(buffer_person_velocity_.data()),
                                                                        GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + batch_from);
            thrust::copy(buffer_person_velocity_.begin(), buffer_person_velocity_.end(), population_->h_people_velocities[p_index].begin() + batch_from);
            thrust::copy(buffer_person_model_.begin(), buffer_person_model_.end(), population_->h_people_models[p_index].begin() + batch_from);
            thrust::copy(buffer_person_color_.begin(), buffer_person_color_.end(), population_->h_people_colors[p_index].begin() + batch_from);
            thrust::copy(buffer_person_model_.begin(), buffer_person_model_.end(), population_->d_people_models[p_index].begin() + batch_from);
            thrust::copy(buffer_person_color_.begin(), buffer_person_color_.end(), population_->d_people_colors[p_index].begin() + batch_from);
        }
    }
    for(int i = 0; i < Config::getInstance().n_pops; i++) {
        printf("[GPUBUffer] init pop %d last person id: %lld x:%f y: %f color: (%f %f %f)\n", i,
               population_->h_population[i][population_->h_population[i].size() - 1].id,
               population_->h_people_models[i][population_->h_people_models[i].size() - 1][0][0],
               population_->h_people_models[i][population_->h_people_models[i].size() - 1][0][1],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][0],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][1],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][2]);
    }
    printf("\n");
}

__host__ __device__ void GPUBuffer::update(){
    float width = (float)Config::getInstance().window_width;
    float height = (float)Config::getInstance().window_height;

    int n_threads = 1024;
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        uint64_t batch_size = (Config::getInstance().max_people_1_pop < population_->h_population[p_index].size())
                              ? Config::getInstance().max_people_1_pop : population_->h_population[p_index].size();
        //This is to make sure threads fit all people in population
        n_threads = (population_->h_population[p_index].size() < n_threads) ? population_->h_population[p_index].size() : n_threads;
        for (uint64_t remain = population_->h_population[p_index].size(); remain > 0; remain -= batch_size) {
            batch_size = (remain < batch_size) ? remain : batch_size;
            uint64_t batch_from = population_->h_population[p_index].size() - remain;
            uint64_t batch_to = population_->h_population[p_index].size() - remain + batch_size;
            buffer_person_.resize(batch_size);
            buffer_person_model_.resize(batch_size);
            buffer_person_color_.resize(batch_size);
            thrust::copy(population_->h_population[p_index].begin() + batch_from, population_->h_population[p_index].begin() + batch_to,buffer_person_.begin());
            thrust::copy(population_->h_people_velocities[p_index].begin() + batch_from, population_->h_people_velocities[p_index].begin() + batch_to,buffer_person_velocity_.begin());
            thrust::copy(population_->d_people_models[p_index].begin() + batch_from, population_->d_people_models[p_index].begin() + batch_to,buffer_person_model_.begin());
            thrust::copy(population_->d_people_colors[p_index].begin() + batch_from, population_->d_people_colors[p_index].begin() + batch_to,buffer_person_color_.begin());
            checkCudaErr(cudaGetLastError());
            adjust_person_entity<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_size, 0.0, width,height,
                                                                  thrust::raw_pointer_cast(buffer_person_.data()),
                                                                  thrust::raw_pointer_cast(buffer_person_model_.data()),
                                                                  thrust::raw_pointer_cast(buffer_person_color_.data()),
                                                                  GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + batch_from);
            thrust::copy(buffer_person_velocity_.begin(), buffer_person_velocity_.end(), population_->h_people_velocities[p_index].begin() + batch_from);
            thrust::copy(buffer_person_model_.begin(), buffer_person_model_.end(), population_->d_people_models[p_index].begin() + batch_from);
            thrust::copy(buffer_person_color_.begin(), buffer_person_color_.end(), population_->d_people_colors[p_index].begin() + batch_from);
            checkCudaErr(cudaGetLastError());
        }
    }
}

void GPUBuffer::updateThread() {
    printf("updateThread\n");
    while (true) {
        update();
    }
}

void GPUBuffer::join() {
    buffer_thread_ = std::thread(&GPUBuffer::updateThread, this);
    buffer_thread_.join();
}
