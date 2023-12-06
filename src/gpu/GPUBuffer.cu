//
// Created by kient on 6/17/2023.
//

#include "GPUBuffer.cuh"
#include "../utils/GPUUtils.cuh"
#include "../cpu/Config.h"
#include "../utils/Thread.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <iostream>
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

__global__ void add_people_to_cells(int work_from, int work_to, int work_batch, int p_index, float width, float height, int n_cols, int n_rows,double velocity,
                                     glm::vec4 pop_color,glm::vec4 cell_color,GPUPerson buffer_people[],
                                     thrust::tuple<int,int,int,int,int,int> cell_data,curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState local_state = state[thread_index];
    for (int index = thread_index; index < (work_batch); index += stride) {
        //person info
        buffer_people[index] = GPUPerson();
        buffer_people[index].id = work_from + index;
        buffer_people[index].cell_id = index;
        buffer_people[index].cell_all_id = thrust::get<0>(cell_data);
        buffer_people[index].cell_has_people_id = thrust::get<1>(cell_data);
        buffer_people[index].pop_index = thrust::get<2>(cell_data);
        buffer_people[index].cell_col = thrust::get<3>(cell_data);
        buffer_people[index].cell_row = thrust::get<4>(cell_data);
        buffer_people[index].velocity = velocity;
        buffer_people[index].state = GPUPerson::ALIVE;
        buffer_people[index].is_render = true;

        //entity info
        //Set position follow .asc file
        float unit_x = width/(float)n_cols;
        float unit_y = height/(float)n_rows;
        float base_x_left = unit_x*buffer_people[index].cell_col;
        float base_x_right = unit_x*buffer_people[index].cell_col + unit_x;
        float base_y_bottom = unit_y*buffer_people[index].cell_row;
        float base_y_top = unit_y*buffer_people[index].cell_row + unit_y;
        float range_x = base_x_right - base_x_left;
        float range_y = base_y_top - base_y_bottom;
        float rand_x = curand_uniform(&local_state);
        float rand_y = curand_uniform(&local_state);
        float x = p_index*width + rand_x*range_x + base_x_left;//shift population to right, for multiple pops
        float y = height - (rand_y*range_y + base_y_bottom);//OGL from bottom to ptop, so invert Y axis only

        glm::mat4 model = glm::mat4(1.0f);
        model = translate(model, glm::vec3(x, y, 0.0f));
        buffer_people[index].last_model = model;
        buffer_people[index].render_model = buffer_people[index].last_model;
        buffer_people[index].last_color = cell_color;
        buffer_people[index].render_color = buffer_people[index].last_color;
        buffer_people[index].cell_color = cell_color;
        buffer_people[index].pop_color = pop_color;

//        buffer_models[index] = model;
//        buffer_colors[index] = cell_color;//glm::vec4(1.0f,0.0f,0.0f,1.0f);//pop_colors[p_index];


//        if(index == work_batch - 1){
//            printf("[device] creating person id: %d %d %d x:%f y:%f color: (%f %f %f) "
//                   "cell_all %d cell_hp %d cell_x_y (%d, %d)\n",
//                   buffer_people[index].id,
//                   buffer_people[index].cell_index,
//                   buffer_people[index].pop_index,
//                   buffer_people[index].render_model[3][0],
//                   buffer_people[index].render_model[3][1],
//                   buffer_people[index].render_color[0],
//                   buffer_people[index].render_color[1],
//                   buffer_people[index].render_color[2],
//                   buffer_people[index].cell_all_id,
//                   buffer_people[index].cell_has_people_id,
//                   buffer_people[index].cell_col,
//                   buffer_people[index].cell_row);
//        }
        __syncthreads();
    }
    state[thread_index] = local_state;
}

__global__ void add_new_people_to_cells(int n_cells, int n_new_start, int *n_new_current, double birth_rate, int p_index, float width, float height, int n_cols, int n_rows, double velocity,
                                          glm::vec4 pop_color, glm::vec4 cell_colors[],GPUPerson buffer_people[],
                                         thrust::tuple<int,int,int,int,int,int> cell_data[],curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState local_state = state[thread_index];
    for (int c_index = thread_index; c_index < n_cells; c_index+=stride) {
        float poisson_mean = thrust::get<5>(cell_data[c_index])*birth_rate/365;
        int n_new_people_1_cell = curand_poisson(&local_state, poisson_mean);
        if(n_new_people_1_cell > 0){
            cell_data[c_index] = thrust::make_tuple(thrust::get<0>(cell_data[c_index]),
                                                    thrust::get<1>(cell_data[c_index]),
                                                    thrust::get<2>(cell_data[c_index]),
                                                    thrust::get<3>(cell_data[c_index]),
                                                    thrust::get<4>(cell_data[c_index]),
                                                    thrust::get<5>(cell_data[c_index]) + n_new_people_1_cell);
            printf("[device] cell %d (%d,%d) adding %d\n",
                   c_index,thrust::get<3>(cell_data[c_index]),thrust::get<4>(cell_data[c_index]),
                   n_new_people_1_cell);
            for(int n_index = 0; n_index < n_new_people_1_cell; n_index++){
                int pop_index = atomicAdd(n_new_current, 1);
                buffer_people[pop_index-n_new_start] = GPUPerson();
                buffer_people[pop_index-n_new_start].id = pop_index;
                buffer_people[pop_index-n_new_start].cell_id = thrust::get<5>(cell_data[c_index]) + n_index;
                buffer_people[pop_index-n_new_start].cell_all_id = thrust::get<0>(cell_data[c_index]);
                buffer_people[pop_index-n_new_start].cell_has_people_id = thrust::get<1>(cell_data[c_index]);
                buffer_people[pop_index-n_new_start].pop_index = thrust::get<2>(cell_data[c_index]);
                buffer_people[pop_index-n_new_start].cell_col = thrust::get<3>(cell_data[c_index]);
                buffer_people[pop_index-n_new_start].cell_row = thrust::get<4>(cell_data[c_index]);
                buffer_people[pop_index-n_new_start].velocity = velocity;
                buffer_people[pop_index-n_new_start].state = GPUPerson::ALIVE;
                buffer_people[pop_index-n_new_start].is_render = true;

                //entity info - set position follow .asc file
                float unit_x = width/(float)n_cols;
                float unit_y = height/(float)n_rows;
                float base_x_left = unit_x*buffer_people[pop_index-n_new_start].cell_col;
                float base_x_right = unit_x*buffer_people[pop_index-n_new_start].cell_col + unit_x;
                float base_y_bottom = unit_y*buffer_people[pop_index-n_new_start].cell_row;
                float base_y_top = unit_y*buffer_people[pop_index-n_new_start].cell_row + unit_y;
                float range_x = base_x_right - base_x_left;
                float range_y = base_y_top - base_y_bottom;
                float rand_x = curand_uniform(&local_state);
                float rand_y = curand_uniform(&local_state);
                float x = p_index*width + rand_x*range_x + base_x_left;//shift population to right, for multiple pops
                float y = height - (rand_y*range_y + base_y_bottom);//OGL from bottom to ptop, so invert Y axis only

                glm::mat4 model = glm::mat4(1.0f);
                model = translate(model, glm::vec3(x, y, 0.0f));
                buffer_people[pop_index-n_new_start].last_model = model;
                buffer_people[pop_index-n_new_start].render_model = buffer_people[pop_index-n_new_start].last_model;
                buffer_people[pop_index-n_new_start].last_color = glm::vec4(0.0f,1.0f,0.0f,1.0f);
                buffer_people[pop_index-n_new_start].render_color = buffer_people[pop_index-n_new_start].last_color;
                buffer_people[pop_index-n_new_start].cell_color = cell_colors[c_index];
                buffer_people[pop_index-n_new_start].pop_color = pop_color;

//                printf("[device] cell %d %d adding person id: %d (%d) %d %d x:%f y:%f color: (%f %f %f) "
//                       "cell_all %d cell_hp %d cell_x_y (%d, %d)\n",
//                       c_index,
//                       n_index,
//                       buffer_people[pop_index-n_new_start].id,
//                       n_new_people_1_cell,
//                       buffer_people[pop_index-n_new_start].cell_id,
//                       buffer_people[pop_index-n_new_start].pop_index,
//                       buffer_people[pop_index-n_new_start].render_model[3][0],
//                       buffer_people[pop_index-n_new_start].render_model[3][1],
//                       buffer_people[pop_index-n_new_start].render_color[0],
//                       buffer_people[pop_index-n_new_start].render_color[1],
//                       buffer_people[pop_index-n_new_start].render_color[2],
//                       buffer_people[pop_index-n_new_start].cell_all_id,
//                       buffer_people[pop_index-n_new_start].cell_has_people_id,
//                       buffer_people[pop_index-n_new_start].cell_col,
//                       buffer_people[pop_index-n_new_start].cell_row);
            }
        }
        __syncthreads();
    }
    state[thread_index] = local_state;
}

__global__ void assign_dead_people_in_cells(int n_cells, int *n_dead_current, double dead_rate, int p_index, float width, float height, int n_cols, int n_rows,
                                          glm::vec4 pop_color, glm::vec4 cell_colors[],
                                          int dead_people[], thrust::tuple<int,int,int,int,int,int> cell_data[],curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState local_state = state[thread_index];
    for (int c_index = thread_index; c_index < n_cells; c_index+=stride) {
        float poisson_mean = thrust::get<5>(cell_data[c_index])*dead_rate/365;
        int n_dead_people_1_cell = curand_poisson(&local_state, poisson_mean);
        if(n_dead_people_1_cell > 0){
            cell_data[c_index] = thrust::make_tuple(thrust::get<0>(cell_data[c_index]),
                                                    thrust::get<1>(cell_data[c_index]),
                                                    thrust::get<2>(cell_data[c_index]),
                                                    thrust::get<3>(cell_data[c_index]),
                                                    thrust::get<4>(cell_data[c_index]),
                                                    thrust::get<5>(cell_data[c_index]) - n_dead_people_1_cell);
            printf("[device] cell %d (%d,%d) removing %d\n",
                   c_index,thrust::get<3>(cell_data[c_index]),thrust::get<4>(cell_data[c_index]),
                   n_dead_people_1_cell);
            for(int n_index = 0; n_index < n_dead_people_1_cell; n_index++){
                int buffer_id = atomicAdd(n_dead_current, 1);
                int d_index = curand_uniform(&local_state)*thrust::get<5>(cell_data[c_index]);
                dead_people[buffer_id] = d_index;
//                printf("[device] cell %d %d removing person id: %d (%d)\n",
//                       c_index,
//                       n_index,
//                       dead_people[buffer_id],
//                       n_dead_people_1_cell);
            }
        }
        __syncthreads();
    }
    state[thread_index] = local_state;
}

__global__ void remove_dead_people_in_cells(int work_from,int work_to,int work_batch, int n_dead,
                                             GPUPerson buffer_people[], int dead_people_id[]){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n_index = thread_index; n_index < work_batch; n_index+=stride) {
        for(int d_index = 0; d_index < n_dead; d_index++){
            if(buffer_people[n_index].cell_id == dead_people_id[d_index]){
//                printf("[device] %d removing person id: %d\n",buffer_people[n_index].cell_has_people_id, buffer_people[n_index].cell_id);
                buffer_people[n_index].id = -1;
                buffer_people[n_index].cell_id = -1;
                buffer_people[n_index].cell_all_id = -1;
                buffer_people[n_index].cell_has_people_id = -1;
                buffer_people[n_index].cell_col = -1;
                buffer_people[n_index].cell_row = -1;
                buffer_people[n_index].pop_index = -1;
                buffer_people[n_index].render_model = buffer_people[n_index].last_model;
                buffer_people[n_index].render_color = glm::vec4(1.0f,0.0f,0.0f,1.0f);
                buffer_people[n_index].state = GPUPerson::DEAD;
                buffer_people[n_index].is_render = false;
            }
        }
        __syncthreads();
    }
}

__global__ void update_person_entity(int work_from, int work_to, int work_batch, int p_index, float width, float height,
                                     GPUPerson *buffer_people, curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState local_state = state[thread_index];
    for (int index = thread_index; index < work_batch; index += stride) {
        if(buffer_people[index].state == GPUPerson::DEAD || buffer_people[index].id == -1 || buffer_people[index].id == -1){
            return;
        }
        glm::mat4 model = buffer_people[index].last_model;
        glm::vec4 color = buffer_people[index].cell_color;
        float x_n1_1 = (curand_uniform(&local_state)-0.5f)*2.0f;
        float y_n1_1 = (curand_uniform(&local_state)-0.5f)*2.0f;
        float x = x_n1_1*buffer_people[index].velocity;
        float y = y_n1_1*buffer_people[index].velocity;
        float rot = curand_uniform(&local_state)*360.0f*buffer_people[index].velocity;
        model = translate(model, glm::vec3(x, y, 0.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, 1.0f));
        model = rotate(model, rot, glm::vec3(0.0f, 0.0f, 1.0f));
        model = translate(model, glm::vec3(0.0f, 0.0f, -1.0f));
        buffer_people[index].render_model = model;
        buffer_people[index].render_color = color;
        buffer_people[index].last_model = model;
        buffer_people[index].last_color = color;
        __syncthreads();
    }
    state[thread_index] = local_state;
}

__global__ void calculate_render_entity(int work_batch, glm::mat4 *projection, glm::mat4 *view, GPUPerson buffer_people[],
                                          glm::vec4 base_person_pos, bool *people_render_flags){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = thread_index; index < work_batch; index += stride) {
        glm::vec4 person_screen_pos = projection[0] * view[0] * buffer_people[index].render_model * base_person_pos;
        if (person_screen_pos.x >= -1.0f && person_screen_pos.x <= 1.0f && person_screen_pos.y >= -1.0f && person_screen_pos.y <= 1.0f) {
            people_render_flags[index] = true;
        }
        else {
            people_render_flags[index] = false;
        }
        __syncthreads();
    }
}

__global__ void update_ogl_buffer(int work_from, int work_to, int work_batch, GPUPerson buffer_people[],
                                  glm::mat4 people_models[], glm::vec4 people_colors[]){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = thread_index; index < work_batch; index += stride) {
        people_models[work_from+index] = buffer_people[index].render_model;
        people_colors[work_from+index] = buffer_people[index].render_color;
        __syncthreads();
    }
}

__host__ void GPUBuffer::initPopOnGPU(Population *population){
    population_ = population;
    float width = Config::getInstance().test_width > 0 ? Config::getInstance().test_width : Config::getInstance().window_width;
    float height = Config::getInstance().test_height > 0 ? Config::getInstance().test_height : Config::getInstance().window_height;

    //Generate pop colors
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist_color(0.5,1.0);
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        population_->d_population_colors[p_index] = glm::vec4(dist_color(gen), dist_color(gen), dist_color(gen), 1.0f);
        for(int c_index = 0; c_index < Config::getInstance().asc_cell_people[p_index].size(); c_index++){
            gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
            population_->d_cell_colors[p_index][c_index] = glm::vec4(dist_color(gen), dist_color(gen), dist_color(gen), 1.0f);
        }
    }
    size_t mem_avai = GetFreeVRam(0);
    printf("[GPUBuffer] Memory available: %zd\n", mem_avai);
    Config::getInstance().max_people_1_batch = Config::getInstance().min_people_1_batch;
//    while(true){
//        buffer_person_ = thrust::device_vector<GPUPerson>(Config::getInstance().max_people_1_batch);
//        if(GetUsedVRam(0) >= mem_avai*Config::getInstance().p_max_mem_allow){
//            Config::getInstance().max_people_1_batch -= Config::getInstance().min_people_1_batch*Config::getInstance().p_people_1_batch_step;
//            break;
//        }
//        Config::getInstance().max_people_1_batch += Config::getInstance().min_people_1_batch*Config::getInstance().p_people_1_batch_step;
//        buffer_person_.clear();
//        thrust::device_vector<GPUPerson>().swap(buffer_person_);
//    }
    printf("[GPUBuffer] Max people 1 pop: %d\n", Config::getInstance().max_people_1_batch);
    GPURandom::getInstance().init(Config::getInstance().max_people_1_batch);

    buffer_person_render_flag_ = thrust::device_vector<bool>(1);

    auto tp_start = std::chrono::high_resolution_clock::now();
    int n_threads = 1024;
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++){
        printf("[GPUBuffer] Population %d size: %d\n", p_index, Config::getInstance().n_people_1_pop_base[p_index]);
        printf("[GPUBuffer] Populate cell info\n");
        int batch_size_prev = 0;
        for(int c_index = 0; c_index < population_->asc_cell_people_current[p_index].size(); c_index++) {
            thrust::tuple<int,int,int,int,int,int> h_cell_data = population_->asc_cell_people_current[p_index][c_index];
            gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
            int batch_size = thrust::get<5>(h_cell_data);
            int batch_from = batch_size_prev;
            int batch_to = batch_from + batch_size;
            batch_size_prev = batch_to;
            buffer_person_.resize(batch_size);
            thrust::copy(population_->h_population[p_index].begin() + batch_from, population_->h_population[p_index].begin() + batch_to,buffer_person_.begin());
            n_threads = (batch_size < n_threads) ? batch_size : n_threads;
            add_people_to_cells<<<((batch_size + n_threads + 1)/n_threads), n_threads>>>(batch_from,batch_to,batch_size,p_index,width,height,
                                                                                             Config::getInstance().pop_asc_file[p_index]->NCOLS,
                                                                                             Config::getInstance().pop_asc_file[p_index]->NROWS,
                                                                                             Config::getInstance().velocity,
                                                                                             population_->d_population_colors[p_index],
                                                                                             population_->d_cell_colors[p_index][c_index],
                                                                                             thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                             h_cell_data,GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + batch_from);
        };
    }
    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        for(int j = Config::getInstance().n_people_1_pop_base[p_index] - 5; j < Config::getInstance().n_people_1_pop_base[p_index]; j++){
            printf("[GPUBuffer] Init pop %d person id: %d pop_index: %d x:%f y:%f color: (%f %f %f) "
                   "cell_all %d cell_hp %d cell_x_y (%d, %d)\n", p_index,
                   population_->h_population[p_index][j].id,
                   population_->h_population[p_index][j].pop_index,
                   population_->h_population[p_index][j].render_model[3][0],
                   population_->h_population[p_index][j].render_model[3][1],
                   population_->h_population[p_index][j].render_color[0],
                   population_->h_population[p_index][j].render_color[1],
                   population_->h_population[p_index][j].render_color[2],
                   population_->h_population[p_index][j].cell_all_id,
                   population_->h_population[p_index][j].cell_has_people_id,
                   population_->h_population[p_index][j].cell_col,
                   population_->h_population[p_index][j].cell_row
                   );
        }
    }
    buffer_person_.clear();
    thrust::device_vector<GPUPerson>().swap(buffer_person_);
    printf("[GPUBuffer] Init population time: %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
    printf("\n");
}

__host__ void GPUBuffer::update(){
    if(!Config::getInstance().is_window_rendered){
        return;
    }
    float width = Config::getInstance().test_width > 0 ? Config::getInstance().test_width : Config::getInstance().window_width;
    float height = Config::getInstance().test_height > 0 ? Config::getInstance().test_height : Config::getInstance().window_height;
    int n_threads = 1024;
    if(population_->add_person) {
        population_->add_person_mtx.lock();
        population_->add_person = false;
        population_->add_person_mtx.unlock();
        auto tp_start = std::chrono::high_resolution_clock::now();
        for (int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
            population_->n_new_people[p_index] = 0;
            thrust::host_vector<GPUPerson> h_new_people_pop;
            thrust::device_vector<GPUPerson> d_new_people_pop;
            thrust::device_vector<thrust::tuple<int,int,int,int,int,int>> d_cell_data = population_->asc_cell_people_current[p_index];
            int batch_from = population_->n_people_1_pop_current[p_index];
            int batch_to = Config::getInstance().n_people_1_pop_max[p_index];
            buffer_person_.resize(batch_to-batch_from);
            thrust::copy(population_->h_population[p_index].begin() + batch_from, population_->h_population[p_index].begin() + batch_to,buffer_person_.begin());
            int n_cells = population_->asc_cell_people_current[p_index].size();
            n_threads = (n_cells < n_threads) ? n_cells : n_threads;
            population_->n_new_people[p_index] = population_->n_people_1_pop_current[p_index];
            int *n_people_current;
            checkCudaErr(cudaMalloc(&n_people_current, sizeof(int)));
            checkCudaErr(cudaMemcpy(n_people_current, &population_->n_people_1_pop_current[p_index], sizeof(int), cudaMemcpyHostToDevice));
            add_new_people_to_cells<<<((n_cells + n_threads + 1)/n_threads), n_threads>>>(n_cells,population_->n_people_1_pop_current[p_index],n_people_current,Config::getInstance().birth_rate,p_index,width,height,
                                                                                            Config::getInstance().pop_asc_file[p_index]->NCOLS,
                                                                                            Config::getInstance().pop_asc_file[p_index]->NROWS,
                                                                                            Config::getInstance().velocity,
                                                                                            population_->d_population_colors[p_index],
                                                                                            thrust::raw_pointer_cast(population_->d_cell_colors[p_index].data()),
                                                                                            thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                            thrust::raw_pointer_cast(d_cell_data.data()),
                                                                                            GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            checkCudaErr(cudaMemcpy(&population_->n_people_1_pop_current[p_index], n_people_current, sizeof(int), cudaMemcpyDeviceToHost));
            population_->n_new_people[p_index] = population_->n_people_1_pop_current[p_index] - population_->n_new_people[p_index];
            population_->asc_cell_people_current[p_index] = d_cell_data;
            thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + (batch_from));
            printf("[GPUBuffer] After adding, total people in population %d: %d (%d)\n", p_index,population_->n_people_1_pop_current[p_index],population_->n_new_people[p_index]);
        }
        auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
        if(Config::getInstance().test_debug){
            printf("[GPUBuffer] Update population add time: %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
        }
        return;
    }

    if(population_->remove_person) {
        population_->remove_person_mtx.lock();
        population_->remove_person = false;
        population_->remove_person_mtx.unlock();
        auto tp_start = std::chrono::high_resolution_clock::now();
        for (int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
            population_->n_dead_people[p_index] = 0;
            thrust::host_vector<GPUPerson> h_dead_people_pop;
            thrust::device_vector<GPUPerson> d_dead_people_pop;
            thrust::device_vector<thrust::tuple<int,int,int,int,int,int>> d_cell_data = population_->asc_cell_people_current[p_index];
            thrust::device_vector<int> d_dead_people_id = thrust::device_vector<int>(Config::getInstance().n_people_1_pop_max[p_index] - population_->n_people_1_pop_current[p_index]);
            int n_cells = population_->asc_cell_people_current[p_index].size();
            n_threads = (n_cells < n_threads) ? n_cells : n_threads;
            population_->n_dead_people[p_index] = population_->n_people_1_pop_current[p_index];
            int *n_dead_current;
            int n_dead_start = 0;
            checkCudaErr(cudaMalloc(&n_dead_current, sizeof(int)));
            checkCudaErr(cudaMemcpy(n_dead_current, &n_dead_start, sizeof(int), cudaMemcpyHostToDevice));
            assign_dead_people_in_cells<<<((n_cells + n_threads + 1) / n_threads), n_threads>>>(n_cells,n_dead_current,Config::getInstance().death_rate,p_index, width,height,
                                                                                                 Config::getInstance().pop_asc_file[p_index]->NCOLS,
                                                                                                 Config::getInstance().pop_asc_file[p_index]->NROWS,
                                                                                                 population_->d_population_colors[p_index],
                                                                                                 thrust::raw_pointer_cast(population_->d_cell_colors[p_index].data()),
                                                                                                 thrust::raw_pointer_cast(d_dead_people_id.data()),
                                                                                                 thrust::raw_pointer_cast(d_cell_data.data()),
                                                                                                 GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            checkCudaErr(cudaMemcpy(&population_->n_dead_people[p_index], n_dead_current, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErr(cudaFree(n_dead_current));

            //Remove dead people
            int batch_size = (Config::getInstance().max_people_1_batch < population_->n_people_1_pop_current[p_index])
                             ? Config::getInstance().max_people_1_batch : population_->n_people_1_pop_current[p_index];
            //This is to make sure threads fit all people in population
            n_threads = (population_->n_people_1_pop_current[p_index] < n_threads) ? population_->n_people_1_pop_current[p_index] : n_threads;
            for (int remain = population_->n_people_1_pop_current[p_index]; remain > 0; remain -= batch_size) {
                batch_size = (remain < batch_size) ? remain : batch_size;
                int batch_from = population_->n_people_1_pop_current[p_index] - remain;
                int batch_to = population_->n_people_1_pop_current[p_index] - remain + batch_size;
                buffer_person_.resize(batch_size);
                remove_dead_people_in_cells<<<((batch_size + n_threads + 1) / n_threads), n_threads>>>(batch_from,batch_to,batch_size,
                                                                                                        population_->n_dead_people[p_index],
                                                                                                        thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                                        thrust::raw_pointer_cast(d_dead_people_id.data()));
                checkCudaErr(cudaDeviceSynchronize());
                checkCudaErr(cudaGetLastError());
                thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + batch_from);
            }
            population_->n_people_1_pop_current[p_index] -= population_->n_dead_people[p_index];
            printf("[GPUBuffer] After remove, total people in population %d: %d (%d)\n", p_index,population_->n_people_1_pop_current[p_index], population_->n_dead_people[p_index]);
        }
        auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
        if(Config::getInstance().test_debug){
            printf("[GPUBuffer] Update population remove time: %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
        }
        return;
    }
    auto tp_start = std::chrono::high_resolution_clock::now();
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        int batch_size = (population_->n_people_1_pop_current[p_index] < Config::getInstance().max_people_1_batch)
                     ? population_->n_people_1_pop_current[p_index] : Config::getInstance().max_people_1_batch;
        //This is to make sure threads fit all people in population
        n_threads = (batch_size < n_threads) ? batch_size : n_threads;
        for (int remain = population_->n_people_1_pop_current[p_index]; remain > 0; remain -= batch_size) {
            batch_size = (remain < batch_size) ? remain : batch_size;
            int batch_from = population_->n_people_1_pop_current[p_index] - remain;
            int batch_to = batch_from + batch_size;
//            printf("Pop %d work batch size %d remain %d, from %d to %d\n", p_index, batch_size, remain, batch_from, batch_to);
            buffer_person_.resize(batch_size);
            /* H2D */
            thrust::copy(population_->h_population[p_index].begin() + batch_from, population_->h_population[p_index].begin() + batch_to,buffer_person_.begin());
            checkCudaErr(cudaGetLastError());
            update_person_entity<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from,batch_to,batch_size, p_index, width,height,
                                                                                          thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                          GPURandom::getInstance().d_states);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
            /* D2H */
            thrust::copy(buffer_person_.begin(), buffer_person_.end(), population_->h_population[p_index].begin() + batch_from);
            checkCudaErr(cudaGetLastError());
        }
        buffer_person_.clear();
        thrust::device_vector<GPUPerson>().swap(buffer_person_);
    }
    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    if(Config::getInstance().test_debug) {
        printf("[GPUBuffer] Update population time: %d ms\n",std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
    }
}

__host__ void GPUBuffer::updateRender(){
    auto tp_start = std::chrono::high_resolution_clock::now();
    float width = (float)Config::getInstance().window_width;
    float height = (float)Config::getInstance().window_height;

    checkCudaErr(cudaMemcpy(population_->d_projection_mat, &(population_->h_projection_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(population_->d_view_mat, &(population_->h_view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaPeekAtLastError());

    int n_threads = 1024;
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        int batch_size = (population_->n_people_1_pop_current[p_index] < Config::getInstance().max_people_1_batch)
                         ? population_->n_people_1_pop_current[p_index] : Config::getInstance().max_people_1_batch;
        //This is to make sure threads fit all people in population
        n_threads = (population_->n_people_1_pop_current[p_index] < n_threads) ? population_->n_people_1_pop_current[p_index] : n_threads;
        for (int remain = population_->n_people_1_pop_current[p_index]; remain > 0; remain -= batch_size) {
            batch_size = (remain < batch_size) ? remain : batch_size;
            int batch_from = population_->n_people_1_pop_current[p_index] - remain;
            int batch_to = population_->n_people_1_pop_current[p_index] - remain + batch_size;
//            printf("Pop %d work batch size %d remain %d, from %d to %d\n", p_index, batch_size, remain, batch_from, batch_to);
            buffer_person_.resize(batch_size);
            /* H2D */
            thrust::copy(population_->h_population[p_index].begin() + batch_from, population_->h_population[p_index].begin() + batch_to,buffer_person_.begin());
            if(Config::getInstance().render_adaptive){
                buffer_person_render_flag_.resize(batch_size);
                calculate_render_entity<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_size,
                                                                                                   population_->d_projection_mat,
                                                                                                   population_->d_view_mat,
                                                                                                   thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                                   population_->d_base_person_pos[p_index],
                                                                                                   thrust::raw_pointer_cast(buffer_person_render_flag_.data()));
                thrust::copy(buffer_person_render_flag_.begin(), buffer_person_render_flag_.end(), population_->d_people_render_flags[p_index].begin() + batch_from);
                checkCudaErr(cudaDeviceSynchronize());
                checkCudaErr(cudaGetLastError());
            }
            //Update OGL buffer, this must be in final step
            update_ogl_buffer<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from,batch_to,batch_size,
                                                                                       thrust::raw_pointer_cast(buffer_person_.data()),
                                                                                       population_->d_ogl_buffer_model_ptr[p_index],
                                                                                       population_->d_ogl_buffer_color_ptr[p_index]);
            checkCudaErr(cudaDeviceSynchronize());
            checkCudaErr(cudaGetLastError());
        }
        if(Config::getInstance().render_adaptive){
            int N = population_->d_people_render_flags[p_index].size();
            population_->d_people_render_indices[p_index].resize(N);
            thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
                                                                       thrust::make_counting_iterator(N),
                                                                       population_->d_people_render_flags[p_index].begin(),
                                                                       population_->d_people_render_indices[p_index].begin(),
                                                                       thrust::placeholders::_1 == true);
            int size = end - population_->d_people_render_indices[p_index].begin();
            population_->d_people_render_indices[p_index].resize(size);
            population_->h_people_render_indices[p_index] = population_->d_people_render_indices[p_index];
        }
        buffer_person_.clear();
        thrust::device_vector<GPUPerson>().swap(buffer_person_);
        buffer_person_render_flag_.clear();
        thrust::device_vector<bool>().swap(buffer_person_render_flag_);
    }
    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    if(Config::getInstance().test_debug) {
        printf("[GPUBuffer] Update population render time: %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
    }
}

void GPUBuffer::start() {
    printf("[GPUBuffer] Thread started\n");
    while (true) {
        if(!Config::getInstance().is_window_rendered){
            break;
        }
        update();
        if(Config::getInstance().render_gui){
            updateRender();
        }
    }
    return;
}

void GPUBuffer::startThread() {
    buffer_thread_ = std::thread(&GPUBuffer::start, this);
    buffer_thread_.join();
}