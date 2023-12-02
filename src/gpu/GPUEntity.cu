//
// Created by kient on 6/16/2023.
//

#include <random>
#include <chrono>
#include "GPUEntity.cuh"
#include "../utils/GPUUtils.cuh"
#include "../utils/GPURandom.cuh"

GPUEntity::GPUEntity() {
}

GPUEntity::~GPUEntity() {
}

__global__ void update_person_entity(glm::mat4 *models, glm::vec4 *colors, unsigned int width, unsigned int height, uint64_t n_instances, float time, curandState *state)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Copy state
    curandState localState = state[thread_index];
    for(uint64_t index = thread_index; index < n_instances; index += stride) {
        glm::mat4 model = models[index];
        models[index] = model;
        glm::vec4 color = colors[index];
        colors[index] = color;
        __syncthreads();
    }
    state[thread_index] = localState;
}

void GPUEntity::initPopEntity(Population* population){
    population_ = population;
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist_vert(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_xy(-10.0, 10.0);
    std::uniform_real_distribution<float> dist_xy1(-1.0, -1.0);
    std::uniform_real_distribution<float> dist_xy2(1.0, 1.0);
    std::uniform_real_distribution<float> dist_rot(0, 360);
    std::uniform_real_distribution<float> dist_color(0.0,1.0);

    entity_vertices = std::vector<std::vector<glm::vec4>>(Config::getInstance().n_pops);
    entity_colors = std::vector<std::vector<glm::vec4>>(Config::getInstance().n_pops);

    glm::mat4 model = glm::mat4(1.0f);
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        entity_vertices[p_index] = std::vector<glm::vec4>();
        entity_colors[p_index] = std::vector<glm::vec4>();
        std::vector<glm::vec4> entity_vertices_1_pop;
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //Use first 2 index in glm::vec4 to define the vertex of a triangle in 2D. The last is always 1.0f
        float dist_x = dist_xy(gen);
        float dist_y = dist_xy(gen);
        entity_vertices_1_pop.push_back(glm::vec4(-0.5f+dist_x,-0.5f+dist_y,0.0f,1.0f));//left bottom
        entity_vertices_1_pop.push_back(glm::vec4(0.5f+dist_x,-0.5f+dist_y,0.0f,1.0f));//right bottom
        entity_vertices_1_pop.push_back(glm::vec4(0.0f+dist_x,0.5f+dist_y,0.0f,1.0f));//head
        entity_vertices[p_index] = entity_vertices_1_pop;

        std::vector<glm::vec4> entity_colors_1_pop;
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        glm::vec4 entity_pop_color = glm::vec4(dist_color(gen),dist_color(gen),dist_color(gen),1.0f);
        //Use first 3 index in glm::vec4 to define the color of triangle. The last is alpha.
        //Push 3 times because it's a triangle, we want all vertices same color.
        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors[p_index] = entity_colors_1_pop;
    }
}

void GPUEntity::initRender(int window_width, int window_height) {
    this->window_width = window_width;
    this->window_height = window_height;

    VAO = new GLuint[Config::getInstance().n_pops];
    VBO = new GLuint*[Config::getInstance().n_pops];
    EBO = new GLuint[Config::getInstance().n_pops];
    SSBO = new GLuint*[Config::getInstance().n_pops];
    shader = new Shader*[Config::getInstance().n_pops];
    ogl_buffer_model_ptr = new glm::mat4*[Config::getInstance().n_pops];
    ogl_buffer_color_ptr = new glm::vec4*[Config::getInstance().n_pops];
    cuda_buffer_model = new cudaGraphicsResource*[Config::getInstance().n_pops];
    ogl_buffer_model_num_bytes = new size_t[Config::getInstance().n_pops];
    cuda_buffer_color = new cudaGraphicsResource*[Config::getInstance().n_pops];
    ogl_buffer_color_num_bytes = new size_t[Config::getInstance().n_pops];
    n_blocks = new int[Config::getInstance().n_pops];

    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::ortho(((float) this->window_width / 2.0f) - ((float) this->window_width / 2.0f),
                                      ((float) this->window_width / 2.0f) + ((float) this->window_width / 2.0f),
                                      ((float) this->window_height / 2.0f) - ((float) this->window_height / 2.0f),
                                      ((float) this->window_height / 2.0f) + ((float) this->window_height / 2.0f),
                                      -1.0f, 1.0f);

    for(int i = 0; i < Config::getInstance().n_pops; i++) {
        printf("[Entity] init pop %d last person id: %lld x:%f y: %f color: (%f %f %f)\n", i,
               population_->h_population[i][population_->h_population[i].size() - 1].id,
               population_->h_people_models[i][population_->h_people_models[i].size() - 1][0][0],
               population_->h_people_models[i][population_->h_people_models[i].size() - 1][0][1],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][0],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][1],
               population_->h_people_colors[i][population_->h_people_colors[i].size() - 1][2]);
    }

    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        shader[p_index] = new Shader();
        VBO[p_index] = new GLuint[2];
        SSBO[p_index] = new GLuint[2];

        shader[p_index]->loadData("../src/shaders/shader.vert", "../src/shaders/shader.frag");
        shader[p_index]->use();
        int viewLoc = glGetUniformLocation(shader[p_index]->ID, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        int projectionLoc = glGetUniformLocation(shader[p_index]->ID, "projection");
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
        shader[p_index]->setMat4("view", view);
        shader[p_index]->setMat4("projection", projection);

        //VAO
        //glBindVertexArray must be before glVertexAttribPointer
        glGenVertexArrays(1, &VAO[p_index]);
        glBindVertexArray(VAO[p_index]);

        //VBO - Position
        glGenBuffers(1, &VBO[p_index][0]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[p_index][0]);
        glBufferData(GL_ARRAY_BUFFER, 3*sizeof(glm::vec4), entity_vertices[p_index].data(), GL_DYNAMIC_DRAW);

        //VBO - color
        glGenBuffers(1, &VBO[p_index][1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[p_index][1]);
        glBufferData(GL_ARRAY_BUFFER, 3*sizeof(glm::vec4), entity_colors[p_index].data(), GL_DYNAMIC_DRAW);

        //EBO
        glGenBuffers(1, &EBO[p_index]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[p_index]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(entity_indices), entity_indices, GL_DYNAMIC_DRAW);

        //SSBO
        //instance data - buffer_object.models, use either this one or matrixVBO
        glGenBuffers(1, &SSBO[p_index][0]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[p_index][0]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, population_->h_people_models[p_index].size() * sizeof(glm::mat4),population_->h_people_models[p_index].data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, SSBO[p_index][0]);//binding 2 in shader.vert

        //SSBO2
        //instance data - buffer_object.models, use either this one or matrixVBO
        glGenBuffers(1, &SSBO[p_index][1]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[p_index][1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, population_->h_people_colors[p_index].size() * sizeof(glm::vec4), population_->h_people_colors[p_index].data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SSBO[p_index][1]);//binding 3 in shader.vert

        // map OpenGL buffer object for writing from CUDA
        checkCudaErr(cudaGraphicsGLRegisterBuffer(&cuda_buffer_model[p_index], SSBO[p_index][0], cudaGraphicsMapFlagsNone));
        checkCudaErr(cudaGraphicsMapResources(1, &cuda_buffer_model[p_index], 0));
        checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&ogl_buffer_model_ptr[p_index], &ogl_buffer_model_num_bytes[p_index],cuda_buffer_model[p_index]));
        checkCudaErr(cudaGraphicsGLRegisterBuffer(&cuda_buffer_color[p_index], SSBO[p_index][1], cudaGraphicsMapFlagsNone));
        checkCudaErr(cudaGraphicsMapResources(1, &cuda_buffer_color[p_index], 0));
        checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&ogl_buffer_color_ptr[p_index], &ogl_buffer_color_num_bytes[p_index],cuda_buffer_color[p_index]));

        //Set data of shader, must be done after bind all buffers
        //aPos
        glBindBuffer(GL_ARRAY_BUFFER, VBO[p_index][0]);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void *) 0);//glsl vec4 pos -> 4
        glEnableVertexAttribArray(0);
        //aColor
        glBindBuffer(GL_ARRAY_BUFFER, VBO[p_index][1]);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void *) 0);//glsl vec4 color -> 4
        glEnableVertexAttribArray(1);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glBindVertexArray(0);
    }
}

void GPUEntity::update(int p_index) {
    //This is to make sure threads fit all people in population
    n_threads = (population_->h_population[p_index].size() < n_threads) ? population_->h_population[p_index].size() : n_threads;
    n_blocks[p_index] = (population_->h_population[p_index].size() + n_threads - 1) / n_threads;
    //Copy from CUDA to OGL, since they are allocated in different memory space
    checkCudaErr(cudaMemcpy(ogl_buffer_model_ptr[p_index], thrust::raw_pointer_cast(population_->d_people_models[p_index].data()),
                            population_->d_people_models[p_index].size() * sizeof(glm::mat4), cudaMemcpyDeviceToDevice));
    checkCudaErr(cudaMemcpy(ogl_buffer_color_ptr[p_index], thrust::raw_pointer_cast(population_->d_people_colors[p_index].data()),
                            population_->d_people_models[p_index].size() * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));

    update_person_entity<<<n_blocks[p_index], n_threads>>>(ogl_buffer_model_ptr[p_index],
                                                           ogl_buffer_color_ptr[p_index],
                                                           this->window_width, this->window_height,
                                                           population_->h_population[p_index].size(),0.0f, GPURandom::getInstance().d_states);

    //Copy back from OGL to CUDA
    checkCudaErr(cudaMemcpy(thrust::raw_pointer_cast(population_->d_people_models[p_index].data()), ogl_buffer_model_ptr[p_index],
                            population_->d_people_models[p_index].size() * sizeof(glm::mat4), cudaMemcpyDeviceToDevice));
    checkCudaErr(cudaMemcpy(thrust::raw_pointer_cast(population_->d_people_colors[p_index].data()), ogl_buffer_color_ptr[p_index],
                            population_->d_people_models[p_index].size() * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaPeekAtLastError());
}