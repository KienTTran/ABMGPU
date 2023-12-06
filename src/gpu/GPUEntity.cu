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

void GPUEntity::initPopEntity(Population* population){
    population_ = population;
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist_vert(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_xy(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_xy1(-1.0, -1.0);
    std::uniform_real_distribution<float> dist_xy2(1.0, 1.0);
    std::uniform_real_distribution<float> dist_rot(0, 360);
    std::uniform_real_distribution<float> dist_color(0.0,1.0);

    entity_vertices = std::vector<std::vector<glm::vec4>>(Config::getInstance().n_pops);
    entity_colors = std::vector<std::vector<glm::vec4>>(Config::getInstance().n_pops);
    for(int p_index = 0; p_index < Config::getInstance().n_pops; p_index++) {
        entity_vertices[p_index] = std::vector<glm::vec4>();
        entity_colors[p_index] = std::vector<glm::vec4>();
        std::vector<glm::vec4> entity_vertices_1_pop;
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //Use first 2 index in glm::vec4 to define the vertex of a triangle in 2D. The last is always 1.0f
        float dist_x = dist_xy(gen);
        float dist_y = dist_xy(gen);
        //First triangle
        const float point_size = Config::getInstance().render_point_coord;
        entity_vertices_1_pop.push_back(glm::vec4(-point_size+dist_x,-point_size+dist_y,0.0f,1.0f));//left bottom
        entity_vertices_1_pop.push_back(glm::vec4(point_size+dist_x,-point_size+dist_y,0.0f,1.0f));//right bottom
        entity_vertices_1_pop.push_back(glm::vec4(0.0f+dist_x,point_size+dist_y,0.0f,1.0f));//head
        //Second triangle
//        entity_vertices_1_pop.push_back(glm::vec4(-0.35f+dist_x,-0.35f+dist_y,0.0f,1.0f));//left bottom
//        entity_vertices_1_pop.push_back(glm::vec4(0.35f+dist_x,-0.35f+dist_y,0.0f,1.0f));//right bottom
//        entity_vertices_1_pop.push_back(glm::vec4(0.0f+dist_x,-0.35f+dist_y,0.0f,1.0f));//head
        entity_vertices[p_index] = entity_vertices_1_pop;

        //Copy base person coord to compute current coord in GPU, any vertex should work, can use centroid for more precision
        population_->h_base_person_pos[p_index] = glm::vec4(0.0f+dist_x,point_size+dist_y,0.0f,1.0f);
        population_->d_base_person_pos[p_index] = population_->h_base_person_pos[p_index];

        std::vector<glm::vec4> entity_colors_1_pop;
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
        glm::vec4 entity_pop_color = glm::vec4(dist_color(gen),dist_color(gen),dist_color(gen),1.0f);
        //Use first 3 index in glm::vec4 to define the color of triangle. The last is alpha.
        //Push 3 times because it's a triangle, we want all vertices same color.
        //First triangle
        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors_1_pop.push_back(entity_pop_color);//head
        //Second triangle
//        entity_colors_1_pop.push_back(entity_pop_color);//head
//        entity_colors_1_pop.push_back(entity_pop_color);//head
//        entity_colors_1_pop.push_back(entity_pop_color);//head
        entity_colors[p_index] = entity_colors_1_pop;
    }
}

void GPUEntity::initRender(int window_width, int window_height) {
    view = glm::mat4(1.0f);
    projection = glm::ortho(0.0f,
                            (float)window_width,
                            0.0f,
                            (float)window_height,
                            -1.0f, 1.0f);
    population_->h_projection_mat = projection;
    population_->h_view_mat = view;
    checkCudaErr(cudaMemcpy(population_->d_projection_mat, &(population_->h_projection_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(population_->d_view_mat, &(population_->h_view_mat), sizeof(glm::mat4), cudaMemcpyHostToDevice));

    VAO = new GLuint[Config::getInstance().n_pops];
    VBO = new GLuint*[Config::getInstance().n_pops];
    EBO = new GLuint[Config::getInstance().n_pops];
    SSBO = new GLuint*[Config::getInstance().n_pops];
    CMD = new GLuint[Config::getInstance().n_pops];
    shader = new Shader*[Config::getInstance().n_pops];
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

        glCreateBuffers(1, &CMD[p_index]);
        glNamedBufferData(CMD[p_index], sizeof(DrawElementsIndirectCommand), &CMD[p_index], GL_STATIC_DRAW);

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
        glBufferData(GL_SHADER_STORAGE_BUFFER, population_->h_people_models[p_index].size() * sizeof(glm::mat4),
                     population_->h_people_models[p_index].data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, SSBO[p_index][0]);//binding 2 in shader.vert

        //SSBO2
        //instance data - buffer_object.models, use either this one or matrixVBO
        glGenBuffers(1, &SSBO[p_index][1]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[p_index][1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, population_->h_people_colors[p_index].size() * sizeof(glm::vec4),
                     population_->h_people_colors[p_index].data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SSBO[p_index][1]);//binding 3 in shader.vert

        // map OpenGL buffer object for writing from CUDA
        checkCudaErr(cudaGraphicsGLRegisterBuffer(&population_->d_cuda_buffer_model[p_index], SSBO[p_index][0], cudaGraphicsMapFlagsWriteDiscard));
        checkCudaErr(cudaGraphicsGLRegisterBuffer(&population_->d_cuda_buffer_color[p_index], SSBO[p_index][1], cudaGraphicsMapFlagsWriteDiscard));
        checkCudaErr(cudaGraphicsMapResources(1, &population_->d_cuda_buffer_model[p_index], 0));
        checkCudaErr(cudaGraphicsMapResources(1, &population_->d_cuda_buffer_color[p_index], 0));
        checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&population_->d_ogl_buffer_model_ptr[p_index],
                                                          &population_->d_ogl_buffer_model_num_bytes[p_index],
                                                          population_->d_cuda_buffer_model[p_index]));
        checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&population_->d_ogl_buffer_color_ptr[p_index],
                                                          &population_->d_ogl_buffer_color_num_bytes[p_index],
                                                          population_->d_cuda_buffer_color[p_index]));

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
    checkCudaErr(cudaGraphicsUnmapResources(1, &population_->d_cuda_buffer_model[p_index], 0));
    checkCudaErr(cudaGraphicsUnmapResources(1, &population_->d_cuda_buffer_color[p_index], 0));
}
