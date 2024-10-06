//
// Created by kient on 6/16/2023.
//

#include <random>
#include <chrono>
#include "RenderEntity.cuh"
#include "../cpu/Model.h"
#include "../cpu/Population/Population.cuh"
#include "../cpu/Population/Properties/PersonIndexGPU.h"

RenderEntity::RenderEntity(Model* model) : model_(model) {
}

RenderEntity::~RenderEntity() {
}

void RenderEntity::initEntity(){
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist_vert(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_xy(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_xy1(-1.0, -1.0);
    std::uniform_real_distribution<float> dist_xy2(1.0, 1.0);
    std::uniform_real_distribution<float> dist_rot(0, 360);
    std::uniform_real_distribution<float> dist_color(0.0,1.0);

    entity_vertices = std::vector<glm::vec4>();
    entity_colors = std::vector<glm::vec4>();
    std::vector<glm::vec4> entity_vertices_1_pop;
    gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
    //Use first 2 index in glm::vec4 to define the vertex of a triangle in 2D. The last is always 1.0f
    float dist_x = dist_xy(gen);
    float dist_y = dist_xy(gen);
    //First triangle
    const float point_size = Model::CONFIG->render_config().point_coord;
    entity_vertices_1_pop.push_back(glm::vec4(-point_size+dist_x,-point_size+dist_y,0.0f,1.0f));//left bottom
    entity_vertices_1_pop.push_back(glm::vec4(point_size+dist_x,-point_size+dist_y,0.0f,1.0f));//right bottom
    entity_vertices_1_pop.push_back(glm::vec4(0.0f+dist_x,point_size+dist_y,0.0f,1.0f));//head
    //Second triangle
//        entity_vertices_1_pop.push_back(glm::vec4(-0.35f+dist_x,-0.35f+dist_y,0.0f,1.0f));//left bottom
//        entity_vertices_1_pop.push_back(glm::vec4(0.35f+dist_x,-0.35f+dist_y,0.0f,1.0f));//right bottom
//        entity_vertices_1_pop.push_back(glm::vec4(0.0f+dist_x,-0.35f+dist_y,0.0f,1.0f));//head
    entity_vertices = entity_vertices_1_pop;

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
    entity_colors = entity_colors_1_pop;
    checkCudaErr(cudaMalloc(&d_ogl_buffer_model_ptr, Model::CONFIG->n_people_init() * Model::CONFIG->gpu_config().pre_allocated_mem_ratio * sizeof(glm::mat4)));
    checkCudaErr(cudaMalloc(&d_ogl_buffer_color_ptr, Model::CONFIG->n_people_init() * Model::CONFIG->gpu_config().pre_allocated_mem_ratio * sizeof(glm::vec4)));
}

void RenderEntity::initRender(int window_width, int window_height) {
    view = glm::mat4(1.0f);
    projection = glm::ortho(0.0f,
                            (float)window_width,
                            0.0f,
                            (float)window_height,
                            -1.0f, 1.0f);

    shader = new Shader();
    VBO = new GLuint[2];
    SSBO = new GLuint[2];
    shader->loadData("../shaders/shader.vert", "../shaders/shader.frag");
    shader->use();
    int viewLoc = glGetUniformLocation(shader->ID, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    int projectionLoc = glGetUniformLocation(shader->ID, "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    shader->setMat4("view", view);
    shader->setMat4("projection", projection);

    glCreateBuffers(1, &CMD);
    glNamedBufferData(CMD, sizeof(DrawElementsIndirectCommand), &CMD, GL_STATIC_DRAW);

    //VAO
    //glBindVertexArray must be before glVertexAttribPointer
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    //VBO - Position
    glGenBuffers(1, &VBO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, 3*sizeof(glm::vec4), entity_vertices.data(), GL_DYNAMIC_DRAW);

    //VBO - color
    glGenBuffers(1, &VBO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, 3*sizeof(glm::vec4), entity_colors.data(), GL_DYNAMIC_DRAW);

    //EBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(entity_indices), entity_indices, GL_DYNAMIC_DRAW);

    auto *pi = Model::POPULATION->getPersonIndex<PersonIndexGPU>();
    //SSBO[0]
    //instance data - buffer_object.models, use either this one or matrixVBO
    glGenBuffers(1, &SSBO[0]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, pi->h_person_models().size() * sizeof(glm::mat4),
                 pi->h_person_models().data(),GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, SSBO[0]);//binding 2 in shader.vert

    //SSBO[1]
    //instance data - buffer_object.models, use either this one or matrixVBO
    glGenBuffers(1, &SSBO[1]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, pi->h_person_colors().size() * sizeof(glm::vec4),
                 pi->h_person_colors().data(),GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SSBO[1]);//binding 3 in shader.vert

    // map OpenGL buffer object for writing from CUDA

    checkCudaErr(cudaGraphicsGLRegisterBuffer(&d_cuda_buffer_model, SSBO[0], cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErr(cudaGraphicsGLRegisterBuffer(&d_cuda_buffer_color, SSBO[1], cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErr(cudaGraphicsMapResources(1, &d_cuda_buffer_model, 0));
    checkCudaErr(cudaGraphicsMapResources(1, &d_cuda_buffer_color, 0));
    checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&d_ogl_buffer_model_ptr,
                                                      &d_ogl_buffer_model_num_bytes,
                                                      d_cuda_buffer_model));
    checkCudaErr(cudaGraphicsResourceGetMappedPointer((void **)&d_ogl_buffer_color_ptr,
                                                      &d_ogl_buffer_color_num_bytes,
                                                      d_cuda_buffer_color));


    //Set data of shader, must be done after bind all buffers
    //aPos
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void *) 0);//glsl vec4 pos -> 4
    glEnableVertexAttribArray(0);
    //aColor
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void *) 0);//glsl vec4 color -> 4
    glEnableVertexAttribArray(1);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindVertexArray(0);
}
