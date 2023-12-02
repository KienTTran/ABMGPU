//
// Created by kient on 6/16/2023.
//

#ifndef MASS_GPUENTITY_CUH
#define MASS_GPUENTITY_CUH

#include "GL/glew.h"
#include <cuda_gl_interop.h>
#include "../utils/Shader.h"
#include "../gpu/Population.cuh"
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>

typedef  struct {
    int  count;
    int  instanceCount;
    int  first;
    int  baseInstance;
} DrawArraysIndirectCommand;

class GPUEntity {
public:
    GPUEntity();

    ~GPUEntity();

    void initPopEntity(Population* population);
    void initRender(int window_width, int window_height);
    void update(int p_index);

public:
    GLuint *VAO;
    Shader **shader;

    Population* population_;
    std::vector<std::vector<glm::vec4>> entity_vertices;
    std::vector<std::vector<glm::vec4>> entity_colors;
    GLint entity_indices[3] = {0,1,2};
    GLuint **VBO;
    GLuint *EBO;
    GLuint **SSBO;
    int window_width;
    int window_height;

    struct cudaGraphicsResource **cuda_buffer_model;
    size_t *ogl_buffer_model_num_bytes; // to get models data from gpu_buffer
    glm::mat4 **ogl_buffer_model_ptr;

    struct cudaGraphicsResource **cuda_buffer_color;
    size_t *ogl_buffer_color_num_bytes;// to get colors data from gpu_buffer
    glm::vec4 **ogl_buffer_color_ptr;


    int n_threads = 1024;
    int *n_blocks;
};


#endif //MASS_GPUENTITY_CUH
