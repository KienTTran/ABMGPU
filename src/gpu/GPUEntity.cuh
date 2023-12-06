//
// Created by kient on 6/16/2023.
//

#ifndef ABMGPU_GPUENTITY_CUH
#define ABMGPU_GPUENTITY_CUH

#include "GL/glew.h"
#include <cuda_gl_interop.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/sequence.h>
#include "../utils/Shader.h"
#include "../gpu/Population.cuh"
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>

typedef struct {
    GLuint count;
    GLuint primCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
} DrawElementsIndirectCommand;

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
    glm::mat4 projection;
    glm::mat4 view;
    GLint entity_indices[3] = {0,1,2};
//    GLint entity_indices[6] = {0,1,2,0,1,3};
    GLuint **VBO;
    GLuint *EBO;
    GLuint **SSBO;
    GLuint *CMD;
};


#endif //ABMGPU_GPUENTITY_CUH
