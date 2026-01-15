/*
 * File:   TypeDef.h
 * Author: nguyentran
 *
 * Created on April 17, 2013, 10:17 AM
 */

#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <array>
#include <list>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <glm/vec4.hpp>
#include <GL/glew.h>
#include "../Events/Event.h"
#include "../../cpu/Population/Person.h"
#include "../Population/Properties/PersonIndex.h"

typedef unsigned long ul;

typedef std::vector<double> DoubleVector;
typedef std::vector<DoubleVector> DoubleVector2;
typedef std::vector<DoubleVector2> DoubleVector3;
typedef std::vector<int> IntVector;
typedef std::vector<int>* IntVectorPtr;
typedef std::vector<IntVector> IntVector2;
typedef std::vector<IntVector2> IntVector3;
typedef std::vector<IntVector*> IntVectorPtrVector;
typedef std::vector<IntVector>* IntVector2Ptr;
typedef std::vector<unsigned int> UIntVector;

typedef std::vector<ul> LongVector;
typedef std::vector<LongVector> LongVector2;

typedef std::vector<std::string> StringVector;
typedef std::vector<StringVector> StringVector2;

typedef std::vector<Event*> EventPtrVector;
typedef std::vector<EventPtrVector> EventPtrVector2;

typedef thrust::tuple<int,int,int,int> ThrustInt4Tuple;

typedef thrust::host_vector<Person*> ThrustPersonPtrVectorHost;
typedef thrust::host_vector<ThrustPersonPtrVectorHost> ThrustPersonPtrVectorHost2;
typedef thrust::host_vector<ThrustPersonPtrVectorHost2> ThrustPersonPtrVectorHost3;
typedef thrust::host_vector<ThrustPersonPtrVectorHost3> ThrustPersonPtrVectorHost4;

typedef thrust::host_vector<GLuint> ThrustGLuintVectorHost;
typedef thrust::host_vector<ThrustGLuintVectorHost> ThrustGLuintVectorHost2;
typedef thrust::host_vector<ThrustGLuintVectorHost2> ThrustGLuintVectorHost3;
typedef thrust::host_vector<ThrustGLuintVectorHost3> ThrustGLuintVectorHost4;
typedef thrust::host_vector<ThrustGLuintVectorHost4> ThrustGLuintVectorHost5;

typedef thrust::host_vector<glm::mat4> ThrustGLMat4VectorHost;
typedef thrust::host_vector<ThrustGLMat4VectorHost> ThrustGLMat4VectorHost2;
typedef thrust::host_vector<ThrustGLMat4VectorHost2> ThrustGLMat4VectorHost3;
typedef thrust::host_vector<ThrustGLMat4VectorHost3> ThrustGLMat4VectorHost4;
typedef thrust::host_vector<ThrustGLMat4VectorHost4> ThrustGLMat4VectorHost5;

typedef thrust::host_vector<glm::vec4> ThrustGLVec4VectorHost;
typedef thrust::host_vector<ThrustGLVec4VectorHost> ThrustGLVec4VectorHost2;
typedef thrust::host_vector<ThrustGLVec4VectorHost2> ThrustGLVec4VectorHost3;
typedef thrust::host_vector<ThrustGLVec4VectorHost3> ThrustGLVec4VectorHost4;
typedef thrust::host_vector<ThrustGLVec4VectorHost4> ThrustGLVec4VectorHost5;

typedef std::list<PersonIndex *> PersonIndexPtrList;

struct GPUConfig{
    int n_threads;
    int people_1_batch;
    double pre_allocated_mem_ratio;
    double population_scale;
    friend std::ostream &operator<<(std::ostream &os, const GPUConfig &mcf) {
        os << "gpu_config";
        return os;
    }
};

struct RenderConfig{
    int window_width;
    int window_height;
    bool display_gui;
    bool close_window_on_finish;
    double point_coord;
    int people_per_triangle = 1;
    float zoom_step = 2.0f;
    friend std::ostream &operator<<(std::ostream &os, const RenderConfig &mcf) {
        os << "render_config";
        return os;
    }
};

struct DebugConfig{
    int width;
    int height;
    bool enable_update;
    bool enable_debug_text;
    bool enable_debug_render;
    bool enable_debug_render_text;
    friend std::ostream &operator<<(std::ostream &os, const DebugConfig &mcf) {
        os << "debug_config";
        return os;
    }
};

struct RasterDb {
    friend std::ostream &operator<<(std::ostream &os, const RasterDb &rdb) {
        os << "raster_db";
        return os;
    }
};

struct RelativeMovingInformation {
    double max_relative_moving_value;
    int number_of_moving_levels;

    //  biting_level_distribution:
    //  #  distribution: Exponential
    //    distribution: Gamma
    //    Exponential:
    double scale;

    double mean;
    double sd;
    DoubleVector v_moving_level_value;
    DoubleVector v_moving_level_density;

    double circulation_percent;
    double length_of_stay_mean;
    double length_of_stay_sd;
    double length_of_stay_theta;
    double length_of_stay_k;

};
#endif /* TYPEDEF_H */
