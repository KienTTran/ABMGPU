//
// Created by kient on 6/17/2023.
//

#ifndef MASS_POPULATION_CUH
#define MASS_POPULATION_CUH

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include  "../Core/Dispatcher.h"
#include "Person.h"
#include "../../utils/GPURandom.cuh"
#include "../../gpu/RenderEntity.cuh"
#include "../../utils/GPUUtils.cuh"
#include <thread>
#include "../Helpers/ObjectHelpers.h"
#include "Properties/PersonIndex.h"

class Person;
class Model;

class Population : public Dispatcher{
public:

    POINTER_PROPERTY(PersonIndexPtrList, person_index_list);

    Model* model_;

    //on HOST
    thrust::host_vector<glm::vec4> h_cell_colors; // for init cell color on device

    //on DEVICE CUDA
    thrust::host_vector<Person*> buffer_person_;
    thrust::device_vector<glm::mat4> buffer_person_models_;
    thrust::device_vector<glm::vec4> buffer_person_colors_;

    struct cudaGraphicsResource *d_cuda_buffer_model_render;
    size_t d_ogl_buffer_model_num_bytes_render; // to get models data from gpu_buffer
    struct cudaGraphicsResource *d_cuda_buffer_color_render;
    size_t d_ogl_buffer_color_num_bytes_render;// to get colors data from gpu_buffer

    int number_of_locations;
    int number_of_init_age_classes;
    int number_of_age_classes;

public:
    Population(Model *model = nullptr);
    ~Population();
    void init();
    void initPop();
    void updateRender();
    void update();
    void performBirthEvent();
    void performDeathEvent();
    virtual void notifyChange(Person *p, const Person::Property &property, const void *oldValue, const void *newValue);
    void addPerson(Person* person);
    void removeAllDeadPerson();
    void removeDeadPerson(Person* person);
    void removePerson(Person* person);
    void start();

    template <typename T>
    T *getPersonIndex();
};

template <typename T>
T *Population::getPersonIndex() {
    for (PersonIndex *person_index : *person_index_list_) {
        if (dynamic_cast<T *>(person_index) != nullptr) {
            T *pi = dynamic_cast<T *>(person_index);
            return pi;
        }
    }
    return nullptr;
}

#endif //MASS_POPULATION_CUH
