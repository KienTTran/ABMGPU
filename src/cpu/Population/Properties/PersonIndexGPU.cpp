//
// Created by kient on 12/9/2023.
//

#include "../../Model.h"
#include "PersonIndexGPU.h"

PersonIndexGPU::PersonIndexGPU() {
    h_person_models_ = thrust::host_vector<glm::mat4>(Model::CONFIG->n_people_init()*Model::CONFIG->gpu_config().pre_allocated_mem_ratio);
    h_person_colors_ = thrust::host_vector<glm::vec4>(Model::CONFIG->n_people_init()*Model::CONFIG->gpu_config().pre_allocated_mem_ratio);
}

PersonIndexGPU::~PersonIndexGPU() {
    h_persons_.clear();
    h_person_models_.clear();
    h_person_colors_.clear();
}

void PersonIndexGPU::add(Person *p) {
    h_persons_.push_back(p);
    p->PersonIndexToRenderHandler::set_index(h_persons_.size() - 1);
    h_person_models_[p->PersonIndexToRenderHandler::index()] = p->model;
    h_person_colors_[p->PersonIndexToRenderHandler::index()] = p->color;
//    h_person_parasites_.push_back(p->parasites);
}

void PersonIndexGPU::remove(Person *p) {

    h_persons_.back()->PersonIndexToRenderHandler::set_index(p->PersonIndexToRenderHandler::index());

    h_person_models_[p->PersonIndexToRenderHandler::index()] = h_person_models_.back();
//    h_person_models_.pop_back();

    h_person_colors_[p->PersonIndexToRenderHandler::index()] = h_person_colors_.back();
//    h_person_colors_.pop_back();

    //move the last element to current position and remove the last holder
    h_persons_[p->PersonIndexToRenderHandler::index()] = h_persons_.back();
    h_persons_.pop_back();

    p->PersonIndexToRenderHandler::set_index(-1);
    //    delete p;
    //    p = nullptr;
}

std::size_t PersonIndexGPU::size() const {
    return h_persons_.size();
}

void PersonIndexGPU::notifyChange(Person *p, const Person::Property &property, const void *oldValue,
                                  const void *newValue) {}

void PersonIndexGPU::update() {
    h_persons_.shrink_to_fit();
    h_person_models_.shrink_to_fit();
    h_person_colors_.shrink_to_fit();
}
