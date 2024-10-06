/* 
 * File:   PersonIndexByLocationStateAgeClass.cpp
 * Author: nguyentran
 * 
 * Created on May 2, 2013, 10:41 AM
 */

#include "PersonIndexByLocationStateAgeClass.h"
#include "PersonIndexByLocationStateAgeClassHandler.h"
#include "../../Core/Config/Config.h"
#include "../../Model.h"

#include <cassert>

PersonIndexByLocationStateAgeClass::PersonIndexByLocationStateAgeClass(const int &no_location, const int &no_host_state,
                                                                       const int &no_age_class) {
  Initialize(no_location, no_host_state, no_age_class);
}

PersonIndexByLocationStateAgeClass::~PersonIndexByLocationStateAgeClass() {

}

void PersonIndexByLocationStateAgeClass::Initialize(const int &no_location, const int &no_host_state,
                                                    const int &no_age_class) {
  h_persons_.clear();

  ThrustPersonPtrVectorHost ppv;
  ThrustPersonPtrVectorHost2 ppv2;
  ppv2.assign(no_age_class, ppv);
  ThrustPersonPtrVectorHost3 ppv3;
  ppv3.assign(no_host_state, ppv2);
  h_persons_.assign(no_location, ppv3);

  h_person_models_.clear();
  ThrustGLMat4VectorHost pmv;
  ThrustGLMat4VectorHost2 pmv2;
  pmv2.assign(no_age_class, pmv);
  ThrustGLMat4VectorHost3 pmv3;
  pmv3.assign(no_host_state, pmv2);
  h_person_models_.assign(no_location, pmv3);

  h_person_colors_.clear();
  ThrustGLVec4VectorHost pcv;
  ThrustGLVec4VectorHost2 pcv2;
  pcv2.assign(no_age_class, pcv);
  ThrustGLVec4VectorHost3 pcv3;
  pcv3.assign(no_host_state, pcv2);
  h_person_colors_.assign(no_location, pcv3);
}

void PersonIndexByLocationStateAgeClass::add(Person *p) {
  assert(h_persons_.size() > p->location && p->location >= 0);
  assert(h_persons_[p->location].size() > p->host_state());
  assert(h_persons_[p->location][p->host_state()].size() > p->age_class());
  assert(p->age_class() >= 0);

  add(p, p->location, p->host_state(), p->age_class());

}

void PersonIndexByLocationStateAgeClass::add(Person *p, const int &location, const Person::HostStates &host_state,
                                             const int &age_class) {
  h_persons_[location][host_state][age_class].push_back(p);
  h_person_models_[location][host_state][age_class].push_back(p->model);
  h_person_colors_[location][host_state][age_class].push_back(p->color);
  p->PersonIndexByLocationStateAgeClassHandler::set_index(h_persons_[location][host_state][age_class].size() - 1);
}


void PersonIndexByLocationStateAgeClass::remove(Person *p) {
  remove_without_set_index(p);
  p->PersonIndexByLocationStateAgeClassHandler::set_index(-1);
}

void PersonIndexByLocationStateAgeClass::remove_without_set_index(Person *p) {

  h_persons_[p->location][p->host_state()][p->age_class()].back()->PersonIndexByLocationStateAgeClassHandler::set_index(
          p->PersonIndexByLocationStateAgeClassHandler::index());

  h_person_models_[p->location][p->host_state()][p->age_class()][p->PersonIndexByLocationStateAgeClassHandler::index()] =
          h_person_models_[p->location][p->host_state()][p->age_class()].back();
  h_person_models_[p->location][p->host_state()][p->age_class()].pop_back();

  h_person_colors_[p->location][p->host_state()][p->age_class()][p->PersonIndexByLocationStateAgeClassHandler::index()] =
          h_person_colors_[p->location][p->host_state()][p->age_class()].back();
  h_person_colors_[p->location][p->host_state()][p->age_class()].pop_back();

  h_persons_[p->location][p->host_state()][p->age_class()][p->PersonIndexByLocationStateAgeClassHandler::index()] =
          h_persons_[p->location][p->host_state()][p->age_class()].back();
  h_persons_[p->location][p->host_state()][p->age_class()].pop_back();

}

std::size_t PersonIndexByLocationStateAgeClass::size() const {
  return 0;
}

void
PersonIndexByLocationStateAgeClass::notifyChange(Person *p, const Person::Property &property, const void *oldValue,
                                                  const void *newValue) {

  switch (property) {
    case Person::LOCATION:change_property(p, *(int *) newValue, p->host_state(), p->age_class());
      break;
    case Person::HOST_STATE:change_property(p, p->location, *(Person::HostStates *) newValue, p->age_class());
      break;
    case Person::AGE_CLASS:change_property(p, p->location, p->host_state(), *(int *) newValue);
      break;
    default:break;
  }

}

void PersonIndexByLocationStateAgeClass::change_property(Person *p, const int &location,
                                                         const Person::HostStates &host_state, const int &age_class) {
  //remove from old position
  remove_without_set_index(p); //to save 1 set and improve performance since the index of p will changed when add

  //add to new position
  add(p, location, host_state, age_class);
}

void PersonIndexByLocationStateAgeClass::update() {
  for (int location = 0; location < Model::CONFIG->number_of_locations(); location++) {
    for (int hs = 0; hs < Person::NUMBER_OF_STATE; hs++) {
      for (int ac = 0; ac < Model::CONFIG->number_of_age_classes(); ac++) {
        thrust::host_vector<Person *>(h_persons_[location][hs][ac]).swap(h_persons_[location][hs][ac]);
        thrust::host_vector<glm::mat4>(h_person_models_[location][hs][ac]).swap(h_person_models_[location][hs][ac]);
        thrust::host_vector<glm::vec4>(h_person_colors_[location][hs][ac]).swap(h_person_colors_[location][hs][ac]);
      }
    }
  }

}
