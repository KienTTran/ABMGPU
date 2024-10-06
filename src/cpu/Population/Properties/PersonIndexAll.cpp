/* 
 * File:   PersonIndexAll.cpp
 * Author: nguyentran
 * 
 * Created on April 17, 2013, 10:15 AM
 */

#include <vector>
#include "PersonIndexAll.h"

PersonIndexAll::PersonIndexAll() = default;

PersonIndexAll::~PersonIndexAll() {
  h_persons_.clear();
  h_person_models_.clear();
  h_person_colors_.clear();
}

void PersonIndexAll::add(Person *p) {
  h_persons_.push_back(p);
  p->PersonIndexAllHandler::set_index(h_persons_.size() - 1);
  h_person_models_.push_back(p->model);
  h_person_colors_.push_back(p->color);
}

void PersonIndexAll::remove(Person *p) {

  h_person_models_[p->PersonIndexAllHandler::index()] = h_person_models_.back();
  h_person_models_.pop_back();

  h_person_colors_[p->PersonIndexAllHandler::index()] = h_person_colors_.back();
  h_person_colors_.pop_back();

  //move the last element to current position and remove the last holder
  h_persons_.back()->PersonIndexAllHandler::set_index(p->PersonIndexAllHandler::index());
  h_persons_[p->PersonIndexAllHandler::index()] = h_persons_.back();
  h_persons_.pop_back();

  p->PersonIndexAllHandler::set_index(-1);
  //    delete p;
  //    p = nullptr;
}

std::size_t PersonIndexAll::size() const {
  return h_persons_.size();
}

void PersonIndexAll::notifyChange(Person *p, const Person::Property &property, const void *oldValue,
                                   const void *newValue) {}

void PersonIndexAll::update() {
  h_persons_.shrink_to_fit();
  h_person_models_.shrink_to_fit();
  h_person_colors_.shrink_to_fit();
}
