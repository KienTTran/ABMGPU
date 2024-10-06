//
// Created by kient on 6/17/2023.
//

#include "Person.h"
#include "../Model.h"
#include "Population.cuh"

Person::Person(){
    location = -1;
    location_col = -1;
    location_row = -1;
    age_class_ = -1;
    age_ = -1;
    population_ = nullptr;
    host_state_ = Person::SUSCEPTIBLE;
};
Person::~Person(){
    location = -1;
    location_col = -1;
    location_row = -1;
    age_class_ = -1;
    age_ = -1;
    population_ = nullptr;
    host_state_ = Person::SUSCEPTIBLE;
}


void Person::init() {
}

void Person::update() {
    assert(host_state_ != DEAD);
}

void Person::set_age(const int& value) {
    if (age_ != value) {
        // TODO::if age access the limit of age structure i.e. 100, remove person???

        NotifyChange(AGE, &age_, &value);

        // update bitting rate level
        age_ = value;

        // update age class
        if (Model::MODEL != nullptr) {
            auto ac = age_class_ == -1 ? 0 : age_class_;

            while (ac < (Model::CONFIG->number_of_age_classes() - 1) && age_ >= Model::CONFIG->age_structure()[ac]) {
                ac++;
            }

            set_age_class(ac);
        }
    }
}


int Person::age() const {
    return age_;
}

int Person::age_class() const {
    return age_class_;
}

void Person::set_age_class(const int& value) {
    if (age_class_ != value) {
        NotifyChange(AGE_CLASS, &age_class_, &value);
        age_class_ = value;
    }
}

Person::HostStates Person::host_state() const {
    return host_state_;
}

void Person::set_host_state(const HostStates& value) {
    if (host_state_ != value) {
        NotifyChange(HOST_STATE, &host_state_, &value);
        if (value == DEAD) {
        }
        host_state_ = value;
    }
}

void Person::NotifyChange(const Property& property, const void* oldValue, const void* newValue) {
    if (population_ != nullptr) {
        population_->notifyChange(this, property, oldValue, newValue);
    }
}
