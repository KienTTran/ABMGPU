//
// Created by kient on 6/17/2023.
//

#ifndef MASS_PERSON_H
#define MASS_PERSON_H

#include <cstdint>
#include <glm/ext/matrix_float4x4.hpp>
#include <thrust/host_vector.h>
#include "Properties/PersonIndexAllHandler.h"
#include "Properties/PersonIndexToRenderHandler.h"
#include "Properties/PersonIndexByLocationStateAgeClassHandler.h"
#include "../Parasite/Parasite.h"

class Population;

class Scheduler;

class Model;

class Person: public PersonIndexAllHandler,
              public PersonIndexByLocationStateAgeClassHandler,
              public PersonIndexToRenderHandler{
public:
    enum Property {
        LOCATION = 0,
        HOST_STATE,
        AGE,
        AGE_CLASS,
        BITING_LEVEL,
        MOVING_LEVEL,
        EXTERNAL_POPULATION_MOVING_LEVEL
    };
    enum HostStates{
        SUSCEPTIBLE = 0,
        EXPOSED = 1,
        ASYMPTOMATIC = 2,
        CLINICAL = 3,
        DEAD = 4,
        NON_EXIST = 5,
        NUMBER_OF_STATE = 6
    };
    glm::mat4 model;
    glm::vec4 color;
    int location;
    int location_col;
    int location_row;
    PROPERTY_HEADER(HostStates, host_state)
    PROPERTY_HEADER(int, age);
    PROPERTY_HEADER(int, age_class)
    POINTER_PROPERTY(Population, population)
    thrust::host_vector<Parasite*> parasites;
public:
    Person();
    virtual ~Person();
    void init();
    void update();

    void NotifyChange(const Property &property, const void *oldValue, const void *newValue);
};


#endif //MASS_GPUPERSON_H
