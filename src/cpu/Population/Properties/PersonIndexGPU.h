//
// Created by kient on 12/9/2023.
//

#ifndef MASS_PERSONINDEXGPU_H
#define MASS_PERSONINDEXGPU_H

#include "PersonIndex.h"
#include "../../Core/PropertyMacro.h"
#include "../../Core/TypeDef.h"

class PersonIndexGPU : public PersonIndex {
DISALLOW_COPY_AND_ASSIGN(PersonIndexGPU)

PROPERTY_REF(ThrustPersonPtrVectorHost, h_persons)
PROPERTY_REF(ThrustGLMat4VectorHost, h_person_models);
PROPERTY_REF(ThrustGLVec4VectorHost, h_person_colors);
//PROPERTY_REF(ThrustParasitePtrVectorHost2, h_person_parasites);

public:
    PersonIndexGPU();

    virtual ~PersonIndexGPU();

virtual void add(Person *p);

virtual void remove(Person *p);

virtual std::size_t size() const;

virtual void update();

virtual void notifyChange(Person *p, const Person::Property &property, const void *oldValue, const void *newValue);

private:
};


#endif //MASS_PERSONINDEXGPU_H
