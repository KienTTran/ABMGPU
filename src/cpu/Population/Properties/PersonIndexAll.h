/* 
 * File:   PersonIndexAll.h
 * Author: nguyentran
 *
 * Created on April 17, 2013, 10:15 AM
 */

#ifndef PERSONINDEXALL_H
#define    PERSONINDEXALL_H

#include "PersonIndex.h"
#include "../../Core/PropertyMacro.h"
#include "../../Core/TypeDef.h"

class PersonIndexAll : public PersonIndex {
 DISALLOW_COPY_AND_ASSIGN(PersonIndexAll)

 PROPERTY_REF(ThrustPersonPtrVectorHost, h_persons)
 PROPERTY_REF(ThrustGLMat4VectorHost, h_person_models);
 PROPERTY_REF(ThrustGLVec4VectorHost, h_person_colors);

 public:
    PersonIndexAll();

  virtual ~PersonIndexAll();

  virtual void add(Person *p);

  virtual void remove(Person *p);

  virtual std::size_t size() const;

  virtual void update();

  virtual void notifyChange(Person *p, const Person::Property &property, const void *oldValue, const void *newValue);

 private:

};

#endif    /* PERSONINDEXALL_H */

