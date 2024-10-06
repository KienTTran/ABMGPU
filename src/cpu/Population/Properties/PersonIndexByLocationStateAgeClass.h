/* 
 * File:   PersonIndexByLocationStateAgeClass.h
 * Author: nguyentran
 *
 * Created on May 2, 2013, 10:41 AM
 */

#ifndef PERSONINDEXBYLOCATIONSTATEAGECLASS_H
#define    PERSONINDEXBYLOCATIONSTATEAGECLASS_H

#include "../../Core/PropertyMacro.h"
#include "../../Core/TypeDef.h"
#include "../../../cpu/Population/Person.h"
#include "PersonIndex.h"

class PersonIndexByLocationStateAgeClass : public PersonIndex {
 DISALLOW_COPY_AND_ASSIGN(PersonIndexByLocationStateAgeClass)

  PROPERTY_REF(ThrustPersonPtrVectorHost4, h_persons);

  PROPERTY_REF(ThrustGLMat4VectorHost4, h_person_models);
  PROPERTY_REF(ThrustGLVec4VectorHost4, h_person_colors);

 public:
  //    PersonIndexByLocationStateAgeClass();
  PersonIndexByLocationStateAgeClass(const int &no_location = 1, const int &no_host_state = 1,
                                     const int &no_age_class = 1);

  //    PersonIndexByLocationStateAgeClass(const PersonIndexByLocationStateAgeClass& orig);
  virtual ~PersonIndexByLocationStateAgeClass();

  void Initialize(const int &no_location = 1, const int &no_host_state = 1, const int &no_age_class = 1);

  virtual void add(Person *p);

  virtual void remove(Person *p);

  virtual std::size_t size() const;

  virtual void update();

  virtual void notifyChange(Person *p, const Person::Property &property, const void *oldValue, const void *newValue);

 private:
  void remove_without_set_index(Person *p);

  void add(Person *p, const int &location, const Person::HostStates &host_state, const int &age_class);

  void change_property(Person *p, const int &location, const Person::HostStates &host_state, const int &age_class);
};

#endif    /* PERSONINDEXBYLOCATIONSTATEAGECLASS_H */

