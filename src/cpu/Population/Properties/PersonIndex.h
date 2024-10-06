/* 
 * File:   PersonIndex.h
 * Author: nguyentran
 *
 * Created on April 17, 2013, 10:01 AM
 */

#ifndef PERSONINDEX_H
#define    PERSONINDEX_H

#include "../../Core/PropertyMacro.h"
#include "../../../cpu/Population/Person.h"

class PersonIndex {
 DISALLOW_COPY_AND_ASSIGN(PersonIndex)

 public:
    PersonIndex();

    virtual ~PersonIndex();

  virtual void add(Person *p) = 0;

  virtual void remove(Person *p) = 0;

  virtual std::size_t size() const = 0;

  virtual void update() = 0;

  virtual void
  notifyChange(Person *p, const Person::Property &property, const void *oldValue, const void *newValue) = 0;

 private:

};

#endif    /* PERSONINDEX_H */

