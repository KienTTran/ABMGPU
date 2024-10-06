/* 
 * File:   PersonIndexByLocationStateAgeClassHandler.h
 * Author: nguyentran
 *
 * Created on May 2, 2013, 10:57 AM
 */

#ifndef PERSONINDEXBYLOCATIONSTATEAGECLASSHANDLER_H
#define    PERSONINDEXBYLOCATIONSTATEAGECLASSHANDLER_H

#include "../../Core/PropertyMacro.h"
#include "IndexHandler.h"

class PersonIndexByLocationStateAgeClassHandler : public IndexHandler {
 DISALLOW_COPY_AND_ASSIGN(PersonIndexByLocationStateAgeClassHandler)

 public:
    PersonIndexByLocationStateAgeClassHandler();

//    PersonIndexByLocationStateAgeClassHandler(const PersonIndexByLocationStateAgeClassHandler& orig);
    virtual ~PersonIndexByLocationStateAgeClassHandler();

 private:

};

#endif    /* PERSONINDEXBYLOCATIONSTATEAGECLASSHANDLER_H */

