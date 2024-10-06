/* 
 * File:   PersonIndexHandler.h
 * Author: nguyentran
 *
 * Created on April 17, 2013, 10:29 AM
 */

#ifndef PERSONINDEXHANDLER_H
#define    PERSONINDEXHANDLER_H

#include <cuda_runtime.h>
#include "../../Core/PropertyMacro.h"

class IndexHandler {
 DISALLOW_COPY_AND_ASSIGN(IndexHandler)

 PROPERTY_REF(std::size_t, index)

 public:
    IndexHandler();

    virtual ~IndexHandler();

};

#endif    /* PERSONINDEXHANDLER_H */

