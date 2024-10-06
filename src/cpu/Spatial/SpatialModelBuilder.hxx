/*
 * SpatialModelBuilder.cpp
 * 
 * Factory pattern to return the spatial model object that should be used.
 */
#ifndef SPATIAL_SPATIALMODELBUILDER_H
#define SPATIAL_SPATIALMODELBUILDER_H

#include "BarabasiSM.h"
#include "../Core/PropertyMacro.h"
#include "GeneralGravitySM.h"
#include "Marshall.hxx"
#include "BurkinaFaso.hxx"
#include "SpatialModel.hxx"
#include "WesolowskiSM.hxx"
#include "yaml-cpp/yaml.h"

namespace Spatial {
class SpatialModelBuilder {
  DISALLOW_COPY_AND_ASSIGN(SpatialModelBuilder)
  
  public:
    SpatialModelBuilder() = default;
    virtual ~SpatialModelBuilder() = default;
    
    static SpatialModel *Build(const std::string &name, const YAML::Node &node) {
      if (name == "Gravity") {
        return new GeneralGravitySM();
      }
      if (name == "Barabasi") {
        return new BarabasiSM(node);
      }
      if (name == "Marshall") {
        return new MarshallSM(node);
      }
      if (name == "Wesolowski") {
        return new WesolowskiSM(node);
      }
      if (name == "BurkinaFaso") {
        return new BurkinaFaso(node);
      }
      return nullptr;
    }
  };
}

#endif
