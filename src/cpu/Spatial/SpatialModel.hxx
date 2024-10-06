/*
 * SpatialModel.hxx
 *
 * Base abstract class for the spatial movement models that are implemented in the simulation.
 */
#ifndef SPATIAL_SPATIALMODEL_H
#define SPATIAL_SPATIALMODEL_H

#include "../Core/PropertyMacro.h"
#include "../Core/TypeDef.h"

namespace Spatial {
class SpatialModel {
 DISALLOW_COPY_AND_ASSIGN(SpatialModel)

 public:
  SpatialModel() = default;

  virtual ~SpatialModel() = default;

  // Allow the spatial model to perform any preparation it must do.
  virtual void prepare() { }

  [[nodiscard]] virtual DoubleVector
  get_v_relative_out_movement_to_destination(const int &from_location, const int &number_of_locations,
                                             const DoubleVector &relative_distance_vector,
                                             const IntVector &v_number_of_residents_by_location) const = 0;;

};
}

#endif
