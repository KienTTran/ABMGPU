//
// Created by Nguyen Tran on 1/29/2018.
//

#ifndef SPATIAL_BARABASISM_H
#define SPATIAL_BARABASISM_H

#include "../Core/PropertyMacro.h"
#include "SpatialModel.hxx"
#include "yaml-cpp/yaml.h"

namespace Spatial {
class BarabasiSM : public SpatialModel {
 DISALLOW_COPY_AND_ASSIGN(BarabasiSM)

 VIRTUAL_PROPERTY_REF(double, r_g_0)

 VIRTUAL_PROPERTY_REF(double, beta_r)

 VIRTUAL_PROPERTY_REF(double, kappa)

 public:
  BarabasiSM(const YAML::Node &node);

  virtual ~ BarabasiSM();

  DoubleVector get_v_relative_out_movement_to_destination(const int &from_location, const int &number_of_locations,
                                                          const DoubleVector &relative_distance_vector,
                                                          const IntVector &v_number_of_residents_by_location) const override;

};
}

#endif //SPATIAL_BARABASISM_H
