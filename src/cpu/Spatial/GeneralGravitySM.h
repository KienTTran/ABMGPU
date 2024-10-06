//
// Created by Nguyen Tran on 1/29/2018.
//

#ifndef POMS_GENERALGRAVITYSM_H
#define POMS_GENERALGRAVITYSM_H

#include "../Core/PropertyMacro.h"
#include "SpatialModel.hxx"

namespace Spatial {

class GeneralGravitySM : public SpatialModel {
 DISALLOW_COPY_AND_ASSIGN(GeneralGravitySM)

 public:
  GeneralGravitySM();

  virtual ~GeneralGravitySM();

  DoubleVector get_v_relative_out_movement_to_destination(const int &from_location, const int &number_of_locations,
                                                          const DoubleVector &relative_distance_vector,
                                                          const IntVector &v_number_of_residents_by_location) const override;
};
}

#endif //POMS_GENERALGRAVITYSM_H
