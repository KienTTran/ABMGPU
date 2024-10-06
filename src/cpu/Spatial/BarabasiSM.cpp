//
// Created by Nguyen Tran on 1/29/2018.
//

#include <cmath>
#include "BarabasiSM.h"
#include "../Helpers/NumberHelpers.h"

namespace Spatial {

  BarabasiSM::BarabasiSM(const YAML::Node &node) {
    r_g_0_ = node["r_g_0"].as<double>();
    beta_r_ = node["beta_r"].as<double>();
    kappa_ = node["kappa"].as<double>();
  }

  BarabasiSM::~BarabasiSM() = default;

  DoubleVector
  BarabasiSM::get_v_relative_out_movement_to_destination(const int &from_location, const int &number_of_locations,
                                                         const DoubleVector &relative_distance_vector,
                                                         const IntVector &v_number_of_residents_by_location) const {
    DoubleVector v_relative_number_of_circulation_by_location(number_of_locations, 0);

    for (int target_location = 0; target_location < number_of_locations; target_location++) {
      if (NumberHelpers::is_equal(relative_distance_vector[target_location], 0.0)) {
        v_relative_number_of_circulation_by_location[target_location] = 0;
      } else {
        v_relative_number_of_circulation_by_location[target_location] =
            pow((relative_distance_vector[target_location] + r_g_0_), -beta_r_)*
                exp(-r_g_0_/kappa_);   // equation from Barabasi's paper
      }
    }

    return v_relative_number_of_circulation_by_location;
  }
}