//
// Created by Nguyen Tran on 1/25/2018.
//

#ifndef SPATIAL_COORDINATE_H
#define SPATIAL_COORDINATE_H

#include <ostream>
#include "../Core/PropertyMacro.h"

namespace Spatial {
class Coordinate {
 DISALLOW_COPY_AND_ASSIGN(Coordinate)

 public:
  float latitude;
  float longitude;

 public:

  Coordinate(float latitude = 0, float longitude = 0);

  virtual ~Coordinate();

 public:
  static double calculate_distance_in_km(const Coordinate &from, const Coordinate &to);

  friend std::ostream &operator<<(std::ostream &os, const Coordinate &coordinate);

};

}

#endif //SPATIAL_COORDINATE_H
