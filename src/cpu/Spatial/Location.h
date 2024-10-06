//
// Created by Nguyen Tran on 1/25/2018.
//

#ifndef SPATIAL_LOCATION_H
#define SPATIAL_LOCATION_H

#include "../Core/PropertyMacro.h"
#include "Coordinate.h"
#include "../Core/TypeDef.h"
#include <memory>
#include <ostream>
#include <vector>

namespace Spatial {

/*!
 *  Location is the smallest entity in the spatial structure.
 *  Location could be district, province, or zone depends on the availability of the data
 */

class Location {
//    DISALLOW_COPY_AND_ASSIGN_(Location)

 public:
  int id;
  int population_size;
  float beta;
  float p_treatment_less_than_5;
  float p_treatment_more_than_5;
  std::unique_ptr<Coordinate> coordinate;
  std::vector<double> age_distribution;
  ThrustInt4Tuple asc_cell_data;
  int location_cols;
  int location_rows;
 public:
  Location(int id, float latitude, float longitude, int population_size);

  virtual ~Location();

  Location(const Location &org);

  Location &operator=(const Location &other);

  friend std::ostream &operator<<(std::ostream &os, const Location &location);
};
}

#endif //SPATIAL_LOCATION_H
