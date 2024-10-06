//
// Created by Nguyen Tran on 1/25/2018.
//
#define _USE_MATH_DEFINES

#include <cmath>
#include <ostream>
#include "Coordinate.h"

namespace Spatial {

Coordinate::Coordinate(float latitude, float longitude) : latitude{latitude}, longitude{longitude} {

}

Coordinate::~Coordinate() {

}

double Coordinate::calculate_distance_in_km(const Coordinate &from, const Coordinate &to) {
  // using Haversine
  double p = M_PI/180;
  int R = 6371; // Radius of the Earth in km
  double dLat = p*(from.latitude - to.latitude);
  double dLon = p*(from.longitude - to.longitude);
  double a = sin(dLat/2)*sin(dLat/2) +
      cos(from.latitude*p)*cos(to.latitude*p)*sin(dLon/2)*sin(dLon/2);
  double c = 2*atan2(sqrt(a), sqrt(1 - a));
  double result = R*c;

  return result;
}

std::ostream &operator<<(std::ostream &os, const Coordinate &coordinate) {
  os << "[latitude: " << coordinate.latitude << " - longitude: " << coordinate.longitude << "]";
  return os;
}
}
