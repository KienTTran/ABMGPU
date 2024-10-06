//
// Created by Nguyen Tran on 1/25/2018.
//

#include "Location.h"

namespace Spatial {
  Location::Location(const int id, float latitude, float longitude, const int population_size) :
    id{id}, population_size{population_size}, beta{0.0f}, p_treatment_less_than_5{0.0f},
    p_treatment_more_than_5{0.0f}, coordinate{std::make_unique<Coordinate>(latitude, longitude)} {
      location_cols = 0;
      location_rows = 0;
  }

  Location::~Location() = default;

  Location::Location(const Location &org) : id{org.id}, population_size{org.population_size},
                                            beta{org.beta}, p_treatment_less_than_5(0), p_treatment_more_than_5(0),
                                            coordinate{
                                              std::make_unique<Coordinate>(
                                                org.coordinate->latitude,
                                                org.coordinate->longitude)
                                            },
                                            age_distribution(org.age_distribution) {}

  std::ostream &operator<<(std::ostream &os, const Location &location) {
//    os << "id: " << location.id << ", population size: " << location.population_size << ", beta: " << location.beta
//       << ", coordinate: " << *location.coordinate << ", age_distribution: [";
//    for (auto i : location.age_distribution) {
//      os << i << ",";
//    }
//    os << "]";
//    os << ", p_treatment: [" << location.p_treatment_less_than_5 << "," << location.p_treatment_more_than_5 << "]"
//       << std::endl;
    return os;
  }

  Location &Location::operator=(const Location &other) {
    id = other.id;
    beta = other.beta;
    population_size = other.population_size;
    p_treatment_less_than_5 = other.p_treatment_less_than_5;
    p_treatment_more_than_5 = other.p_treatment_more_than_5;
    coordinate = std::make_unique<Coordinate>(other.coordinate->latitude, other.coordinate->longitude);
    age_distribution = other.age_distribution;
    return *this;
  }
}