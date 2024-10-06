/*
 * SeasonalEquation.cpp
 *
 * Implement the equation based seasonal model.
 */
#include "SeasonalInfo.h"

#if defined(WIN32) || defined(_WIN32)
#include <cmath>
#include <cstdlib>
#endif

#include "../Core/Config/CustomConfigItem.h"
#include "../Core/Config/Config.h"
#include "../../cpu/Constants.h"
#include "../Helpers/TimeHelpers.h"
#include "../Model.h"

SeasonalEquation* SeasonalEquation::build(const YAML::Node &node, Config* config) {
  // Prepare the object to be returned
  auto value = new SeasonalEquation();

  // Note the node for settings
  auto settings = node["equation"];

  // Before doing anything, check to see if there is a raster
  if (settings["raster"] && settings["raster"].as<bool>()) {
    value->set_from_raster(settings);
    return value;
  }

  // Check to make sure the nodes exist
  if (settings["base"].IsNull()  || settings["a"].IsNull() || settings["b"].IsNull() || settings["phi"].IsNull()) {
    throw std::invalid_argument("One or more of the seasonality equation parameters are missing.");
  }
  if (settings["base"].size() == 0  || settings["a"].size() == 0 || settings["b"].size() == 0 || settings["phi"].size() == 0) {
    throw std::invalid_argument("One or more of the seasonality equation parameters is an empty array.");
  }

  // Warn the user if enough nodes were not provided
  if (settings["a"].size() > 1 && settings["a"].size() < config->number_of_locations()) {
    LOG(WARNING) << fmt::format("Only {} seasonal  equation settings provided, but {} are needed for all locations", settings["a"].size(), config->number_of_locations());
  }

  // Set the values from the array and return
  for (auto i = 0ul; i < config->number_of_locations(); i++) {
    auto input_loc = settings["a"].size() < config->number_of_locations() ? 0 : i;
    value->set_seasonal_period(settings, input_loc);
  }
  return value;
}

double SeasonalEquation::get_seasonal_factor(const date::sys_days &today, const int &location) {
  // Note what day of the year it is
  int day = TimeHelpers::day_of_year(today);

  // Seasonal factor is determined by the algorithm:
  //
  // multiplier = base + (a * sin⁺(b * pi * (t - phi) / 365))
  auto multiplier = A[location] * sin(B[location] * Constants::PI() * (day - phi[location]) / Constants::DAYS_IN_YEAR());
  multiplier = (multiplier < 0) ? 0 : multiplier;
  multiplier += base[location];

  // Return the multiplier
  return multiplier;
}

// Set the values based upon the contents of a raster file.
void SeasonalEquation::set_from_raster(const YAML::Node &node) {
  // Get the raster data and make sure it is valid
  AscFile* raster = SpatialData::get_instance().get_raster(SpatialData::SpatialFileType::Ecoclimatic);
  if (raster == nullptr) {
    throw std::invalid_argument("Seasonal equation  raster flag set without eco-climatic raster loaded.");
  }

  // Prepare to run
  LOG(INFO) << "Setting seasonal equation using raster data.";

  // Load the values based upon the raster data
  auto size = node["a"].size();
  for (int row = 0; row < raster->NROWS; row++) {
    for (int col = 0; col < raster->NCOLS; col++) {
      // Pass if we have no data here
      if (raster->data[row][col] == raster->NODATA_VALUE) { continue; }

      // Verify the index
      int index = static_cast<int>(raster->data[row][col]);
      if (index < 0) { throw std::out_of_range(fmt::format("Raster value at row: {}, col: {} is less than zero.", row, col)); }
      if (index > (size - 1)) { throw std::out_of_range(fmt::format("Raster value at row: {}, col: {} exceeds bounds of {}.", row, col, size)); }

      // Set the seasonal period
      set_seasonal_period(node, index);
    }
  }
}

// Set the period for a single location given the index
void SeasonalEquation::set_seasonal_period(const YAML::Node &node, unsigned long index) {
  base.push_back(node["base"][index].as<double>());
  A.push_back(node["a"][index].as<double>());
  B.push_back(node["b"][index].as<double>());
  phi.push_back(node["phi"][index].as<double>());

  // Update the reference values as well
  reference_base.push_back(node["base"][index].as<double>());
  reference_A.push_back(node["a"][index].as<double>());
  reference_B.push_back(node["b"][index].as<double>());
  reference_phi.push_back(node["phi"][index].as<double>());
}

// Update the seasonality of the equation from the current to the new one.
void SeasonalEquation::update_seasonality(int from, int to) {
  for (auto ndx = 0; ndx < base.size(); ndx++) {
    if (base[ndx] == reference_base[from] && A[ndx] == reference_A[from] && B[ndx] == reference_B[from] && phi[ndx] == reference_phi[from]) {
      base[ndx] = reference_base[to];
      A[ndx] = reference_A[to];
      B[ndx] = reference_B[to];
      phi[ndx] = reference_phi[to];
    }
  }
}
