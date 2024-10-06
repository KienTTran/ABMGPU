#define _USE_MATH_DEFINES
#define NOMINMAX

#include "CustomConfigItem.h"
#include "Config.h"
#include "../../GIS/SpatialData.h"
#include "../../Spatial/SpatialModelBuilder.hxx"
#include <gsl/gsl_cdf.h>

void total_time::set_value(const YAML::Node &node) {
  value_ = (date::sys_days { config_->ending_date() } - date::sys_days(config_->starting_date() )).count();
  printf("total_time is: %d\n", value_);
}

void number_of_age_classes::set_value(const YAML::Node &node) {
  value_ = static_cast<int>(config_->age_structure().size());
}

void number_of_locations::set_value() {
  value_ = static_cast<int>(config_->location_db().size());
}

void number_of_locations::set_value(const YAML::Node &node) {
  set_value();
}

void start_of_comparison_period::set_value(const YAML::Node &node) {
  const auto ymd = node[name_].as<date::year_month_day>();
  value_ = (date::sys_days{ymd} - date::sys_days(config_->starting_date())).count();
}

void spatial_distance_matrix::set_value(const YAML::Node &node) {
  if (SpatialData::get_instance().has_raster()) {
    LOG(WARNING) << "Raster data detected, using it to generate distances";
    SpatialData::get_instance().generate_distances();
    return;
  }

  LOG(WARNING) << "Generating Euclidian distances using coordinates provided (number of locations = " << config_->number_of_locations() << ")";
  value_.resize(static_cast<unsigned long>(config_->number_of_locations()));
  for (auto from_location = 0ul; from_location < config_->number_of_locations(); from_location++) {
    value_[from_location].resize(static_cast<unsigned long long int>(config_->number_of_locations()));
    for (auto to_location = 0ul; to_location < config_->number_of_locations(); to_location++) {
      value_[from_location][to_location] = Spatial::Coordinate::calculate_distance_in_km(
              *config_->location_db()[from_location].coordinate,
              *config_->location_db()[to_location].coordinate);
    }
  }
}

seasonal_info::~seasonal_info() {
  if (value_ != nullptr) {
    delete value_;
    value_ = nullptr;
  }
}

void seasonal_info::set_value(const YAML::Node &node) {
  value_ = SeasonalInfoFactory::build(node[name_], config_);
}

spatial_model::~spatial_model() {
  if (value_ != nullptr) {
    delete value_;
    value_ = nullptr;
  }
}

void spatial_model::set_value(const YAML::Node &node) {
  const auto sm_name = node[name_]["name"].as<std::string>();
  value_ = Spatial::SpatialModelBuilder::Build(sm_name, node[name_][sm_name]);
  LOG(INFO) << "Using spatial model: " << sm_name;
}


void circulation_info::set_value(const YAML::Node &node) {
  auto info_node = node[name_];
  value_.max_relative_moving_value = info_node["max_relative_moving_value"].as<double>();

  value_.number_of_moving_levels = info_node["number_of_moving_levels"].as<int>();

  value_.scale = info_node["moving_level_distribution"]["Exponential"]["scale"].as<double>();

  value_.mean = info_node["moving_level_distribution"]["Gamma"]["mean"].as<double>();
  value_.sd = info_node["moving_level_distribution"]["Gamma"]["sd"].as<double>();

  //calculate density and level value here

  const auto var = value_.sd * value_.sd;

  const auto b = var / (value_.mean - 1); //theta
  const auto a = (value_.mean - 1) / b; //k

  value_.v_moving_level_density.clear();
  value_.v_moving_level_value.clear();

  const auto max = value_.max_relative_moving_value - 1; //maxRelativeBiting -1
  const auto number_of_level = value_.number_of_moving_levels;

  const auto step = max / static_cast<double>(number_of_level - 1);

  auto j = 0;
  double old_p = 0;
  double sum = 0;
  for (double i = 0; i <= max + 0.0001; i += step) {
    const auto p = gsl_cdf_gamma_P(i + step, a, b);
    double value = 0;
    value = (j == 0) ? p : p - old_p;
    value_.v_moving_level_density.push_back(value);
    old_p = p;
    value_.v_moving_level_value.push_back(i + 1);
    sum += value;
    j++;

  }

  //normalized
  double t = 0;
  for (auto &i : value_.v_moving_level_density) {
    i = i + (1 - sum) / value_.v_moving_level_density.size();
    t += i;
  }

  assert((unsigned)value_.number_of_moving_levels == value_.v_moving_level_density.size());
  assert((unsigned)value_.number_of_moving_levels == value_.v_moving_level_value.size());
  assert(fabs(t - 1) < 0.0001);

  value_.circulation_percent = info_node["circulation_percent"].as<double>();

  const auto length_of_stay_mean = info_node["length_of_stay"]["mean"].as<double>();
  const auto length_of_stay_sd = info_node["length_of_stay"]["sd"].as<double>();

  const auto stay_variance = length_of_stay_sd * length_of_stay_sd;
  const auto k = stay_variance / length_of_stay_mean; //k
  const auto theta = length_of_stay_mean / k; //theta

  value_.length_of_stay_mean = length_of_stay_mean;
  value_.length_of_stay_sd = length_of_stay_sd;
  value_.length_of_stay_theta = theta;
  value_.length_of_stay_k = k;
}