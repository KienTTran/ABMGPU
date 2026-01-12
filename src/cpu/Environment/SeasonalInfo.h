/*
 * SeasonalInfo.h
 *
 * Define the various seasonal information methods that are supported by
 */
#ifndef SEASONALINFO_H
#define SEASONALINFO_H

#include <algorithm>
#include <date/date.h>
#include <fstream>
#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include "../Core/TypeDef.h"

class Config;

class ISeasonalInfo {
  public:
    // Return the seasonal factor for the given day and location, based upon the loaded configuration.
    virtual double get_seasonal_factor(const date::sys_days &today, const int &location) {
      throw std::runtime_error("Runtime call to virtual function");
    }
};

class SeasonalDisabled : public ISeasonalInfo {
  public:
    double get_seasonal_factor(const date::sys_days &today, const int &location) override { return 1.0; }
};

class SeasonalEquation : public ISeasonalInfo {
  private:
    DoubleVector base;
    DoubleVector A;
    DoubleVector B;
    DoubleVector phi;

    // The reference values contain the inputs from the YAML so the UpdateEcozoneEvent
    // can change the ecozone (i.e., seasonal information) during model execution
    DoubleVector reference_base;
    DoubleVector reference_A;
    DoubleVector reference_B;
    DoubleVector reference_phi;

    void set_from_raster(const YAML::Node &node);
    void set_seasonal_period(const YAML::Node &node, unsigned long index);

  public:
    static SeasonalEquation* build(const YAML::Node &node, Config* config);
    double get_seasonal_factor(const date::sys_days &today, const int &location) override;
    void update_seasonality(int from, int to);
};

class SeasonalRainfall : public ISeasonalInfo {
  private:
    DoubleVector adjustments;
    int period;

    void read(std::string &filename);

  public:
    static SeasonalRainfall* build(const YAML::Node &node);
    double get_seasonal_factor(const date::sys_days &today, const int &location) override;
};

class SeasonalInfoFactory {
  public:
    static ISeasonalInfo* build(const YAML::Node &node, Config* config) {
      // If seasonality has been disabled then don't bother parsing the rest
      auto enabled = node["enable"].as<bool>();
      if (!enabled) {
        std::cout << "Seasonal information disabled."<< std::endl;
        return new SeasonalDisabled();
      }

      // Check to make sure the mode node exists
      try {
        if (node["mode"].IsNull()) {
          throw std::invalid_argument("Seasonal information mode is not found.");
        }
      } catch (YAML::InvalidNode &ex) {
        throw std::invalid_argument("Seasonal information mode node is not found.");
      }

      // Return the correct object for the named mode
      auto mode = node["mode"].as<std::string>();
      std::transform(mode.begin(), mode.end(), mode.begin(), ::toupper);
      if (mode == "EQUATION") {
        std::cout << "Using equation-based seasonal information."<< std::endl;
        return SeasonalEquation::build(node, config);
      }
      if (mode == "RAINFALL") {
        std::cout << "Using rainfall-based seasonal information."<< std::endl;
        return SeasonalRainfall::build(node);
      }
      throw std::runtime_error(fmt::format("Unknown seasonal mode {}", mode));
    }
};

#endif
