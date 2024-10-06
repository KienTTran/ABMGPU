#ifndef CUSTOMCONFIGITEM_H
#define CUSTOMCONFIGITEM_H

#include <string>
#include <utility>
#include "ConfigItem.hxx"
#include "../PropertyMacro.h"
#include "../../Environment/SeasonalInfo.h"

namespace YAML {
class Node;
}

class Config;

class total_time : public ConfigItem<int> {
 public:
  total_time(const std::string &name, const int &default_value, Config *config) : ConfigItem<int>(name, default_value,config) {}

  void set_value(const YAML::Node &node) override;
};

class start_of_comparison_period : public ConfigItem<int> {
public:
    start_of_comparison_period(const std::string &name, const int &default_value, Config *config) : ConfigItem<int>(
            name, default_value, config) {}

    void set_value(const YAML::Node &node) override;
};


class number_of_age_classes : public ConfigItem<unsigned long> {
public:
    number_of_age_classes(const std::string &name, const unsigned long &default_value, Config *config) : ConfigItem<unsigned long>(name,
                                                                                                                                   default_value,
                                                                                                                                   config) {}

    void set_value(const YAML::Node &node) override;
};

class number_of_locations : public ConfigItem<unsigned long> {
public:
    number_of_locations(const std::string &name, const unsigned long &default_value, Config *config) : ConfigItem<unsigned long>(name,
                                                                                                                                 default_value,
                                                                                                                                 config) {}
    // Update the number of locations based upon the location_db size
    void set_value();

    // Update the number of locations based upon the location_db size, the node is ignored
    void set_value(const YAML::Node &node) override;
};

class spatial_distance_matrix : public ConfigItem<std::vector<std::vector<double>>> {
public:
    spatial_distance_matrix(const std::string &name, const std::vector<std::vector<double>> &default_value,
                            Config *config) : ConfigItem<std::
    vector<
            std::vector<double>>>(
            name, default_value, config) {}

    void set_value(const YAML::Node &node) override;
};

// This class allows for the seasonal information object to be loaded based upon the configuration
class seasonal_info : public IConfigItem {
DISALLOW_COPY_AND_ASSIGN(seasonal_info)
DISALLOW_MOVE(seasonal_info)

private:
    ISeasonalInfo* value_{nullptr};

public:
    // Invoked via macro definition
    explicit seasonal_info(const std::string &name, ISeasonalInfo *default_value, Config *config = nullptr) :
            IConfigItem(config, name), value_{default_value} { }

    ~seasonal_info() override;

    ISeasonalInfo* operator()() { return value_; }
    void set_value(const YAML::Node &node) override;
};

namespace Spatial {
    class SpatialModel;
}

class spatial_model : public IConfigItem {
    DISALLOW_COPY_AND_ASSIGN(spatial_model)

    DISALLOW_MOVE(spatial_model)

public:
    Spatial::SpatialModel *value_{nullptr};
public:
    //constructor
    explicit spatial_model(const std::string &name, Spatial::SpatialModel *default_value, Config *config = nullptr) :
            IConfigItem(config, name),
            value_{default_value} {}

    // destructor
    virtual ~spatial_model();

    virtual Spatial::SpatialModel *operator()() {
        return value_;
    }

    void set_value(const YAML::Node &node) override;
};


class circulation_info : public IConfigItem {
DISALLOW_COPY_AND_ASSIGN(circulation_info)

DISALLOW_MOVE(circulation_info)

public:
    RelativeMovingInformation value_;
public:
    //constructor
    explicit circulation_info(const std::string &name, RelativeMovingInformation default_value,
                              Config *config = nullptr) :
            IConfigItem(config, name),
            value_{std::move(default_value)} {}

    // destructor
    virtual ~circulation_info() = default;

    virtual RelativeMovingInformation &operator()() {
        return value_;
    }

    void set_value(const YAML::Node &node) override;
};

#endif // CUSTOMCONFIGITEM_H
