/*
 * File:   Config.h
 * Author: nguyentra
 *
 * Created on March 27, 2013, 10:38 AM
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <date/date.h>
#include <string>
#include <vector>
#include "ConfigItem.hxx"
#include "CustomConfigItem.h"
#include "../PropertyMacro.h"
#include "../TypeDef.h"
#include "../../GIS/AscFile.h"
#include "../../Spatial/Location.h"
#include "YamlConverter.hxx"

class Model;

class Config {
  DISALLOW_COPY_AND_ASSIGN(Config)
  DISALLOW_MOVE(Config)

public:
    POINTER_PROPERTY(Model, model)
    std::vector<IConfigItem*> config_items {};

    CONFIG_ITEM(asc_pop_nrows, int, 0)
    CONFIG_ITEM(asc_pop_ncols, int, 0)
    CONFIG_ITEM(n_people_init, int, 0)

    CONFIG_ITEM(initial_seed_number, unsigned long, 0)
    CONFIG_ITEM(days_between_notifications, int, 100)
    CONFIG_ITEM(report_frequency, int, 30)
    CONFIG_ITEM(starting_date, date::year_month_day, date::year_month_day { date::year { 1999 } / 1 / 1 })
    CONFIG_ITEM(ending_date, date::year_month_day, date::year_month_day { date::year { 1999 } / 1 / 2 })
    CONFIG_ITEM(start_collect_data_day, int, 365)
    CONFIG_ITEM(number_of_tracking_days, int, 0)
    CUSTOM_CONFIG_ITEM(total_time, 100)
    CONFIG_ITEM(gpu_config, GPUConfig, GPUConfig())
    CONFIG_ITEM(render_config, RenderConfig, RenderConfig())
    CONFIG_ITEM(debug_config, DebugConfig, DebugConfig())

    CONFIG_ITEM(age_structure, std::vector<int>, std::vector<int>{})
    CONFIG_ITEM(initial_age_structure, std::vector<int>, std::vector<int>{})

    CUSTOM_CONFIG_ITEM(number_of_age_classes, 0)

    // Either the raster_db field or the location_db MUST be supplied in the YAML
    // and they MUST appear by this point in the file as well
    CONFIG_ITEM(raster_db, RasterDb, RasterDb())
    CONFIG_ITEM(location_db, std::vector<Spatial::Location>,
                std::vector<Spatial::Location>{Spatial::Location(0, 0, 0, 10000)})

    CUSTOM_CONFIG_ITEM(number_of_locations, 0)

    CUSTOM_CONFIG_ITEM(spatial_distance_matrix, DoubleVector2())

    CONFIG_ITEM(birth_rate, double, 0)

    CONFIG_ITEM(as_iov, double, 0.2)

    CONFIG_ITEM(death_rate_by_age_class, DoubleVector, DoubleVector())

    CONFIG_ITEM(mortality_when_treatment_fail_by_age_class, DoubleVector, DoubleVector())

    CUSTOM_CONFIG_ITEM(spatial_model, nullptr)

    CUSTOM_CONFIG_ITEM(circulation_info, RelativeMovingInformation())

    CUSTOM_CONFIG_ITEM(start_of_comparison_period, 0)

public:
    explicit Config(Model *model = nullptr);

    virtual ~Config();

    void readConfigFile(const std::string &config_file_name = "config.yml");

};

#endif /* CONFIG_H */
