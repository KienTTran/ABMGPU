#ifndef YAMLCONVERTER_H
#define YAMLCONVERTER_H

#include <date/date.h>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>
#include "../TypeDef.h"
#include "../../Spatial/Location.h"
#include "../../GIS/SpatialData.h"

namespace YAML {
    template <>
    struct convert<date::sys_days> {
      static Node encode(const date::sys_days& rhs) {
        Node node;
        node.push_back(date::format("%Y/%m/%d", rhs));
        return node;
      }

      static bool decode(const Node& node, date::sys_days& rhs) {
        if (!node.IsScalar()) {
          return false;
        }
        std::stringstream ss(node.as<std::string>());
        date::from_stream(ss, "%Y/%m/%d", rhs);
        return true;
      }
    };

    template <>
    struct convert<date::year_month_day> {
      static Node encode(const date::year_month_day& rhs) {
        Node node;
        node.push_back(date::format("%Y/%m/%d", rhs));
        return node;
      }

      static bool decode(const Node& node, date::year_month_day& rhs) {
        if (!node.IsScalar()) {
          return false;
        }
        std::stringstream ss(node.as<std::string>());
        from_stream(ss, "%Y/%m/%d", rhs);
        return true;
      }
    };

    template <>
    struct convert<GPUConfig> {
        static Node encode(const GPUConfig& gcfe) {
            Node node;
            node.push_back("gpu_config");
            return node;
        }
        static bool decode(const Node& node, GPUConfig& gcfd) {
            gcfd.n_threads = node["n_threads"].as<int>();
            gcfd.people_1_batch = node["people_1_batch"].as<int>();
            gcfd.pre_allocated_mem_ratio = node["pre_allocated_mem_ratio"].as<double>();
            return true;
        }
    };


    template <>
    struct convert<RenderConfig> {
        static Node encode(const RenderConfig& rcfe) {
            Node node;
            node.push_back("render_config");
            return node;
        }
        static bool decode(const Node& node, RenderConfig& rcfd) {
            rcfd.display_gui = node["display_gui"].as<bool>();
            rcfd.close_window_on_finish = node["close_window_on_finish"].as<bool>();
            rcfd.point_coord = node["point_coord"].as<double>();
            rcfd.window_width = node["window_width"].as<int>();
            rcfd.window_height = node["window_height"].as<int>();
            return true;
        }
    };

    template <>
    struct convert<DebugConfig> {
        static Node encode(const DebugConfig& dgbcfe) {
            Node node;
            node.push_back("debug_config");
            return node;
        }
        static bool decode(const Node& node, DebugConfig& dgbcfd) {
            dgbcfd.enable_debug_render = node["enable_debug_render"].as<bool>();
            dgbcfd.enable_debug_text = node["enable_debug_text"].as<bool>();
            dgbcfd.enable_debug_render_text = node["enable_debug_render_text"].as<bool>();
            dgbcfd.enable_update = node["enable_update"].as<bool>();
            dgbcfd.height = node["height"].as<int>();
            dgbcfd.width = node["width"].as<int>();
            return true;
        }
    };

    template<>
    struct convert<RasterDb> {
        static Node encode(const RasterDb &rdb) {
            Node node;
            node.push_back("raster_db");
            return node;
        }
        static bool decode(const Node &node, RasterDb &rdb) {
            return SpatialData::get_instance().parse(node);
        }
    };

    template<>
    struct convert<std::vector<Spatial::Location>> {
        static Node encode(const std::vector<Spatial::Location> &rhs) {
            Node node;
            node.push_back("location_db");
            return node;
        }

        // Decode the contents of the location_db node
        static bool decode(const Node &node, std::vector<Spatial::Location> &location_db) {

            // Check to see if the location informatoin has already been entered, this implies
            // that there was a raster_db node present and an error starte exists
            if (location_db.size() != 0) {
                throw std::runtime_error("location_db has already been instantiated, is a raster_db present in the file?");
            }

            // If the user is supplying raster data, location_info will likely not be there.
            // Since we are just decoding the file, just focus on loading the data and defer
            // validation of the file to the recipent of the data
            auto number_of_locations = node["location_info"].size();
            for (std::size_t i = 0; i < number_of_locations; i++) {
                location_db.emplace_back(node["location_info"][i][0].as<int>(),
                                         node["location_info"][i][1].as<float>(),
                                         node["location_info"][i][2].as<float>(), 0);
            }

            for (std::size_t loc = 0; loc < number_of_locations; loc++) {
                auto input_loc = node["age_distribution_by_location"].size() < number_of_locations ? 0 : loc;

                for (std::size_t i = 0; i < node["age_distribution_by_location"][input_loc].size(); i++) {
                    location_db[loc].age_distribution.push_back(
                            node["age_distribution_by_location"][input_loc][i].as<double>());
                }
            }
            for (std::size_t loc = 0; loc < number_of_locations; loc++) {
                auto input_loc = node["p_treatment_for_less_than_5_by_location"].size() < number_of_locations ? 0 : loc;
                location_db[loc].p_treatment_less_than_5 = node["p_treatment_for_less_than_5_by_location"][input_loc].as<float>();
            }
            for (std::size_t loc = 0; loc < number_of_locations; loc++) {
                auto input_loc = node["p_treatment_for_more_than_5_by_location"].size() < number_of_locations ? 0 : loc;
                location_db[loc].p_treatment_more_than_5 = node["p_treatment_for_more_than_5_by_location"][input_loc].as<float>();
            }

            // If a raster was loaded for these items then use that instead
            if (SpatialData::get_instance().has_raster(SpatialData::SpatialFileType::Beta)) {
                printf("Beta raster and value supplied, ignoring beta_by_location setting\n");
            } else {
                for (std::size_t loc = 0; loc < number_of_locations; loc++) {
                    auto input_loc = node["beta_by_location"].size() < number_of_locations ? 0 : loc;
                    location_db[loc].beta = node["beta_by_location"][input_loc].as<float>();
                }
            }
            if (SpatialData::get_instance().has_raster(SpatialData::SpatialFileType::Population)) {
                printf("Population raster and value supplied, ignoring population_size_by_location setting\n");
            } else {
                for (std::size_t loc = 0; loc < number_of_locations; loc++) {
                    auto input_loc = node["population_size_by_location"].size() < number_of_locations ? 0 : loc;
                    location_db[loc].population_size = node["population_size_by_location"][input_loc].as<int>();
                }
            }

            return true;
        }
    };

}  // namespace YAML
#endif  // YAMLCONVERTER_H
