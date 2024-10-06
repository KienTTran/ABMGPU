/*
 * BurkinaFaso.hxx
 * 
 * Tuned movement model for Burkina Faso based upon Marshall et al. (2018), 
 * with a penalty applied based upon the travel time to the nearest city. 
 * Intradistrict movement in capital is also penalized as well.
 * 
 * Marshall et al., 2018
 */
#ifndef BURKINAFASO_HXX
#define BURKINAFASO_HXX

#include "../Core/Config/Config.h"
#include "../Helpers/NumberHelpers.h"
#include "../Model.h"
#include "SpatialModel.hxx"

#include "easylogging++.h"
#include "yaml-cpp/yaml.h"

namespace Spatial {
    class BurkinaFaso : public SpatialModel {
        DISALLOW_COPY_AND_ASSIGN(BurkinaFaso)

        VIRTUAL_PROPERTY_REF(double, tau)
        VIRTUAL_PROPERTY_REF(double, alpha)
        VIRTUAL_PROPERTY_REF(double, rho)
        VIRTUAL_PROPERTY_REF(double, capital)
        VIRTUAL_PROPERTY_REF(double, penalty)

        private:
            // Hold on to the total number of locations, so we can free the kernel
            unsigned long locations = 0;

            // These variables will be computed when the prepare method is called
            double* travel = nullptr;
            double** kernel = nullptr;

            // Precompute the kernel function for the movement model
            void prepare_kernel() {
              // Prepare the kernel object
              kernel = new double*[locations];

              // Get the distance matrix
              auto distance = Model::CONFIG->spatial_distance_matrix();

              // Iterate through all the locations and calculate the kernel
              for (auto source = 0; source < locations; source++) {
                kernel[source] = new double[locations];
                for (auto destination = 0; destination < locations; destination++) {
                  kernel[source][destination] = std::pow(1 + (distance[source][destination] / rho_), (-alpha_));
                }
              }
            }

            // Prepare the travel raster for the movement model
            void prepare_travel(const SpatialData::SpatialFileType type) {
              // Get the travel times raster
              AscFile* raster = SpatialData::get_instance().get_raster(type);
              if (raster == nullptr) {
                throw std::runtime_error(fmt::format("{} called without travel data loaded", __FUNCTION__));
              }

              // Use the min and max to normalize the raster into an array
              auto id = 0;
              travel = new double[locations];
              for (auto row = 0; row < raster->NROWS; row++) {
                for (auto col = 0; col < raster->NCOLS; col++) {
                  if (raster->data[row][col] == raster->NODATA_VALUE) { continue; }
                  travel[id] = raster->data[row][col];
                  id++;
                }
              }
            }

        public:
            explicit BurkinaFaso(const YAML::Node &node) {
                tau_ = node["tau"].as<double>();
                alpha_ = node["alpha"].as<double>();
                rho_ = std::pow(10, node["log_rho"].as<double>());
                capital_ = node["capital"].as<double>();
                penalty_ = node["penalty"].as<double>();
            }

            ~BurkinaFaso() override {
              // Delete memory allocated to the kernel
              if (kernel != nullptr) {
                for (auto ndx = 0; ndx < locations; ndx++) {
                  delete kernel[ndx];
                }
                delete kernel;
              }

              // Delete everything else
              delete travel;
            }

            void prepare() override {
              // Note the number of locations
              locations = Model::CONFIG->number_of_locations();

              // Allow the work to be done
              prepare_kernel();
              prepare_travel(SpatialData::SpatialFileType::Travel);
            }

            [[nodiscard]]
            DoubleVector get_v_relative_out_movement_to_destination(
                    const int &from_location, const int &number_of_locations,
                    const DoubleVector &relative_distance_vector,
                    const IntVector &v_number_of_residents_by_location) const override {

                // Dependent objects should have been created already, so throw an exception if they are not
                if (kernel == nullptr || travel == nullptr) {
                  throw std::runtime_error(fmt::format("{} called without kernel or travel surface prepared", __FUNCTION__));
                }

                // Note the population size
                auto population = v_number_of_residents_by_location[from_location];

                // Note the source district
                auto source_district = SpatialData::get_instance().get_district(from_location);

                // Prepare the vector for results
                std::vector<double> results(number_of_locations, 0.0);

                for (auto destination = 0; destination < number_of_locations; destination++) {
                    // Continue if there is nothing to do
                    if (NumberHelpers::is_equal(relative_distance_vector[destination], 0.0)) { continue; }

                    // Calculate the proportional probability
                    double probability = std::pow(population, tau_) * kernel[from_location][destination];

                    // Adjust the probability by the friction surface
                    probability = probability / (1 + travel[from_location] + travel[destination] );

                    // If the source and the destination are both in the capital district,
                    // penalize the travel by 50%
                    if (source_district == capital_ && 
                        SpatialData::get_instance().get_district(destination) == capital_) {
                            probability /= penalty_;
                    }

                    results[destination] = probability;
                }

                // Done, return the results
                return results;
            }
    };
}

#endif