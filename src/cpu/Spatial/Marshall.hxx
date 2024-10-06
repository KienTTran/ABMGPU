/*
 * MarshallSM.hxx
 * 
 * Gravity model for human migration based upon a distance kernel function.
 * 
 * Marshall et al., 2018
 */
#ifndef MARSHALLSM_HXX
#define MARSHALLSM_HXX

#include "../Core/Config/Config.h"
#include "../Helpers/NumberHelpers.h"
#include "../Model.h"
#include "SpatialModel.hxx"

#include "yaml-cpp/yaml.h"

namespace Spatial {
    class MarshallSM : public SpatialModel {
        DISALLOW_COPY_AND_ASSIGN(MarshallSM)

        VIRTUAL_PROPERTY_REF(double, tau)
        VIRTUAL_PROPERTY_REF(double, alpha)
        VIRTUAL_PROPERTY_REF(double, rho)

        private:
          // Hold on to the total number of locations, so we can free the kernel
          unsigned long locations = 0;

          // Pointer to the kernel object since it only needs to be computed once
          double** kernel = nullptr;

          // Precompute the kernel function for the movement model
          void prepare_kernel() {
            // Allocate the memory
            kernel = new double*[locations];

            // Get the distance matrix
            auto distance = Model::CONFIG->spatial_distance_matrix();

            // Iterate through all  the locations and calculate the kernel
            for (auto source = 0; source < locations; source++) {
              kernel[source] = new double[locations];
              for (auto destination = 0; destination < locations; destination++) {
                kernel[source][destination] = std::pow(1 + (distance[source][destination] / rho_), (-alpha_));
              }
            }
          }

        public:
            explicit MarshallSM(const YAML::Node &node) {
                tau_ = node["tau"].as<double>();
                alpha_ = node["alpha"].as<double>();
                rho_ = std::pow(10, node["log_rho"].as<double>());
            }

            ~MarshallSM() override {
              if (kernel != nullptr) {
                for (auto ndx = 0; ndx < locations; ndx++) {
                  delete kernel[ndx];
                }
                delete kernel;
              }
            }

            void prepare() override {
              locations = Model::CONFIG->number_of_locations();
              prepare_kernel();
            }

            [[nodiscard]]
            DoubleVector get_v_relative_out_movement_to_destination(
                    const int &from_location, const int &number_of_locations,
                    const DoubleVector &relative_distance_vector,
                    const IntVector &v_number_of_residents_by_location) const override { 

                // Note the population size
                auto population = v_number_of_residents_by_location[from_location];

                // Prepare the vector for results
                std::vector<double> results(number_of_locations, 0.0);

                for (auto destination = 0; destination < number_of_locations; destination++) {
                    // Continue if there is nothing to do
                    if (NumberHelpers::is_equal(relative_distance_vector[destination], 0.0)) { continue; }

                    // Calculate the proportional probability
                    double probability = std::pow(population, tau_) * kernel[from_location][destination];
                    results[destination] = probability;
                }

                // Done, return the results
                return results;
            }    
    };
}

#endif