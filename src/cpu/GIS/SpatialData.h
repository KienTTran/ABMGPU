/*
 * SpatialData.h
 * 
 * Definitions of the thread-safe singleton pattern spatial class which manages the spatial aspects of the model from a high level.
 */
#ifndef SPATIALDATA_H
#define SPATIALDATA_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "AscFile.h"

class SpatialData {
    public:
        enum SpatialFileType {
            // Only use the data to define the model's location listing
            Locations,

            // Population data
            Population,

            // Transmission intensity, linked to the Entomological Inoculation Rates (EIR)
            Beta,

            // District location
            Districts,

            // Travel time data
            Travel,

            // Eco-climatic zones that are used for seasonal variation
            Ecoclimatic,

            // Probability of treatment, under 5
            PrTreatmentUnder5,

            // Probability of treatment, over 5
            PrTreatmentOver5,

            // Number of sequential items in the type
            Count
        };

        struct RasterInformation {
            // Flag to indicate the value has not been set yet
            static const int NOT_SET = -1;

            // The number of columns in the raster
            int number_columns = NOT_SET;

            // The number of rows in the raster
            int number_rows = NOT_SET;

            // The lower-left X coordinate of the raster
            double x_lower_left_corner = NOT_SET;

            // The lower-left Y coordinate of the raster
            double y_lower_left_corner = NOT_SET;

            // The size of the cell, typically in meters
            double cellsize = NOT_SET;
        };

    private:
        const std::string BETA_RASTER = "beta_raster";
        const std::string DISTRICT_RASTER = "district_raster";
        const std::string LOCATION_RASTER = "location_raster";
        const std::string POPULATION_RASTER = "population_raster";
        const std::string TRAVEL_RASTER = "travel_raster";
        const std::string ECOCLIMATIC_RASTER = "ecoclimatic_raster";
        const std::string TREATMENT_RATE_UNDER5 = "pr_treatment_under5";
        const std::string TREATMENT_RATE_OVER5 = "pr_treatment_over5";

        // Array of the ASC file data, use SpatialFileType as the index
        AscFile** data;

        // Flag to indicate if data has been loaded since the last time it was checked
        bool dirty = false;

        // The size of the cells in the raster, the units shouldn't matter, but this was
        // written when we were using 5x5 km cells
        float cell_size = 0;

        // First district index, default -1, lazy initialization to actual value
        int first_district = -1;

        // Count of district loaded in the map, default zero, lazy initialization to actual value
        int district_count = 0;

        // Constructor
        SpatialData();

        // Deconstructor
        ~SpatialData();

        // Check the loaded spatial catalog for errors, returns true if there are errors
        bool check_catalog(std::string& errors);

        // Generate the locations for the location_db
        void generate_locations();

        // Load the given raster file into the spatial catalog and assign the given label
        void load(const std::string &filename, SpatialFileType type);

        // Load all the spatial data from the node
        void load_files(const YAML::Node &node);

        // Load the raster indicated into the location_db; works with betas and probability of treatment
        void load_raster(SpatialFileType type);

        // Perform any clean-up operations after parsing the YAML file is complete
        void parse_complete();

    public:
        // Not supported by singleton.
        SpatialData(SpatialData const&) = delete;

        // Not supported by singleton.
        void operator=(SpatialData const&) = delete;

        // Get a reference to the spatial object.
        static SpatialData& get_instance() {
            static SpatialData instance;
            return instance;
        }

        // Return the raster header or the default structure if no raster are loaded
        RasterInformation get_raster_header();

        // Return true if any raster file has been loaded, false otherwise
        bool has_raster();

        // Return true if a raster file has been loaded, false otherwise
        bool has_raster(SpatialFileType type) { return data[type] != nullptr; }

        // Generate the Euclidean distances for the location_db
        void generate_distances() const;

        // Get the district id that corresponds to the cell id
        int get_district(int location);

        // Get the count of districts loaded, or -1 if they have not been loaded
        int get_district_count();

        // Get the locations that are within the given district, throws an error if not districts are loaded
        std::vector<int> get_district_locations(int district);

        // Returns the index of the first district.
        // Note that the index may be one (ArcGIS default) or zero; however, a delayed error is generated if the value
        // is not one of the two.
        int get_first_district();

        // Get a reference to the AscFile raster, may be a nullptr
        AscFile* get_raster(SpatialFileType type) { return data[type]; }

        // Parse the YAML node provided to extract all the relevant information for the simulation
        bool parse(const YAML::Node &node);

        // Refresh the data from the model (i.e., Location DB) to the spatial data
        void refresh();

        // Write the current spatial data to the filename and path indicated, output will be an ASC file
        void write(const std::string &filename, SpatialFileType type);

};

#endif
