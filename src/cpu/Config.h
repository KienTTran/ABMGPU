//
// Created by kient on 6/18/2023.
//

#ifndef MASS_CONFIG_H
#define MASS_CONFIG_H

#include <yaml-cpp/yaml.h>
#include "../utils/AscFile.h"

class Config {
public:
    YAML::Node config;

    int window_width;
    int window_height;

    int n_pops;
    uint64_t *n_people_1_pop;
    std::vector<std::string> pop_asc_files;

    AscFile **pop_asc_file;
    const double max_pop_factor = 1.2;
    const uint64_t min_people_1_pop = 10000000;
    const double p_max_mem_allow = 0.5;
    uint64_t max_people_1_pop = 0;

    float entity_velocity = 0.1f;

public:
    Config();
    ~Config();

    static Config& getInstance() // Singleton is accessed via getInstance()
    {
        static Config instance; // lazy singleton, instantiated on first use
        return instance;
    }

    void readConfigFile(std::string file_path);
};


#endif //MASS_CONFIG_H
