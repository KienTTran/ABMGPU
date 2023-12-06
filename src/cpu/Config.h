//
// Created by kient on 6/18/2023.
//

#ifndef ABMGPU_CONFIG_H
#define ABMGPU_CONFIG_H

#include <yaml-cpp/yaml.h>
#include "../utils/AscFile.h"


class Config {
public:
    YAML::Node config;
    std::vector<std::string> pop_asc_files;
    AscFile **pop_asc_file;
    bool is_window_rendered;

    int window_width;
    int window_height;
    int render_max_objects;
    bool render_gui;
    bool render_adaptive;
    float render_point_coord;

    int n_pops;
    int *n_people_1_pop_base;
    int *n_people_1_pop_max;
    std::vector<std::tuple<int,int,int,int,int,int>> *asc_cell_all;
    std::vector<std::tuple<int,int,int,int,int,int>> *asc_cell_people;
    std::vector<std::tuple<int,int,std::vector<std::tuple<int,int,int,int,int,int>>>> *asc_cell_info_range;
    double max_pop_factor;
    int min_people_1_batch;
    double p_people_1_batch_step;
    double p_max_mem_allow;
    int max_people_1_batch;
    double birth_rate;
    double death_rate;
    double velocity;

    int test_width;
    int test_height;
    bool test_update;
    bool test_debug;

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


#endif //ABMGPU_CONFIG_H
