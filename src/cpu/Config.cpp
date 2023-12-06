//
// Created by kient on 6/18/2023.
//

#include <iostream>
#include "Config.h"

Config::Config() {
}

Config::~Config() {
}

void Config::readConfigFile(std::string file_path) {
    config = YAML::LoadFile(file_path);

    window_width = config["window"]["width"].as<int>();
    window_height = config["window"]["height"].as<int>();

    printf("window_width: %d\n", window_width);
    printf("window_height: %d\n", window_height);

    entity_velocity = config["entity"]["velocity"].as<float>();

    pop_asc_files = config["populations"].as<std::vector<std::string>>();
    n_pops = pop_asc_files.size();
    n_people_1_pop = new uint64_t[n_pops];
    pop_asc_file = new AscFile*[n_pops];

    for(int p_index = 0; p_index < n_pops; p_index++) {
        n_people_1_pop[p_index] = 0;
        printf("file_path: %s\n", pop_asc_files[p_index].c_str());
        pop_asc_file[p_index] = AscFileManager::read(pop_asc_files[p_index]);
        for (auto ndx = 0; ndx < pop_asc_file[p_index]->NROWS; ndx++) {
            for (auto ndy = 0; ndy < pop_asc_file[p_index]->NCOLS; ndy++) {
                if(pop_asc_file[p_index]->data[ndx][ndy] > 0){
                    n_people_1_pop[p_index] += pop_asc_file[p_index]->data[ndx][ndy];
                }
            }
        }
//        n_people_1_pop[p_index] = 10000;
        n_people_1_pop[p_index] = (uint64_t) (n_people_1_pop[p_index] * max_pop_factor);
        printf("n_people_1_pop[%d]: %llu\n", p_index, n_people_1_pop[p_index]);
    }

}