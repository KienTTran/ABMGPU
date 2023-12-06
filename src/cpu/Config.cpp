//
// Created by kient on 6/18/2023.
//

#include <iostream>
#include <chrono>
#include "Config.h"

Config::Config() {
}

Config::~Config() {
}

void Config::readConfigFile(std::string file_path) {
    config = YAML::LoadFile(file_path);
    is_window_rendered = false;

    min_people_1_batch = config["GPU"]["people_1_batch"].as<int>();
    p_people_1_batch_step = config["GPU"]["p_people_1_batch_step"].as<double>();
    max_pop_factor = config["GPU"]["max_pop_factor"].as<double>();
    p_max_mem_allow = config["GPU"]["p_max_mem_allow"].as<double>();
    max_pop_factor = config["GPU"]["max_pop_factor"].as<double>();
    birth_rate = config["birth_rate"].as<double>();
    death_rate = config["death_rate"].as<double>();
    velocity = config["velocity"].as<double>();

    window_width = config["window"]["width"].as<int>();
    window_height = config["window"]["height"].as<int>();
    render_max_objects = config["render"]["max_objects"].as<int>();
    render_gui = config["render"]["GUI"].as<bool>();
    render_adaptive = config["render"]["adaptive"].as<bool>();
    render_point_coord = config["render"]["point_coord"].as<float>();

    printf("window_width: %d\n", window_width);
    printf("window_height: %d\n", window_height);

    test_width = config["test"]["width"].as<int>();
    test_height = config["test"]["height"].as<int>();
    test_update = config["test"]["update"].as<bool>();
    test_debug = config["test"]["debug"].as<bool>();

    pop_asc_files = config["populations"].as<std::vector<std::string>>();
    n_pops = pop_asc_files.size();
    n_people_1_pop_base = new int[n_pops];
    n_people_1_pop_max = new int[n_pops];
    pop_asc_file = new AscFile*[n_pops];
    asc_cell_all = new std::vector<std::tuple<int,int,int,int,int,int>>[n_pops];
    asc_cell_people = new std::vector<std::tuple<int,int,int,int,int,int>>[n_pops];
    asc_cell_info_range = new std::vector<std::tuple<int,int,std::vector<std::tuple<int,int,int,int,int,int>>>>[n_pops];

    for(int p_index = 0; p_index < n_pops; p_index++) {
        n_people_1_pop_base[p_index] = 0;
        n_people_1_pop_max[p_index] = 0;
        asc_cell_all[p_index] = std::vector<std::tuple<int,int,int,int,int,int>>();
        asc_cell_people[p_index] = std::vector<std::tuple<int,int,int,int,int,int>>();
        asc_cell_info_range[p_index] = std::vector<std::tuple<int,int,std::vector<std::tuple<int,int,int,int,int,int>>>>();
        printf("file_path: %s\n", pop_asc_files[p_index].c_str());
        pop_asc_file[p_index] = AscFileManager::read(pop_asc_files[p_index]);
        std::vector<std::tuple<int,int,int,int,int,int>> pop_data;
        std::tuple<int,int,std::vector<std::tuple<int,int,int,int,int,int>>> cell_range;
        int id = 0;
        int start_range = 0;
        int last_range = 0;
        int range_total = 0;
        int last_range_total = 0;
        int range_step = 0;
        int range_start_id = 0;
        bool reset_range = true;
        for (auto ndx = 0; ndx < pop_asc_file[p_index]->NROWS; ndx++) {
            for (auto ndy = 0; ndy < pop_asc_file[p_index]->NCOLS; ndy++) {
                std::tuple<int,int,int,int,int,int> cell_data;
                if(pop_asc_file[p_index]->data[ndx][ndy] > 0){
                    n_people_1_pop_base[p_index] += pop_asc_file[p_index]->data[ndx][ndy];
                    cell_data = std::make_tuple(ndx*pop_asc_file[p_index]->NCOLS+ndy,id,p_index,ndy,ndx,pop_asc_file[p_index]->data[ndx][ndy]);
                    pop_data.push_back(cell_data);
                    last_range = pop_asc_file[p_index]->data[ndx][ndy];
                    range_total += last_range;
                    range_step += 1;
                    if(reset_range) {
                        start_range = last_range;
                        range_start_id = id;
                    }
                    reset_range = false;
//                    printf(">= 0 start_range %d last_range: %d\n", start_range,last_range);
                    //Add last cell
                    if(ndx == pop_asc_file[p_index]->NROWS-1 && ndy == pop_asc_file[p_index]->NCOLS-1) {
                        cell_range = std::make_tuple(last_range_total,range_total,pop_data);
                        asc_cell_info_range[p_index].push_back(cell_range);
                    }
                    id++;
                    asc_cell_people[p_index].push_back(cell_data);
                }
                else{
                    cell_data = std::make_tuple(ndx*pop_asc_file[p_index]->NCOLS+ndy,-1,p_index,ndy,ndx,0);
                    if(last_range > 0){
//                        printf("< 0 RESET start_range %d last_range: %d\n", start_range,last_range);
                        cell_range = std::make_tuple(last_range_total,range_total,pop_data);
                        asc_cell_info_range[p_index].push_back(cell_range);
                        pop_data.clear();
                        reset_range = true;
                        last_range = 0;
                        last_range_total = range_total;
                        range_step = 0;
                    }
                }
                asc_cell_all[p_index].push_back(cell_data);
            }
        }
        n_people_1_pop_max[p_index] = n_people_1_pop_base[p_index] * max_pop_factor;
        printf("n_people_1_pop_base[%d]: %llu\n", p_index, n_people_1_pop_base[p_index]);
        printf("n_people_1_pop_max[%d]: %llu\n", p_index, n_people_1_pop_max[p_index]);

//        for(int index = 0; index < asc_cell_all[p_index].size(); index++){
//            std::tuple<int,int,int,int,int,int> cell_data = asc_cell_all[p_index][index];
//            if(std::get<1>(cell_data) >= 0){
//                printf("pop %d asc_cell_all[%d]: %d, %d, %d, (%d, %d), %d\n", p_index, index, std::get<0>(cell_data), std::get<1>(cell_data), std::get<2>(cell_data), std::get<3>(cell_data),std::get<4>(cell_data),std::get<5>(cell_data));
//            }
//        }
//        for(int b_index = 0; b_index < asc_cell_info_range[p_index].size(); b_index++){
//            std::tuple<int,int,std::vector<std::tuple<int,int,int,int,int>>> cell_range = asc_cell_info_range[p_index][b_index];
//            printf("pop %d cell_data_range[%d]: %d, %d\n", p_index, b_index, std::get<0>(cell_range), std::get<1>(cell_range));
//            std::vector<std::tuple<int,int,int,int,int>> cell_data = std::get<2>(cell_range);
//            for(int c_index = 0; c_index < cell_data.size(); c_index++){
//                printf("\tcell_data[%d]: %d, %d, (%d, %d), %d\n", c_index, std::get<0>(cell_data[c_index]), std::get<1>(cell_data[c_index]), std::get<2>(cell_data[c_index]), std::get<3>(cell_data[c_index]),std::get<4>(cell_data[c_index]));
//            }
//        }
    }
}