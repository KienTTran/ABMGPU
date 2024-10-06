/*
 * File:   Config.cpp
 * Author: nguyentran
 *
 * Created on March 27, 2013, 10:38 AM
 */

#include "Config.h"
#include <fstream>

Config::Config(Model *model) : model_(model){
}

Config::~Config() {
};

void Config::readConfigFile(const std::string &config_file_name) {
  YAML::Node config;
  try {
//    LOG(INFO) << "Reading config file: " << config_file_name;
    printf("Reading config file: %s\n", config_file_name.c_str());
    config = YAML::LoadFile(config_file_name);
  } catch (YAML::BadFile &ex) {
//    LOG(FATAL) << config_file_name << " not found or err... Ex: " << ex.msg;
    printf("%s not found or err... Ex: %s\n", config_file_name.c_str(), ex.msg.c_str());
  } catch (YAML::Exception &ex) {
//    LOG(FATAL) << "error: " << ex.msg << " at line " << ex.mark.line + 1 << ":" << ex.mark.column + 1;
    printf("error: %s at line %d:%d\n", ex.msg.c_str(), ex.mark.line + 1, ex.mark.column + 1);
  } catch(const YAML::ParserException& ex) {
    std::cout << ex.what() << std::endl;
  }

  for (auto &config_item : config_items) {
//    LOG(INFO) << "Reading config item: " << config_item->name();
    printf("Reading config item: %s\n", config_item->name().c_str());
    config_item->set_value(config);
  }
}

