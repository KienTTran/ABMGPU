#ifndef CONFIGITEM_H
#define CONFIGITEM_H

#include "IConfigItem.h"
#include "YamlConverter.hxx"
#include <yaml-cpp/yaml.h>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <easylogging++.h>

#define CONFIG_ITEM(name, type, default_value)\
  ConfigItem<type> name{#name,default_value, this};

#define CUSTOM_CONFIG_ITEM(name, default_value)\
  ::name name{#name,default_value, this};

class Config;

template<typename T>
class ConfigItem : public IConfigItem {
 protected:
  T value_{};
 public:
  //constructor
  explicit ConfigItem(const std::string &name, T default_value, Config* config = nullptr);

  // destructor
  ~ConfigItem() override = default;

  // copy constructor
  ConfigItem(const ConfigItem &) = delete;

  // copy assignment
  ConfigItem &operator=(const ConfigItem &) = delete;

  // move constructor
  ConfigItem(ConfigItem &&) = delete;

  // move assignment
  ConfigItem &operator=(ConfigItem &&) = delete;

  virtual T &operator()();

  void set_value(const YAML::Node &node) override;
};

template<typename T>
ConfigItem<T>::ConfigItem(const std::string &name, T default_value, Config* config) : IConfigItem(config, name),
                                                                                      value_{std::move(default_value)} {
}

template<typename T>
T &ConfigItem<T>::operator()() {
  return value_;
};

template<typename T>
void ConfigItem<T>::set_value(const YAML::Node &node) {
  {
    if (node[name_]) {
      value_ = node[name_].template as<T>();
    } else {
      std::stringstream ss;
      ss << this->value_;
      LOG(WARNING) << name_ << " used default value of " << value_;
    }
  }
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  std::string sep;
  os << "[";
  for (const auto &i : v) {
    os << sep << i;
    sep = " , ";
  }
  os << " ]";
  return os;
}

template<typename T>
class ConfigItem<std::vector<T>> : public IConfigItem {
 protected:
  std::vector<T> value_{};
 public:
  //constructor
  explicit ConfigItem(std::string name, std::vector<T> default_value, Config* config = nullptr);

  // destructor
  ~ConfigItem() override = default;

  // copy constructor
  ConfigItem(const ConfigItem &) = delete;

  // copy assignment
  ConfigItem &operator=(const ConfigItem &) = delete;

  // move constructor
  ConfigItem(ConfigItem &&) = delete;

  // move assignment
  ConfigItem &operator=(ConfigItem &&) = delete;

  virtual std::vector<T> &operator()();

  void set_value(const YAML::Node &node) override;
};

template<typename T>
ConfigItem<std::vector<T>>::
ConfigItem(std::string name, std::vector<T> default_value, Config* config): IConfigItem(config, name),
                                                                            value_{std::move(default_value)} {
  // ReSharper disable once CppClassIsIncomplete
}

template<typename T>
std::vector<T> &ConfigItem<std::vector<T>>::operator()() {
  return value_;
}

template<typename T>
void ConfigItem<std::vector<T>>::set_value(const YAML::Node &node) {
  typedef std::vector<T> VectorT;
  if (node[this->name_]) {
    this->value_ = node[this->name_].template as<VectorT>();
  } else {
    std::stringstream ss;
    std::string sep;
    ss << "[";
    for (const auto &value : this->value_) {
      ss << sep << value;
      sep = " , ";
    }
    ss << "]";
    VLOG(1) << fmt::format("{} used default value of {}", this->name_, ss.str());
    LOG(INFO) << fmt::format("{} used default value of {}", this->name_, ss.str());
  }
}

#endif // CONFIGITEM_H
