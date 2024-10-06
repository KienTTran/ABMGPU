#ifndef ICONFIGITEM_H
#define ICONFIGITEM_H

#include <string>

namespace YAML {
class Node;
}

class Config;

class IConfigItem {
 protected:
  Config *config_{nullptr};
  std::string name_;
 public:
  explicit IConfigItem(Config *config, const std::string &name);

  virtual ~IConfigItem() = default;

  virtual const std::string &name() {
    return name_;
  }

  virtual void set_value(const YAML::Node &node) = 0;
};

#endif // ICONFIGITEM_H
