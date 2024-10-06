#include "IConfigItem.h"
#include "Config.h"

IConfigItem::IConfigItem(Config *config, const std::string &name) : config_{config}, name_{name} {
  config_->config_items.emplace_back(this);
}
