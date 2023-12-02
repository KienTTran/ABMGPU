
#include "cpu/Config.h"
#include "cpu/Model.h"

int main(int argc, char* argv[]) {
    Config *config = &Config::getInstance();
    config->readConfigFile("../input/config.yml");
    Model *model = new Model();
    model->init();
    model->run();
    return 0;
}
