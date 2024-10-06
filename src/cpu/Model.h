//
// Created by kient on 6/17/2023.
//

#ifndef MASS_MODEL_H
#define MASS_MODEL_H

#include "Core/Config/Config.h"
#include "Reporter/Reporter.h"
#include "Core/Random.h"
#include "Core/Scheduler.h"

class Config;
class Population;
class RenderEntity;
class Renderer;
class Scheduler;
class ModelDataCollector;

class Model {
public:
    Config* config_;
    Random *random_;
    Population* population_;
    RenderEntity* render_entity_;
    Renderer* renderer_;
    Scheduler* scheduler_;
    ModelDataCollector* mdc_;
    Person* person_;

    static Model *MODEL;
    static Config *CONFIG;
    static Random *RANDOM;
    static Population *POPULATION;
    static RenderEntity *RENDER_ENTITY;
    static Renderer *RENDERER;
    static Scheduler *SCHEDULER;
    static ModelDataCollector *MDC;
    static Person *PERSON;

public:
    bool model_finished = false;
    POINTER_PROPERTY(ModelDataCollector, data_collector)

    PROPERTY_REF(unsigned long, initial_seed_number)
    PROPERTY_REF(std::vector<Reporter *>, reporters)
    PROPERTY_REF(std::string, config_filename)
    PROPERTY_REF(std::string, output_path)
    PROPERTY_REF(int, cluster_job_number)
    PROPERTY_REF(int, study_number)                              // Should be -1 when not a valid study number
    PROPERTY_REF(std::string, tme_filename)
    PROPERTY_REF(std::string, override_parameter_filename)
    PROPERTY_REF(int, override_parameter_line_number) // base 1
    PROPERTY_REF(int, gui_type)
    PROPERTY_REF(bool, dump_movement)
    PROPERTY_REF(bool, individual_movement)
    PROPERTY_REF(bool, cell_movement)
    PROPERTY_REF(bool, district_movement)
    PROPERTY_REF(bool, is_farm_output)
    PROPERTY_REF(std::string, reporter_type)
    PROPERTY_REF(int, replicate)

public:
    DISALLOW_COPY_AND_ASSIGN(Model)
    DISALLOW_MOVE(Model)
    explicit Model();
    ~Model();
    void init(int job_number, const std::string& std);
    void run();
    void before_run();

    void after_run();

    void begin_time_step();

    void perform_population_events_daily() const;

    void daily_update(const int &current_time);

    void monthly_update();

    void yearly_update();

    void report_begin_of_time_step();

    void monthly_report();

    void add_reporter(Reporter *reporter);

};


#endif //MASS_MODEL_H
