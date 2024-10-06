//
// Created by kient on 6/17/2023.
//

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include "Model.h"
#include "../cpu/Population/Population.cuh"
#include "../gpu/RenderEntity.cuh"
#include "../cpu/Renderer.h"
#include "../cpu/Core/Random.h"
#include "../cpu/MDC/ModelDataCollector.h"
#include "Helpers/StringHelpers.h"

Model* Model::MODEL = nullptr;
Config* Model::CONFIG = nullptr;
Random* Model::RANDOM = nullptr;
Population* Model::POPULATION = nullptr;
Scheduler* Model::SCHEDULER = nullptr;
ModelDataCollector* Model::MDC = nullptr;
RenderEntity* Model::RENDER_ENTITY = nullptr;
Renderer* Model::RENDERER = nullptr;

__host__ __device__ Model::Model() {
    config_ = new Config(this);
    random_ = new Random();
    population_ = new Population(this);
    render_entity_ = new RenderEntity(this);
    renderer_ = new Renderer(this);
    scheduler_ = new Scheduler(this);
    mdc_ = new ModelDataCollector(this);

    MODEL = this;
    CONFIG = config_;
    RANDOM = random_;
    POPULATION = population_;
    RENDER_ENTITY = render_entity_;
    RENDERER = renderer_;
    SCHEDULER = scheduler_;
    MDC = mdc_;
}
__host__ __device__ Model::~Model() {
}

void Model::add_reporter(Reporter* reporter) {
    reporters_.push_back(reporter);
    reporter->set_model(this);
}

void Model::init(int job_number, const std::string& std) {
    config_->readConfigFile(config_filename_);

    LOG(INFO) << "Initialize Random";
    // Initialize Random Seed
    initial_seed_number_ = Model::CONFIG->initial_seed_number() == 0 ? initial_seed_number_ : Model::CONFIG->initial_seed_number();
    random_->initialize(initial_seed_number_);

    population_->init();//init variables
    population_->initPop();//init h_population using gpu
    render_entity_->initEntity();//send h_population to render
    renderer_->init(render_entity_);

    // MARKER add reporter here
    VLOG(1) << "Initialing reporter(s)...";
    try {
        if (reporter_type_.empty()) {
            add_reporter(Reporter::MakeReport(Reporter::MONTHLY_REPORTER));
        } else {
            for (const auto& type : StringHelpers::split(reporter_type_, ',')) {
                if (Reporter::ReportTypeMap.find(type) != Reporter::ReportTypeMap.end()) {
                    add_reporter(Reporter::MakeReport(Reporter::ReportTypeMap[type]));
                } else {
                    std::cerr << "ERROR! Unknown reporter type: " << type << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
        for (auto* reporter : reporters_) {
            reporter->initialize(job_number, std);
        }
    } catch (std::invalid_argument &ex) {
        LOG(ERROR) << "Initialing reporter generated exception: " << ex.what();
        exit(EXIT_FAILURE);
    } catch (std::runtime_error &ex) {
        LOG(ERROR) << "Runtime error encountered while initializing reporter: " << ex.what();
        exit(EXIT_FAILURE);
    }

    VLOG(1) << "Initializing scheduler...";
    LOG(INFO) << "Starting day is " << CONFIG->starting_date();
    scheduler_->initialize(CONFIG->starting_date(), config_->total_time());
    scheduler_->set_days_between_notifications(config_->days_between_notifications());
}

void Model::run() {
    LOG(INFO) << "Model starting...";
    before_run();

    auto start = std::chrono::system_clock::now();
    LOG(INFO) << "Start running model";
    if(Model::CONFIG->render_config().display_gui){
        std::thread scheduler_thread(&Scheduler::run, scheduler_);
        renderer_->start();
        scheduler_thread.join();
    }
    else{
        scheduler_->run();
    }
    auto end = std::chrono::system_clock::now();

    after_run();
    LOG(INFO) << "Model finished!";

    // Note the final run-time of the model
    std::chrono::duration<double> elapsed_seconds = end-start;
    LOG(INFO) << fmt::format("Elapsed time (s): {0}", elapsed_seconds.count());
}

void Model::monthly_report() {
    data_collector_->perform_population_statistic();

    for (auto* reporter : reporters_) {
        reporter->monthly_report();
    }

}

void Model::report_begin_of_time_step() {
    for (auto* reporter : reporters_) {
        reporter->begin_time_step();
    }
}

void Model::before_run() {
    LOG(INFO) << scheduler_->current_time() << " " << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " before_run";
    for (auto* reporter : reporters_) {
        reporter->before_run();
    }
}

void Model::after_run() {
    LOG(INFO) << scheduler_->current_time() << " " << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " after_run";

    data_collector_->update_after_run();

    for (auto* reporter : reporters_) {
        reporter->after_run();
    }
}

void Model::begin_time_step() {
    LOG(INFO) << scheduler_->current_time() <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " begin_time_step";
    //reset daily variables
    data_collector_->begin_time_step();
    report_begin_of_time_step();
}

void Model::perform_population_events_daily() const {
    LOG(INFO) << scheduler_->current_time() <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " perform_population_events_daily";
//    // TODO: turn on and off time for art mutation in the input file
//    population_->perform_infection_event();
    population_->performBirthEvent();
//    population_->perform_circulation_event();
}

void Model::daily_update(const int &current_time) {
    LOG(INFO) << scheduler_->current_time() << " " << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) <<
    " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
    " " << " Perform daily update";
//    //for safety remove all dead by calling perform_death_event
    population_->performDeathEvent();

    population_->update();
    population_->updateRender();
//
//    //update / calculate daily UTL
    data_collector_->end_of_time_step();
//
//    //update force of infection
//    population_->update_force_of_infection(current_time);
//
//    //check to switch strategy
//    treatment_strategy_->update_end_of_time_step();
}

void Model::monthly_update() {
    LOG(INFO) << scheduler_->current_time() << " " << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " Perform monthly_update update";
    monthly_report();
//
//    //reset monthly variables
    data_collector()->monthly_reset();
//
//    //
//    treatment_strategy_->monthly_update();
//
//    //update treatment coverage
//    treatment_coverage_->monthly_update();

}

// ReSharper disable once CppMemberFunctionMayBeConst
void Model::yearly_update() {
    LOG(INFO) << scheduler_->current_time() << " " << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) <<
              " " << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) <<
              " " << " Perform yearly_update update";
    data_collector_->perform_yearly_update();
}