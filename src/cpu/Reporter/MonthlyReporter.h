//
// Created by Nguyen Tran on 3/5/2018.
//

#ifndef POMS_MONTHLYREPORTER_H
#define POMS_MONTHLYREPORTER_H

#include "Reporter.h"
#include <fstream>
#include <memory>
#include <string>

// forward declare to avoid heavy include in header
namespace spdlog {
    class logger;
}

class MonthlyReporter final : public Reporter {
    DISALLOW_COPY_AND_ASSIGN(MonthlyReporter)
    DISALLOW_MOVE(MonthlyReporter)

  public:
    MonthlyReporter();
    ~MonthlyReporter() override;

    void before_run() override {}
    void begin_time_step() override {}

    void initialize(int job_number, std::string path) override;
    void after_run() override;
    void monthly_report() override;

private:
    // Keep these private; reporter owns its output sinks
    std::ofstream monthly_data_file_;
    std::ofstream summary_data_file_;

    // spdlog loggers (if you use spdlog for writing as well)
    std::shared_ptr<spdlog::logger> monthly_logger_;
    std::shared_ptr<spdlog::logger> summary_logger_;
};

#endif  // POMS_MONTHLYREPORTER_H
