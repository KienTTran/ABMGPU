//
// Created by Nguyen Tran on 3/5/2018.
//

#ifndef POMS_BFREPORTER_H
#define POMS_BFREPORTER_H

#include "Reporter.h"
#include <fstream>
#include <sstream>

class MonthlyReporter : public Reporter {
 DISALLOW_COPY_AND_ASSIGN(MonthlyReporter)

 DISALLOW_MOVE(MonthlyReporter)

public:

    std::ofstream monthly_data_file;
    std::ofstream summary_data_file;

public:

    MonthlyReporter();

    ~MonthlyReporter() override;

    void before_run() override { }

    void begin_time_step() override { }

    void initialize(int job_number, std::string path) override;

    void after_run() override;

    void monthly_report() override;

};

#endif //POMS_BFREPORTER_H
