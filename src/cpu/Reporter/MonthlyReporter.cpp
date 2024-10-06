//
// Created by Nguyen Tran on 3/5/2018.
//

#include "MonthlyReporter.h"
#include "../Model.h"
#include "../Core/Config/Config.h"
#include "../MDC/ModelDataCollector.h"
#include "../Helpers/TimeHelpers.h"
#include "../Constants.h"
#include "easylogging++.h"
#include <date/date.h>
#include "../Core/Scheduler.h"

MonthlyReporter::MonthlyReporter() = default;

MonthlyReporter::~MonthlyReporter() = default;

void MonthlyReporter::initialize(int job_number, std::string path) {
  // Create the configuration for the monthly reporter
  el::Configurations monthly_reporter_logger;
  monthly_reporter_logger.setToDefault();
  monthly_reporter_logger.set(el::Level::Info, el::ConfigurationType::Format, "%msg");
  monthly_reporter_logger.setGlobally(el::ConfigurationType::ToFile, "true");
  monthly_reporter_logger.setGlobally(el::ConfigurationType::Filename, fmt::format("{}monthly_data_{}.txt", path, job_number));
  monthly_reporter_logger.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
  monthly_reporter_logger.setGlobally(el::ConfigurationType::LogFlushThreshold, "100");
  el::Loggers::reconfigureLogger("monthly_reporter", monthly_reporter_logger);

  // Create the configuration for the summary reporter
  el::Configurations summary_reporter_logger;
  summary_reporter_logger.setToDefault();
  summary_reporter_logger.set(el::Level::Info, el::ConfigurationType::Format, "%msg");
  summary_reporter_logger.setGlobally(el::ConfigurationType::ToFile, "true");
  summary_reporter_logger.setGlobally(el::ConfigurationType::Filename, fmt::format("{}summary_{}.txt", path, job_number));
  summary_reporter_logger.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
  summary_reporter_logger.setGlobally(el::ConfigurationType::LogFlushThreshold, "100");
  el::Loggers::reconfigureLogger("summary_reporter", summary_reporter_logger);

  monthly_data_file.open(fmt::format("{}/monthly_data_{}.txt", Model::MODEL->output_path(), Model::MODEL->cluster_job_number()));
  summary_data_file.open(fmt::format("{}/summary_{}.txt", Model::MODEL->output_path(), Model::MODEL->cluster_job_number()));
}

void MonthlyReporter::monthly_report()
{
  ss << Model::SCHEDULER->current_time() << Tsv::sep;
  ss << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) << Tsv::sep;
  ss << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) << Tsv::sep;
//  ss << Model::POPULATION->size() << Tsv::sep;
  ss << group_sep;

  monthly_data_file << ss.str() << std::endl;

  CLOG(INFO, "monthly_reporter") << ss.str();
  ss.str("");
}

void MonthlyReporter::after_run()
{
  ss.str("");
//  ss << Model::RANDOM->seed();
  ss << Tsv::sep << Model::CONFIG->number_of_locations() << Tsv::sep;
  ss << Model::CONFIG->location_db()[0].beta << Tsv::sep;
  ss << Model::CONFIG->location_db()[0].population_size << Tsv::sep;
  ss << group_sep;

  summary_data_file << ss.str() << std::endl;

  CLOG(INFO, "summary_reporter") << ss.str();
  ss.str("");
}
