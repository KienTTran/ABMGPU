//
// Created by Nguyen Tran on 3/5/2018.
//

#include "MonthlyReporter.h"

#include "../Model.h"
#include "../Core/Config/Config.h"
#include "../MDC/ModelDataCollector.h"
#include "../Helpers/TimeHelpers.h"
#include "../Constants.h"
#include "../Core/Scheduler.h"

#include <date/date.h>

// spdlog
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <chrono>
#include <utility>

namespace {

// Normalize base path so we can safely do base + filename.
std::string normalize_dir(std::string path) {
  if (!path.empty() && path.back() != '/' && path.back() != '\\') {
    path.push_back('/');
  }
  return path;
}

}  // namespace

MonthlyReporter::MonthlyReporter() = default;
MonthlyReporter::~MonthlyReporter() = default;

void MonthlyReporter::initialize(int job_number, std::string path) {
  path = normalize_dir(std::move(path));

  const auto monthly_log_path = fmt::format("{}monthly_data_{}.txt", path, job_number);
  const auto summary_log_path = fmt::format("{}summary_{}.txt", path, job_number);

  // Create (or replace) file loggers. Truncate = true (fresh file per run).
  monthly_logger_ = spdlog::basic_logger_mt("monthly_reporter", monthly_log_path, /*truncate=*/true);
  summary_logger_ = spdlog::basic_logger_mt("summary_reporter", summary_log_path, /*truncate=*/true);

  // Message-only format (equivalent to "%msg")
  monthly_logger_->set_pattern("%v");
  summary_logger_->set_pattern("%v");

  // Flush immediately at INFO and above (robust for long simulations)
  monthly_logger_->flush_on(spdlog::level::info);
  summary_logger_->flush_on(spdlog::level::info);

  // Optional: keep explicit file streams (if you still want them)
  monthly_data_file_.open(fmt::format("{}/monthly_data_{}.txt",
                                      Model::MODEL->output_path(),
                                      Model::MODEL->cluster_job_number()));
  summary_data_file_.open(fmt::format("{}/summary_{}.txt",
                                      Model::MODEL->output_path(),
                                      Model::MODEL->cluster_job_number()));
}

void MonthlyReporter::monthly_report() {
  std::stringstream ss;

  ss << Model::SCHEDULER->current_time() << Tsv::sep;
  ss << std::chrono::system_clock::to_time_t(Model::SCHEDULER->calendar_date) << Tsv::sep;
  ss << date::format("%Y\t%m\t%d", Model::SCHEDULER->calendar_date) << Tsv::sep;
//  ss << Model::POPULATION->size() << Tsv::sep;
  ss << group_sep;

  const std::string line = ss.str();

  if (monthly_data_file_.is_open()) {
    monthly_data_file_ << line << '\n';
  }
  if (monthly_logger_) {
    monthly_logger_->info("{}", line);
  }
}

void MonthlyReporter::after_run() {
  std::stringstream ss;

//  ss << Model::RANDOM->seed();
  ss << Tsv::sep << Model::CONFIG->number_of_locations() << Tsv::sep;
  ss << Model::CONFIG->location_db()[0].beta << Tsv::sep;
  ss << Model::CONFIG->location_db()[0].population_size << Tsv::sep;
  ss << group_sep;

  const std::string line = ss.str();

  if (summary_data_file_.is_open()) {
    summary_data_file_ << line << '\n';
  }
  if (summary_logger_) {
    summary_logger_->info("{}", line);
  }
}
