/* 
 * Reporter.cpp
 * 
 * Implements a factory pattern to generate the various reporter types.
 */

#include "Reporter.h"

#include "../Constants.h"
#include "../Core/Config/Config.h"
#include "../MDC/ModelDataCollector.h"
#include "../Model.h"
#include "MonthlyReporter.h"

std::map<std::string, Reporter::ReportType> Reporter::ReportTypeMap{
//    {"Console", CONSOLE},
    {"MonthlyReporter", MONTHLY_REPORTER},
//    {"MMC", MMC_REPORTER},
//    {"DbReporter", DB_REPORTER},
//    {"DbReporterDistrict", DB_REPORTER_DISTRICT},
//    {"MovementReporter", MOVEMENT_REPORTER},
//    {"PopulationReporter", POPULATION_REPORTER},
//    {"CellularReporter", CELLULAR_REPORTER},
//    {"GenotypeCarriers", GENOTYPE_CARRIERS},
//    {"SeasonalImmunity", SEASONAL_IMMUNITY},
//    {"Null", NULL_REPORTER}
};

Reporter *Reporter::MakeReport(ReportType report_type) {
  switch (report_type) {
//    case CONSOLE:return new ConsoleReporter();
    case MONTHLY_REPORTER:return new MonthlyReporter();
//    case MMC_REPORTER:return new MMCReporter();
//    case DB_REPORTER: return new DbReporter();
//    case DB_REPORTER_DISTRICT: return new DbReporterDistrict();
//    case MOVEMENT_REPORTER: return new MovementReporter();
//    case POPULATION_REPORTER: return new PopulationReporter();
//    case CELLULAR_REPORTER: return new CellularReporter();
//    case GENOTYPE_CARRIERS: return new GenotypeCarriersReporter();
//    case SEASONAL_IMMUNITY: return new SeasonalImmunity();
//    case NULL_REPORTER: return new NullReporter();
    default:
      std::cout << "No reporter type supplied"<< std::endl;
      throw std::runtime_error("No reporter type supplied");
  }
}