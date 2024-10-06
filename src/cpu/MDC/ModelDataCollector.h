/*
 * ModelDataCollector.cpp
 * 
 * Definition of the model data collector class.
 * 
 * NOTE That age_class and age_group are used fairly interchangeably.
 */

#ifndef MODELDATACOLLECTOR_H
#define MODELDATACOLLECTOR_H

#include "../Core/PropertyMacro.h"
#include "../Core/TypeDef.h"

class Model;

class ModelDataCollector {
DISALLOW_COPY_AND_ASSIGN(ModelDataCollector)

POINTER_PROPERTY(Model, model)

PROPERTY_REF(DoubleVector, total_immune_by_location)

PROPERTY_REF(DoubleVector2, total_immune_by_location_age_class)

// Population size currently in the location
PROPERTY_REF(IntVector, popsize_by_location)

// Population size accounting for residence
PROPERTY_REF(IntVector, popsize_residence_by_location)

// Number of births by location; reset by monthly_update
PROPERTY_REF(IntVector, births_by_location)

// Number of deaths by location; reset by monthly_update
PROPERTY_REF(IntVector, deaths_by_location)

// Used to calculate the blood slide prevalence
PROPERTY_REF(IntVector2, popsize_by_location_age_class)
PROPERTY_REF(IntVector2, popsize_by_location_age_class_by_5)

PROPERTY_REF(IntVector2, popsize_by_location_hoststate)

PROPERTY_REF(DoubleVector, blood_slide_prevalence_by_location)

PROPERTY_REF(DoubleVector2, blood_slide_number_by_location_age_group)

PROPERTY_REF(DoubleVector2, blood_slide_prevalence_by_location_age_group)

PROPERTY_REF(DoubleVector2, blood_slide_number_by_location_age_group_by_5)

PROPERTY_REF(DoubleVector2, blood_slide_prevalence_by_location_age_group_by_5)

PROPERTY_REF(DoubleVector, fraction_of_positive_that_are_clinical_by_location)

PROPERTY_REF(LongVector, total_number_of_bites_by_location)

PROPERTY_REF(LongVector, total_number_of_bites_by_location_year)

PROPERTY_REF(LongVector, person_days_by_location_year)

PROPERTY_REF(DoubleVector2, EIR_by_location_year)

PROPERTY_REF(DoubleVector, EIR_by_location)

PROPERTY_REF(LongVector, cumulative_clinical_episodes_by_location)

PROPERTY_REF(DoubleVector2, average_number_biten_by_location_person)

PROPERTY_REF(DoubleVector, percentage_bites_on_top_20_by_location)

PROPERTY_REF(DoubleVector, cumulative_discounted_NTF_by_location)

PROPERTY_REF(DoubleVector, cumulative_NTF_by_location)

PROPERTY_REF(IntVector, today_TF_by_location)

PROPERTY_REF(IntVector, today_number_of_treatments_by_location)

PROPERTY_REF(IntVector, today_RITF_by_location)

PROPERTY_REF(IntVector2, total_number_of_treatments_60_by_location)

PROPERTY_REF(IntVector2, total_RITF_60_by_location)

PROPERTY_REF(IntVector2, total_TF_60_by_location)

PROPERTY_REF(DoubleVector, current_RITF_by_location)

PROPERTY_REF(DoubleVector, current_TF_by_location)

PROPERTY_REF(IntVector, cumulative_mutants_by_location)

PROPERTY_REF(int, current_utl_duration)

PROPERTY_REF(IntVector, UTL_duration)

// The total number of treatment given for the given therapy id
PROPERTY_REF(IntVector, number_of_treatments_with_therapy_ID)

// The total number of treatment failures with the given therapy
// NOTE we are not tracking the number of successes since it is generally
//      safe to assume that successes = (treatments - failures); however,
//      some error may be introduced for the treatments given close to 
//      model completion
PROPERTY_REF(IntVector, number_of_treatments_fail_with_therapy_ID)

PROPERTY_REF(double, AMU_per_parasite_pop)

PROPERTY_REF(double, AMU_per_person)

PROPERTY_REF(double, AMU_for_clinical_caused_parasite)

PROPERTY_REF(double, AFU)

PROPERTY_REF(double, discounted_AMU_per_parasite_pop)

PROPERTY_REF(double, discounted_AMU_per_person)

PROPERTY_REF(double, discounted_AMU_for_clinical_caused_parasite)

PROPERTY_REF(double, discounted_AFU)

PROPERTY_REF(IntVector2, multiple_of_infection_by_location)

PROPERTY_REF(DoubleVector, current_EIR_by_location)

PROPERTY_REF(LongVector, last_update_total_number_of_bites_by_location)

PROPERTY_REF(DoubleVector2, last_10_blood_slide_prevalence_by_location)

PROPERTY_REF(DoubleVector2, last_10_blood_slide_prevalence_by_location_age_class)

PROPERTY_REF(DoubleVector2, last_10_fraction_positive_that_are_clinical_by_location)

PROPERTY_REF(DoubleVector3, last_10_fraction_positive_that_are_clinical_by_location_age_class)

PROPERTY_REF(DoubleVector3, last_10_fraction_positive_that_are_clinical_by_location_age_class_by_5)

PROPERTY_REF(IntVector, total_parasite_population_by_location)

PROPERTY_REF(IntVector, number_of_positive_by_location)

PROPERTY_REF(IntVector2, total_parasite_population_by_location_age_group)

PROPERTY_REF(IntVector2, number_of_positive_by_location_age_group)

PROPERTY_REF(IntVector2, number_of_clinical_by_location_age_group)

PROPERTY_REF(IntVector2, number_of_clinical_by_location_age_group_by_5)

PROPERTY_REF(IntVector2, number_of_death_by_location_age_group)

PROPERTY_REF(double, tf_at_15)

PROPERTY_REF(double, single_resistance_frequency_at_15)

PROPERTY_REF(double, double_resistance_frequency_at_15)

PROPERTY_REF(double, triple_resistance_frequency_at_15)

PROPERTY_REF(double, quadruple_resistance_frequency_at_15)

PROPERTY_REF(double, quintuple_resistance_frequency_at_15)

PROPERTY_REF(double, art_resistance_frequency_at_15)

PROPERTY_REF(double, total_resistance_frequency_at_15)

PROPERTY_REF(IntVector, today_tf_by_therapy)

PROPERTY_REF(IntVector, today_number_of_treatments_by_therapy)

PROPERTY_REF(DoubleVector, current_tf_by_therapy)

PROPERTY_REF(IntVector2, total_number_of_treatments_60_by_therapy)

PROPERTY_REF(IntVector2, total_tf_60_by_therapy)

PROPERTY_REF(double, mean_moi)

PROPERTY_REF(LongVector, number_of_mutation_events_by_year)
PROPERTY_REF(long, current_number_of_mutation_events)

// NOTE The tracking for the following variables begins after 
// NOTE start_collect_data_day, and they are reset by monthly_update

// Monthly number of malaria deaths by location
PROPERTY_REF(IntVector, malaria_deaths_by_location)

// Monthly number of malaria deaths by location, binned by age class
PROPERTY_REF(IntVector2, malaria_deaths_by_location_age_class)

// Monthly number of non-treatments by location
PROPERTY_REF(IntVector, monthly_nontreatment_by_location)

// Monthly number of non-treatments by location binned by age class
PROPERTY_REF(IntVector2, monthly_nontreatment_by_location_age_class)

// Monthly number of clinical episodes by location
PROPERTY_REF(IntVector, monthly_number_of_clinical_episode_by_location);

// Monthly number of clinical episodes by location binned by age class
PROPERTY_REF(IntVector2, monthly_number_of_clinical_episode_by_location_age_class);

// Monthly number of new treatments by location
PROPERTY_REF(IntVector, monthly_number_of_new_infections_by_location);

// Monthly treatments by location
PROPERTY_REF(IntVector, monthly_number_of_treatment_by_location);

// Monthly number of treatment failures (i.e., treatment unsuccessfully administered) by location
PROPERTY_REF(IntVector, monthly_treatment_failure_by_location);

// Monthly number of treatment failures by location and age class
PROPERTY_REF(IntVector2, monthly_treatment_failure_by_location_age_class);

public:
  // The number of reported multiple of infection (MOI)
  static const int NUMBER_OF_REPORTED_MOI = 8;

  explicit ModelDataCollector(Model* model = nullptr);

  virtual ~ModelDataCollector() = default;

  void initialize();

  void perform_population_statistic();

  void perform_yearly_update();

  // Reset monthly tracking variables after the reporters had a chance to access them.
  void monthly_reset();

  virtual void collect_number_of_bites(const int &location, const int &number_of_bites);

  virtual void collect_1_clinical_episode(const int &location, const int &age_class);

  virtual void update_person_days_by_years(const int &location, const int &days);

  void calculate_eir();

  // Record one birth at the given location index, this value will be reset each month
  void record_1_birth(const int &location);

  // Record one death, note that this is independent of recording a malaria death
  void record_1_death(const int &location, const int &birthday, const int &number_of_times_bitten, const int &age_group);

  // Record one infection at the given location
  void record_1_infection(const int &location);

  // Record one malaria death, note that this only records the cause (i.e., malaria) and record_1_death 
  // still needs to be called to ensure proper metrics
  void  record_1_malaria_death(const int &location, const int &age_class);

  void calculate_percentage_bites_on_top_20();

  // Indicates a combined "treatment failure" due to either 1) failure to treat or 2) the treatment not being successful
  void record_1_TF(const int &location, const bool &by_drug);

  void record_1_treatment(const int &location, const int &therapy_id);

  // Records one case in which the individual did not receive treatment
  void record_1_non_treated_case(const int &location, const int &age_class);

  void record_1_mutation(const int &location);

  void begin_time_step();

  void end_of_time_step();

  // Update the useful therapeutic life (UTL) vector
  void update_UTL_vector();

  // Records one case in which a treatment was administered but failed due to patient death or failure to clear
  void record_1_treatment_failure_by_therapy(const int &location, const int &age_class, const int &therapy_id);

  void update_after_run();
//
//  // Calculate the artemisinin mono-therapy use (AMU) and artemisinin combination therapy failure rate (ATF)
//  [[maybe_unused]]
//  void record_AMU_AFU(Person* person, Therapy* therapy, ClonalParasitePopulation* clinical_caused_parasite);

  double get_blood_slide_prevalence(const int &location, const int &age_from, const int &age_to);

  // Return true if data is being recorded, false otherwise.
  static bool recording_data();

private:

  void update_average_number_bitten(const int &location, const int &birthday, const int &number_of_times_bitten);

  // Zero out the population statistics tracked by perform_population_statistic()
  void zero_population_statistics();
};

#endif

