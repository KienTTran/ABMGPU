/*
 * Scheduler.h
 *
 * Define the scheduler for the simulation, this allows events to be queued and
 * run at the correct interval where the time step is equal to one day.
 */
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <chrono>
#include "date/date.h"

#include "../Core/PropertyMacro.h"
#include "../Core/TypeDef.h"

class Model;

class Scheduler {
 DISALLOW_COPY_AND_ASSIGN(Scheduler)

 DISALLOW_MOVE(Scheduler)

 PROPERTY_REF(int, current_time)

 PROPERTY_HEADER(int, total_available_time)

 POINTER_PROPERTY(Model, model)

 PROPERTY_REF(bool, is_force_stop)

 // Number of days to wait between updating the user
 PROPERTY(int, days_between_notifications)

private:
  // Padding interval to use on the end of the total simulation time
  const int SCHEDULE_PADDING = 365 * 2;

  void begin_time_step() const;
  void end_time_step() const;

  void clear_all_events();
  static void clear_all_events(EventPtrVector2 &events_list);
  static void execute_events_list(EventPtrVector &events_list);
  virtual void schedule_event(EventPtrVector &time_events, Event *event);

  bool is_today_first_day_of_month() const;
  bool is_today_first_day_of_year() const;

public:
  date::sys_days calendar_date;

  EventPtrVector2 individual_events_list_;
  EventPtrVector2 population_events_list_;

  explicit Scheduler(Model *model = nullptr);

  virtual ~Scheduler();

  // Extend the total time that the current schedule will run
  void extend_total_time(int new_total_time);

  // Schedule an event that only operates upon an individual
  void schedule_individual_event(Event *event);

  // Schedule an event that operates on the whole population or simulation environment
  void schedule_population_event(Event *event);

  // Prepare the scheduler with the starting date and total time to run
  void initialize(const date::year_month_day &starting_date, const int &total_time);

  // Run the scheduler until the total time has been exhausted
  void run();

  // Return true if the simulation can stop, false otherwise
  bool can_stop() const;

  // Return the current day in the year based upon the current scheduler day
  int current_day_in_year() const;
};

#endif
