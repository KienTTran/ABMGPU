//
// Created by Nguyen Tran on 6/21/18.
//

#ifndef PCMS_TIMEHELPERS_H
#define PCMS_TIMEHELPERS_H

#include <sstream>
#include "date/date.h"

inline std::ostream &operator<<(std::ostream &stream, const date::sys_days &o_days) {
  stream << date::year_month_day{o_days};
  return stream;
}

class TimeHelpers {
 public:
  static int number_of_days(
      const int &y1, const unsigned int &m1, const unsigned int &d1,
      const int &y2, const unsigned int &m2, const unsigned int &d2);

  static int number_of_days(const date::sys_days &first, const date::sys_days &last);

  template<typename T>
  static T convert_to(const std::string &input);

  static int number_of_days_to_next_year(const date::sys_days &today);

  static int get_simulation_time_birthday(const int &days_to_next_birthday, const int &age, const date::sys_days &
  starting_day);

  static int day_of_year(const int &y, const unsigned &m, const unsigned &d);

  static int day_of_year(const date::sys_days &day);

  static int month_of_year(const date::sys_days &day);

};

inline int TimeHelpers::number_of_days(const int &y1, const unsigned int &m1, const unsigned int &d1, const int &y2,
                                       const unsigned int &m2, const unsigned int &d2) {
  using namespace date;
  return (sys_days{year{y2}/month{m2}/day{d2}} - sys_days{year{y1}/month{m1}/day{d1}}).count();
}

inline int TimeHelpers::number_of_days(const date::sys_days &first, const date::sys_days &last) {
  return (last - first).count();
}

template<typename T>
T TimeHelpers::convert_to(const std::string &input) {
  T result{};
  std::stringstream ss(input);
  date::from_stream(ss, "%Y/%m/%d", result);
  return result;
}

inline int TimeHelpers::number_of_days_to_next_year(const date::sys_days &today) {

  const date::sys_days next_year{date::year_month_day{today} + date::years{1}};
  return number_of_days(today, next_year);
}

inline int TimeHelpers::get_simulation_time_birthday(const int &days_to_next_birthday, const int &age,
                                                     const date::sys_days &starting_day) {
  const auto calendar_birthday = date::floor<date::days>(
      starting_day + date::days{days_to_next_birthday + 1} - date::years{age + 1});

  return number_of_days(starting_day, calendar_birthday);

}

inline int TimeHelpers::day_of_year(const int &y, const unsigned &m, const unsigned &d) {
  using namespace date;

  if (m < 1 || m > 12 || d < 1 || d > 31) return 0;

  return (sys_days{year{y}/month{m}/day{d}} -
      sys_days{year{y}/jan/0}).count();
}

inline int TimeHelpers::day_of_year(const date::sys_days &day) {
  date::year_month_day ymd{day};
  return number_of_days(date::sys_days{ymd.year()/1/0}, day);
}

inline int TimeHelpers::month_of_year(const date::sys_days &day) {
  date::year_month_day ymd{day};
  return static_cast<unsigned>(ymd.month());
}

#endif //PCMS_TIMEHELPERS_H
