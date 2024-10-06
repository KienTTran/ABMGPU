//
// Created by Nguyen Tran on 6/21/18.
//

#ifndef PCMS_CONSTANTS_H
#define PCMS_CONSTANTS_H

#define MY_CONSTANT(constant_type, constant_name, constant_value)\
public:\
  static constant_type const &constant_name(){\
    static constant_type constant_name##_{constant_value};\
    return constant_name##_;\
  }

class Constants {
  MY_CONSTANT(int, DAYS_IN_YEAR, 365)
  MY_CONSTANT(double, PI, 3.14159265358979)

  //  static std::chrono::hours const &ONE_DAY() {
  //    static std::chrono::hours one_day_{24};
  //    return one_day_;
  //  }
};

#endif //PCMS_CONSTANTS_H
