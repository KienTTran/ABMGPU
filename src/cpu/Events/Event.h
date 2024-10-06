/* 
 * File:   Event.h
 * Author: nguyentran
 *
 * Created on May 3, 2013, 3:13 PM
 */

#ifndef EVENT_H
#define    EVENT_H

#include "../Population/Properties/IndexHandler.h"
#include "../Core/PropertyMacro.h"
#include <string>

class Dispatcher;

class Scheduler;

class Event : public IndexHandler {
 DISALLOW_COPY_AND_ASSIGN(Event)

 DISALLOW_MOVE(Event)

 public:
  Scheduler *scheduler{nullptr};
  Dispatcher *dispatcher{nullptr};
  bool executable{false};
  int time{-1};

    Event();

  //    Event(const Event& orig);
    virtual ~Event();

  void perform_execute();

  virtual std::string name() = 0;

 private:
  virtual void execute() = 0;

};

#endif    /* EVENT_H */
