//
// Created by kient on 8/4/2023.
//

#ifndef MASS_PERSONUPDATEEVENT_H
#define MASS_PERSONUPDATEEVENT_H


#include "Event.h"

class PersonUpdateEvent: public Event {

DISALLOW_COPY_AND_ASSIGN(PersonUpdateEvent)

public:
    PersonUpdateEvent();

    virtual ~PersonUpdateEvent();

    static void schedule_event(Scheduler *scheduler, const int &time);

    std::string name() override;

private:
    void execute() override;


};


#endif //MASS_PERSONUPDATEEVENT_H
