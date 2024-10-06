//
// Created by kient on 8/4/2023.
//

#ifndef MASS_PERSONUPDATERENDEREVENT_H
#define MASS_PERSONUPDATERENDEREVENT_H


#include "Event.h"

class PersonUpdateRenderEvent: public Event {

DISALLOW_COPY_AND_ASSIGN(PersonUpdateRenderEvent)

public:
    PersonUpdateRenderEvent();

    virtual ~PersonUpdateRenderEvent();

    static void schedule_event(Scheduler *scheduler, const int &time);

    std::string name() override;

private:
    void execute() override;

};


#endif //MASS_PERSONUPDATERENDEREVENT_H
