//
// Created by kient on 8/4/2023.
//

#include "PersonUpdateRenderEvent.h"


#include "../Core/Scheduler.h"
#include "../Model.h"
#include "../Population/Population.cuh"

PersonUpdateRenderEvent::PersonUpdateRenderEvent() = default;

PersonUpdateRenderEvent::~PersonUpdateRenderEvent() = default;

void PersonUpdateRenderEvent::schedule_event(Scheduler *scheduler, const int &time) {
    if (scheduler!=nullptr) {
        auto *person_update_event = new PersonUpdateRenderEvent();
        person_update_event->dispatcher = nullptr;
        person_update_event->time = time;

        scheduler->schedule_population_event(person_update_event);
    }
}

std::string PersonUpdateRenderEvent::name() {
    return "Person Update Render Event";
}

void PersonUpdateRenderEvent::execute() {
    //Update population here
    std::cout << "Person Update Render Event executed at time: " << time<< std::endl;
    Model::POPULATION->updateRender();
}