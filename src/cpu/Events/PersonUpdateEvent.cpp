//
// Created by kient on 8/4/2023.
//

#include "../Core/Scheduler.h"
#include "../../cpu/Model.h"
#include "../Population/Population.cuh"
#include "PersonUpdateEvent.h"

PersonUpdateEvent::PersonUpdateEvent() = default;

PersonUpdateEvent::~PersonUpdateEvent() = default;

void PersonUpdateEvent::schedule_event(Scheduler *scheduler, const int &time) {
    if (scheduler!=nullptr) {
        auto *person_update_event = new PersonUpdateEvent();
        person_update_event->dispatcher = nullptr;
        person_update_event->time = time;

        scheduler->schedule_population_event(person_update_event);
    }
}

std::string PersonUpdateEvent::name() {
    return "Person Update Event";
}

void PersonUpdateEvent::execute() {
    //Update population here
    std::cout << "Person Update Event executed at time: " << time<< std::endl;
    Model::POPULATION->update();
}
