/* 
 * File:   Event.cpp
 * Author: nguyentran
 * 
 * Created on May 3, 2013, 3:13 PM
 */

#include "Event.h"
#include  "../Core/Dispatcher.h"

Event::Event() = default;

Event::~Event() {
  if (dispatcher!=nullptr) {
    dispatcher->remove(this);
  }
  dispatcher = nullptr;
  scheduler = nullptr;
}

void Event::perform_execute() {
  // Return if there is nothing to do
  if (!executable) { return; }

  // Execute the update event attached to the dispatcher
  if (dispatcher != nullptr) {
    dispatcher->update();
  }

  // Execute the event
  execute();    

  // Update the dispatcher
  if (dispatcher != nullptr) {
    dispatcher->remove(this);
    dispatcher = nullptr;
  }

  // Disable ourselves
  executable = false;
}
