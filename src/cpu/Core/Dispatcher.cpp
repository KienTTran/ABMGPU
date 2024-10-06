/* 
 * File:   Dispatcher.cpp
 * Author: nguyentran
 * 
 * Created on May 3, 2013, 3:46 PM
 */

#include "Dispatcher.h"
#include "../Population/Properties/IndexHandler.h"
#include "../Events/Event.h"
#include "../Helpers/ObjectHelpers.h"

Dispatcher::Dispatcher() : events_(nullptr) {}

void Dispatcher::init() {
  events_ = new EventPtrVector();
}

Dispatcher::~Dispatcher() {
  Dispatcher::clear_events();
  ObjectHelpers::delete_pointer<EventPtrVector>(events_);
}

void Dispatcher::add(Event *event) {
  events_->push_back(event);
  event->IndexHandler::set_index(events_->size() - 1);
}

void Dispatcher::remove(Event *event) {
  events_->back()->IndexHandler::set_index(event->IndexHandler::index());

  //    std::cout << "1"<<event->IndexHandler::index()<< std::endl;
  events_->at(event->IndexHandler::index()) = events_->back();
  //    std::cout << "2"<< std::endl;

  events_->pop_back();
  event->IndexHandler::set_index(-1);
}

void Dispatcher::clear_events() {
  if (events_==nullptr) return;
  if (events_->empty()) return;
  //    std::cout << "Clear event"<< std::endl;

  for (auto &event : *events_) {
    event->dispatcher = nullptr;
    event->executable = false;
  }

  events_->clear();
}

void Dispatcher::update() {
  events_->shrink_to_fit();
}