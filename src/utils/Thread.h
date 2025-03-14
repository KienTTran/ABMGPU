//
// Created by kient on 6/25/2023.
//

#ifndef MASS_THREAD_H
#define MASS_THREAD_H

#include <mutex>


class Thread {
public:
    Thread();
    ~Thread();
    static Thread& getInstance() // Singleton is accessed via getInstance()
    {
        static Thread instance; // lazy singleton, instantiated on first use
        return instance;
    }
public:
};


#endif //MASS_THREAD_H
