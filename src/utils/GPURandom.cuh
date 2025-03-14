//
// Created by kient on 6/17/2023.
//

#ifndef MASS_GPURANDOM_CUH
#define MASS_GPURANDOM_CUH

#include <curand_kernel.h>

class GPURandom {
public:
    curandState *d_states;
    int n_threads;
    int n_blocks;
public:
    GPURandom();
    ~GPURandom();
    static GPURandom& getInstance() // Singleton is accessed via getInstance()
    {
        static GPURandom instance; // lazy singleton, instantiated on first use
        return instance;
    }

    void init(int n, long seed);
    void free();
};


#endif //MASS_GPURANDOM_CUH
