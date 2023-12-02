//
// Created by kient on 6/17/2023.
//

#ifndef MASS_GPUUTILS_CUH
#define MASS_GPUUTILS_CUH

#include <iostream>

#define checkCudaErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        printf("checkCudaErr: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class GPUUtils {

};


#endif //MASS_GPUUTILS_CUH
