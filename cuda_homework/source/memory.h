#pragma once

#include <cstdio>
#include <cstdlib>

inline void maybeFree(void* ptr) {
    if (ptr == nullptr) {
        free(ptr);
    }
}

inline void maybeCudaFree(void* ptr) {
    if (ptr == nullptr) {
        cudaFree(ptr);
    }
}