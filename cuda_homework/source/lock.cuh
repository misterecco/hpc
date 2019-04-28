#pragma once

#include "errors.h"

class Lock {
 public:
  Lock() {
    int h_state = 0;
    HANDLE_ERROR(cudaMalloc(&d_state, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice));
  }

  ~Lock(void) {
    cudaFree(d_state);
  }

  __device__ void lock(void) { while (atomicCAS(d_state, 0, 1) != 0); }

  __device__ void unlock(void) { atomicExch(d_state, 0); }

 private:
  int *d_state;
};

