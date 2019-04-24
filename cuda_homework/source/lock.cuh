#pragma once

class Lock {
 public:
   Lock() {
     int h_state = 0;
     // TODO: check errors
     cudaMalloc(&d_state, sizeof(int));
     cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice);
   }

   ~Lock(void) { 
     cudaFree(d_state); 
   }

   __device__ void lock(void) { while (atomicCAS(d_state, 0, 1) != 0); }

   __device__ void unlock(void) { atomicExch(d_state, 0); }

 private:
  int *d_state;
};

