#include <time.h>
#include <stdio.h>

//#define RADIUS 3;
//#define NUM_ELEMENTS 1000;
#define NUM_THREADS 1024 

static void handleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) {
  //PUT YOUR CODE HERE
  int tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  for (int i = -RADIUS; i <= RADIUS; i++) {
    out[tid] += in[(tid + i + NUM_ELEMENTS) % NUM_ELEMENTS];
  }
  //--DONE
}

void cpu_stencil_1d(int *in, int *out) {
  //PUT YOUR CODE HERE
  int cs = 0;
  for (int i = -RADIUS; i <= RADIUS; i++) {
    cs += in[(i + NUM_ELEMENTS) % NUM_ELEMENTS];
  }
  out[0] = cs;
  for (int i = 1; i < NUM_ELEMENTS; i++) {
    cs = cs - in[(i - 1 - RADIUS + NUM_ELEMENTS) % NUM_ELEMENTS] + in[(i + RADIUS + NUM_ELEMENTS) % NUM_ELEMENTS];
    out[i] = cs;
  }
  //--DONE
}

int main(int argc, char** argv) {
  //PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
  int *in = (int*) malloc(NUM_ELEMENTS * sizeof(int));
  int *out = (int*) malloc(NUM_ELEMENTS * sizeof(int));
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    in[i] = i;
  }
  //--DONE

  
  cudaEvent_t start, chkpt1, chkpt2, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&chkpt1);
  cudaEventCreate(&chkpt2);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  //PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
  int *devIn, *devOut;
  cudaMalloc((void**)&devIn, NUM_ELEMENTS * sizeof(int));
  cudaMalloc((void**)&devOut, NUM_ELEMENTS * sizeof(int));
  cudaMemcpy(devIn, in, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
  //--DONE

  cudaEventRecord(chkpt1, 0);

  //PUT YOUR CODE HERE - KERNEL EXECUTION
  stencil_1d<<<(NUM_ELEMENTS + NUM_THREADS)/NUM_THREADS,NUM_THREADS>>>(devIn, devOut);
  //--DONE

  cudaCheck(cudaPeekAtLastError());

  cudaEventRecord(chkpt2, 0);

  //PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
  cudaMemcpy(out, devOut, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
  //--DONE

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, chkpt1);
  printf("Copy to GPU:  %3.1f ms\n", elapsedTime);
  cudaEventElapsedTime( &elapsedTime, chkpt1, chkpt2);
  printf("Kernel execution:  %3.1f ms\n", elapsedTime);
  cudaEventElapsedTime( &elapsedTime, chkpt2, stop);
  printf("Copy from GPU:  %3.1f ms\n", elapsedTime);
  cudaEventElapsedTime( &elapsedTime, start, stop);
  printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(chkpt1);
  cudaEventDestroy(chkpt2);
  cudaEventDestroy(stop);

  //PUT YOUR CODE HERE - FREE DEVICE MEMORY  
  cudaFree(devIn);
  cudaFree(devOut);
  //--DONE

  struct timespec cpu_start, cpu_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

  cpu_stencil_1d(in, out);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.1f ms\n", result);

  free(in);
  free(out);
  
  return 0;
}


