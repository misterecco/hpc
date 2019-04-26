#pragma once

#include <cstdio>
#include <limits>
#include <vector>
#include <assert.h>
#include <cooperative_groups.h>

#include "config.h"
#include "errors.h"
#include "hashtable.cuh"
#include "heap.cuh"
#include "lock.cuh"

using namespace cooperative_groups;

#define BLOCKS 2
#define THREADS_PER_BLOCK 8
#define QUEUES_PER_BLOCK 8
#define TABLE_SIZE 64 * 1024 * 1024
#define HASH_TABLE_SIZE 1024 * 1024

template<typename Problem, typename State, typename QState>
class Solver {
 public:
  Solver(const Config& config);
  void solve();
  __device__ void findSolution();
  ~Solver();

  Problem* problem;

  State* statesHost = nullptr;
  State* statesCuda = nullptr;
  int* statesSizeCuda = nullptr;
  QState* queuesCuda = nullptr;
  int* queueSizesCuda = nullptr;
  int* hashtableCuda = nullptr;
  int* finishedCuda = nullptr;
  int* bestStateCuda = nullptr;
  int* bestBlockStatesCuda = nullptr;
  int* endConditionCuda = nullptr;
  int bestState = -1;
  Lock lockCuda;

  __device__ void lock();
  __device__ void unlock();
  __device__ void extract(int& bestState);
};

template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::Solver(const Config& config)
 : problem(new Problem(config)) { }

// TODO: free all memory
template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::~Solver() {
  delete problem;

  if (statesHost != nullptr) {
    free(statesHost);
  }

  if (statesCuda != nullptr) {
    cudaFree(statesCuda);
  }

  if (queuesCuda != nullptr) {
    cudaFree(queuesCuda);
  }

  if (queueSizesCuda != nullptr) {
    cudaFree(queueSizesCuda);
  }

  if (hashtableCuda != nullptr) {
    cudaFree(hashtableCuda);
  }
}

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::lock() {
  if (threadIdx.x == 0) {
    lockCuda.lock();
  }
  __syncthreads();
}

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::unlock() {
  if (threadIdx.x == 0) {
    lockCuda.unlock();
  }
  __syncthreads();
}

#define STATES_UNROLLED_PER_STEP 1

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::extract(int& bestState) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int blockOffset = blockIdx.x * blockDim.x;

  __shared__ int offsets[THREADS_PER_BLOCK];
  __shared__ int maxOffset;

  lock();

  if (threadIdx.x == 0) {
    offsets[0] = *statesSizeCuda;
    for (int i = 1; i < THREADS_PER_BLOCK; i++) {
      offsets[i] = offsets[i-1] +
        8 * min(STATES_UNROLLED_PER_STEP, queueSizesCuda[i - 1 + blockOffset]);
    }
    *statesSizeCuda = offsets[THREADS_PER_BLOCK - 1] +
                      8 * min(STATES_UNROLLED_PER_STEP, queueSizesCuda[THREADS_PER_BLOCK - 1 + blockOffset]);
    maxOffset = *statesSizeCuda;
    // printf("blockIdx: %d: ", blockIdx.x);
    // for (int i = 0; i < THREADS_PER_BLOCK; i++) {
    //   printf("%d ", offsets[i]);
    // }
    // printf("\n");
  }

  unlock();

  int firstFreeSlot = offsets[threadIdx.x];

  int expandedStatesCount = min(STATES_UNROLLED_PER_STEP, queueSizesCuda[idx]) * 8;

  for (int i = 0; i < STATES_UNROLLED_PER_STEP && !empty(queueSizesCuda[idx]); i++) {
    QState qst = pop(queuesCuda + HEAP_SIZE * idx, queueSizesCuda[idx]);
    State st = statesCuda[qst.stateNumber];
    problem->expand(statesCuda, st, qst.stateNumber, firstFreeSlot, bestState);
    firstFreeSlot += 8;
  }

  for (int i = 0; i < expandedStatesCount; i++) {
    deduplicate(statesCuda, hashtableCuda, offsets[threadIdx.x] + i);
  }

  lock();

  int targetBlock = (blockIdx.x + threadIdx.x) % BLOCKS;
  for (int i = offsets[0] + threadIdx.x; i < maxOffset; i += THREADS_PER_BLOCK) {
    State& newState = statesCuda[i];
    if (newState.isNull()) continue;

    newState.print(problem->n);

    QState queueEntry {
      .f = newState.f,
      .stateNumber = i,
    };

    push(queuesCuda + HEAP_SIZE * (THREADS_PER_BLOCK * targetBlock +
          threadIdx.x), queueSizesCuda[THREADS_PER_BLOCK * targetBlock +
        threadIdx.x], queueEntry);

    targetBlock = (targetBlock + 1) % BLOCKS;
  }

  unlock();
}

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::findSolution() {
  grid_group grid = this_grid();
  int gti = threadIdx.x + blockIdx.x * blockDim.x;
  int ti = threadIdx.x;
  int bi = blockIdx.x;

  __shared__ int bestTargetStates[THREADS_PER_BLOCK];
  __shared__ int endCondition[THREADS_PER_BLOCK];

  bestTargetStates[ti] = -1;
  if (ti == 0) {
    bestBlockStatesCuda[bi] = -1;
  }

  /*
  if (ti == 0 && bi == 0) {
    printf("Target state node: %d\n", problem->endNodeCuda);
  }
  */

  while(*finishedCuda == 0) {
    extract(bestTargetStates[ti]);

    __syncthreads();

    if (ti == 0) {
      for (int i = 0; i < THREADS_PER_BLOCK; i++) {
        int bestPerBlock = bestBlockStatesCuda[bi];
        if (bestPerBlock == -1 || (bestTargetStates[i] != -1
            && statesCuda[bestTargetStates[i]].f < statesCuda[bestPerBlock].f)) {
          bestBlockStatesCuda[bi] = bestTargetStates[i];
        }
      }
    }

    grid.sync();

    if (ti == 0 && bi == 0) {
      for (int i = 0; i < BLOCKS; i++) {
        int bestPerBlock = bestBlockStatesCuda[i];
        if (bestPerBlock != -1 && (*bestStateCuda == -1
            || statesCuda[bestPerBlock].f < statesCuda[*bestStateCuda].f)) {
          *bestStateCuda = bestPerBlock;
        }
      }
    }

    grid.sync();

    if (*bestStateCuda != -1) {
      if (ti == 0 && bi == 0) {
        printf("Best state: ");
        statesCuda[*bestStateCuda].print(problem->n);
      }

      if (empty(queueSizesCuda[gti])
            || top(queuesCuda + HEAP_SIZE * gti, queueSizesCuda[gti]).f >
            statesCuda[*bestStateCuda].f) {
        endCondition[ti] = 1;
      }

      __syncthreads();

      if (ti == 0) {
        int finished = 1;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
          if (!endCondition[i]) {
            finished = 0;
            break;
          }
        }
        endConditionCuda[bi] = finished;
      }

      grid.sync();

      if (ti == 0 && bi == 0) {
        int finished = 1;
        for (int i = 0; i < BLOCKS; i++) {
          if (!endConditionCuda[i]) {
            finished = 0;
            break;
          }
        }
        *finishedCuda = finished;
      }

      grid.sync();

      if (*finishedCuda) {
        break;
      }
    }

    if (ti == 0 && bi == 0) {
      /*
      printf("Hash table: \n");
      for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        printf("%d ", hashtableCuda[i]);
      }
      printf("\n");
      */

      int finished = 1;
      for (int i = 0; i < BLOCKS * THREADS_PER_BLOCK; i++) {
        printf("i: %d, queueSize: %d\n", i, queueSizesCuda[i]);
        if (queueSizesCuda[i] > 0) {
          finished = 0;
          // break;
        }
      }
      if (finished) {
        *finishedCuda = 1;
      }
    }

    grid.sync();
  }
}


template<typename Problem, typename State, typename QState>
__global__ void kernel(
    Solver<Problem, State, QState> solver, Problem problem) {
  solver.problem = &problem;
  solver.findSolution();
}


template<typename Problem, typename State, typename QState>
void Solver<Problem, State, QState>::solve() {
  HANDLE_ERROR(cudaMalloc(&statesCuda, sizeof(State) * TABLE_SIZE));
  HANDLE_ERROR(cudaMalloc(&queuesCuda,
        sizeof(QState) * THREADS_PER_BLOCK * BLOCKS * HEAP_SIZE));
  HANDLE_ERROR(cudaMalloc(&queueSizesCuda, sizeof(int) * BLOCKS *
        QUEUES_PER_BLOCK));
  HANDLE_ERROR(cudaMalloc(&hashtableCuda, sizeof(int) * HASH_TABLE_SIZE));
  HANDLE_ERROR(cudaMalloc(&statesSizeCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&finishedCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&bestStateCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&bestBlockStatesCuda, sizeof(int) * BLOCKS));
  HANDLE_ERROR(cudaMalloc(&endConditionCuda, sizeof(int) * BLOCKS));

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());
  printf("cudaMallocs complete!\n");

  statesHost = (State*) HANDLE_NULLPTR(malloc(sizeof(State) * TABLE_SIZE));

  for (int i = 0; i < TABLE_SIZE; i++) {
    statesHost[i] = State();
  }

  State initState = problem->getInitState();
  QState initQState = problem->getInitQState();

  statesHost[0] = initState;

  HANDLE_ERROR(cudaMemcpy(statesCuda, statesHost, sizeof(State) * TABLE_SIZE,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(queuesCuda, &initQState, sizeof(QState),
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(bestStateCuda, &bestState, sizeof(int),
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(queueSizesCuda, 0, sizeof(int) * BLOCKS *
        QUEUES_PER_BLOCK));
  HANDLE_ERROR(cudaMemset(endConditionCuda, 0, sizeof(int) * BLOCKS));
  HANDLE_ERROR(cudaMemset(statesSizeCuda, 1, 1));
  HANDLE_ERROR(cudaMemset(finishedCuda, 0, sizeof(int)));

  int one = 1;
  HANDLE_ERROR(cudaMemcpy(queueSizesCuda, &one, sizeof(int),
        cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());
  printf("Memory transfers complete!\n");

  // TODO: add timing

  void* kernelArgs[] = {(void*) this, (void*) problem};
  HANDLE_ERROR(cudaLaunchCooperativeKernel(
    (void*) kernel<Problem, State, QState>, BLOCKS,
               THREADS_PER_BLOCK, kernelArgs));

  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  printf("Computation finished!\n");

  HANDLE_ERROR(cudaMemcpy(statesHost, statesCuda, sizeof(State) * TABLE_SIZE,
        cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&bestState, bestStateCuda, sizeof(int),
        cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  printf("bestState: %d\n", bestState);

  if (bestState == -1) {
    printf("Unreachable\n");
    return;
  }

  problem->expandSolution(statesHost, bestState);
}


