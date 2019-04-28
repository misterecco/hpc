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
#include "memory.h"

using namespace cooperative_groups;

#define BLOCKS 2
#define THREADS_PER_BLOCK 8
#define TABLE_SIZE (64 * 1024 * 1024)
#define HASH_TABLE_SIZE (64 * 1024 * 1024)

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
  int* hashtableCuda = nullptr;
  int* finishedCuda = nullptr;
  int* bestStateCuda = nullptr;
  int* bestBlockStatesCuda = nullptr;
  int* endConditionCuda = nullptr;
  int bestState = -1;
  Lock lockCuda;
  Queues<8 * 8192, BLOCKS * THREADS_PER_BLOCK, QState> queues;

  __device__ void lock();
  __device__ void unlock();
  __device__ void extract(int& bestState);
};

template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::Solver(const Config& config)
 : problem(new Problem(config)) { }

template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::~Solver() {
  delete problem;
  maybeFree(statesHost);
  maybeCudaFree(statesCuda);
  maybeCudaFree(statesSizeCuda);
  maybeCudaFree(hashtableCuda);
  maybeCudaFree(finishedCuda);
  maybeCudaFree(bestStateCuda);
  maybeCudaFree(bestBlockStatesCuda);
  maybeCudaFree(endConditionCuda);
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

#define UNROLLING_ROUNDS 1

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
          Problem::statesUnrolledPerStep
          * min(UNROLLING_ROUNDS, queues.size(i - 1 + blockOffset));
    }
    *statesSizeCuda = offsets[THREADS_PER_BLOCK - 1] +
        Problem::statesUnrolledPerStep
        * min(UNROLLING_ROUNDS, queues.size(THREADS_PER_BLOCK - 1 + blockOffset));
    maxOffset = *statesSizeCuda;
    // printf("blockIdx: %d: ", blockIdx.x);
    // for (int i = 0; i < THREADS_PER_BLOCK; i++) {
    //   printf("%d ", offsets[i]);
    // }
    // printf("\n");
  }

  unlock();

  int firstFreeSlot = offsets[threadIdx.x];

  int expandedStatesCount = min(UNROLLING_ROUNDS, queues.size(idx))
                            * Problem::statesUnrolledPerStep;

  for (int i = 0; i < UNROLLING_ROUNDS && !queues.empty(idx); i++) {
    QState qst = queues.pop(idx);
    State st = statesCuda[qst.stateNumber];
    problem->expand(statesCuda, st, qst.stateNumber, firstFreeSlot, bestState);
    firstFreeSlot += Problem::statesUnrolledPerStep;
  }

  for (int i = 0; i < expandedStatesCount; i++) {
    deduplicate(statesCuda, hashtableCuda, offsets[threadIdx.x] + i, HASH_TABLE_SIZE);
  }

  lock();

  // TODO: think about better state distribution
  int targetBlock = (blockIdx.x + threadIdx.x) % BLOCKS;
  for (int i = offsets[0] + threadIdx.x; i < maxOffset; i += THREADS_PER_BLOCK) {
    State& newState = statesCuda[i];
    if (newState.isNull()) continue;

    // newState.print(problem->n);

    QState queueEntry {
      .f = newState.f,
      .stateNumber = i,
    };

    queues.push(THREADS_PER_BLOCK * targetBlock + threadIdx.x, queueEntry);

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

      if (queues.empty(gti) || queues.top(gti).f > statesCuda[*bestStateCuda].f) {
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
        printf("i: %d, queueSize: %d\n", i, queues.size(i));
        if (queues.size(i) > 0) {
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

  int* hashtableHost = (int*) HANDLE_NULLPTR(malloc(sizeof(int) * HASH_TABLE_SIZE));
  for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    hashtableHost[i] = -1;
  }
  HANDLE_ERROR(cudaMemcpy(hashtableCuda, hashtableHost, sizeof(int) * HASH_TABLE_SIZE,
        cudaMemcpyHostToDevice));
  free(hashtableHost);

  for (int i = 0; i < TABLE_SIZE; i++) {
    statesHost[i] = State();
  }

  State initState = problem->getInitState();
  QState initQState = problem->getInitQState();

  queues.init(initQState);

  statesHost[0] = initState;

  HANDLE_ERROR(cudaMemcpy(statesCuda, statesHost, sizeof(State) * TABLE_SIZE,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(bestStateCuda, &bestState, sizeof(int),
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(endConditionCuda, 0, sizeof(int) * BLOCKS));
  HANDLE_ERROR(cudaMemset(statesSizeCuda, 1, 1));
  HANDLE_ERROR(cudaMemset(finishedCuda, 0, sizeof(int)));


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
