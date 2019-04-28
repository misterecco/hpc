#pragma once

#include <cstdio>
#include <limits>
#include <vector>
#include <assert.h>
#include <cooperative_groups.h>

#include "config.h"
#include "errors.h"
#include "hashtable.cuh"
#include "lock.cuh"
#include "memory.h"
#include "queues.cuh"

using namespace cooperative_groups;

#define BLOCKS 8
#define THREADS_PER_BLOCK 128
#define TABLE_SIZE (96 * 1024 * 1024)
#define HASH_TABLE_SIZE (32 * 1024 * 1024)
#define UNROLLING_ROUNDS 1
#define ALLOC_PACK 8

template<typename Problem, typename State, typename QState>
class Solver {
 public:
  Solver(const Config& config);
  void solve();
  __device__ void findSolution();
  ~Solver();

  Problem* problem;
  Lock lockCuda;
  Queues<QState> queues;
  Hashtable<State> hashtable;

  State* statesHost = nullptr;
  State* statesCuda = nullptr;
  int* statesSizeCuda = nullptr;
  int* finishedCuda = nullptr;
  int* bestStateCuda = nullptr;
  int* bestBlockStatesCuda = nullptr;
  int* endConditionCuda = nullptr;
  int bestState = -1;

  __device__ void lock();
  __device__ void unlock();
  __device__ void extract(int& bestState, int freeSlots[ALLOC_PACK]);
};

template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::Solver(const Config& config)
 : problem(new Problem(config)),
   hashtable(HASH_TABLE_SIZE),
   queues(8 * 8192, BLOCKS * THREADS_PER_BLOCK) { }

template<typename Problem, typename State, typename QState>
Solver<Problem, State, QState>::~Solver() {
  delete problem;
  maybeFree(statesHost);
  maybeCudaFree(statesCuda);
  maybeCudaFree(statesSizeCuda);
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

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::extract(int& bestState,
          int freeSlots[ALLOC_PACK]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // int blockOffset = blockIdx.x * blockDim.x;

  int usedSlots[ALLOC_PACK] = {-1, -1, -1, -1, -1, -1, -1, -1};
  __shared__ int usedSlotsBlock[ALLOC_PACK * THREADS_PER_BLOCK];
  __shared__ int usbIdx;

  if (threadIdx.x == 0) {
    usbIdx = 0;
  }
  __syncthreads();

  int requestCount = 0;
  for (int i = 0; i < ALLOC_PACK; i++) {
    if (freeSlots[i] != -1) break;
    requestCount++;
  }

  if (requestCount > 0) {
    int firstFreeSlot = atomicAdd(statesSizeCuda, requestCount);
    assert(firstFreeSlot < TABLE_SIZE);

    for (int i = 0; i < requestCount; i++) {
      freeSlots[i] = firstFreeSlot + i;
    }
  }
  if (idx == 0) {
    printf("statesSizeCuda: %d\n", *statesSizeCuda);
  }

  // TODO: should I keep this loop? If so - adjust ALLOC_PACK
  int expandedStatesCount = min(UNROLLING_ROUNDS, queues.size(idx))
                            * Problem::statesUnrolledPerStep;

  for (int i = 0; i < UNROLLING_ROUNDS && !queues.empty(idx); i++) {
    QState qst = queues.pop(idx);
    State st = statesCuda[qst.stateNumber];
    // printf("expanding: stNum: %d ", qst.stateNumber);
    // st.print(problem->n);
    problem->expand(statesCuda, st, qst.stateNumber, hashtable, freeSlots,
                    usedSlots, bestState);
  }

  /*
  printf("usedSlots:");
  for (int i = 0; i < ALLOC_PACK; i++) {
    printf("%d ", usedSlots[i]);
  }
  printf("\n");
  */

  int usedSlotsCount = 0;
  for (int i = 0; i < expandedStatesCount; i++) {
    if (usedSlots[i] == -1) break;
    usedSlotsCount++;
    hashtable.deduplicate(statesCuda, usedSlots[i]);
  }

  int firstInd = atomicAdd(&usbIdx, usedSlotsCount);

  for (int i = 0; i < usedSlotsCount; i++) {
    usedSlotsBlock[firstInd + i] = usedSlots[i];
  }

  /*
  printf("Used slots block: ");
  for (int i = 0; i < usbIdx; i++) {
    printf("%d ", usedSlotsBlock[i]);
  }
  printf("\n");
  */

  lock();

  // printf("after dedup\n");

  // TODO: think about better state distribution
  int targetBlock = (blockIdx.x + threadIdx.x) % BLOCKS;
  for (int i = threadIdx.x; i < usbIdx; i += THREADS_PER_BLOCK) {
    const int stNum = usedSlotsBlock[i];

    State& newState = statesCuda[stNum];
    if (newState.isNull()) continue;

    // newState.print(problem->n);

    QState queueEntry {
      .f = newState.f,
      .stateNumber = stNum,
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
  int allocatedFreeSlots[ALLOC_PACK];

  for (int i = 0; i < ALLOC_PACK; i++) {
    allocatedFreeSlots[i] = -1;
  }

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
    extract(bestTargetStates[ti], allocatedFreeSlots);

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

      if (queues.empty(gti) || queues.top(gti).f >= statesCuda[*bestStateCuda].f) {
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
          break;
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
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  HANDLE_ERROR(cudaMalloc(&statesCuda, sizeof(State) * TABLE_SIZE));
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

  queues.init(initQState);
  hashtable.init();

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

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Execution time: %.0f\n", elapsedTime);

  if (bestState == -1) {
    printf("Unreachable\n");
    return;
  }

  problem->printSolution(statesHost, bestState);
}
