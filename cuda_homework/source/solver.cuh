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

template<typename Problem, typename State, typename QState>
class Solver {
 public:
  Solver(const Config& config)
  : problem(new Problem(config)),
    hashtable(Problem::kHashTableSize),
    kTableSize(Problem::kTableSize),
    kBlocks(Problem::kBlocks),
    kThreadsPerBlock(Problem::kThreadsPerBlock),
    queues(Problem::kQueueSize, Problem::kBlocks * Problem::kThreadsPerBlock) { }

  void solve();
  __device__ void findSolution();
  ~Solver();

  Problem* problem;
 private:
  const int kBlocks;
  const int kThreadsPerBlock;
  const int kTableSize;

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
  __device__ void extract(int& bestState, int freeSlots[Problem::kStatesUnrolledPerRound]);
};


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
          int freeSlots[Problem::kStatesUnrolledPerRound]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int usedSlots[Problem::kStatesUnrolledPerRound];
  for (int i = 0; i < Problem::kStatesUnrolledPerRound; i++) {
    usedSlots[i] = -1;
  }
  __shared__ int usedSlotsBlock[Problem::kStatesUnrolledPerRound * Problem::kThreadsPerBlock];
  __shared__ int usbIdx;

  if (threadIdx.x == 0) {
    usbIdx = 0;
  }
  __syncthreads();

  int requestCount = 0;
  for (int i = 0; i < Problem::kStatesUnrolledPerRound; i++) {
    if (freeSlots[i] != -1) break;
    requestCount++;
  }

  if (requestCount > 0) {
    int firstFreeSlot = atomicAdd(statesSizeCuda, requestCount);
    assert(firstFreeSlot < kTableSize);

    for (int i = 0; i < requestCount; i++) {
      freeSlots[i] = firstFreeSlot + i;
    }
  }

  int expandedStatesCount = min(Problem::kUnrollingRounds, queues.size(idx))
                            * Problem::kStatesUnrolledPerStep;

  for (int i = 0; i < Problem::kUnrollingRounds && !queues.empty(idx); i++) {
    QState qst = queues.pop(idx);
    State st = statesCuda[qst.stateNumber];
    problem->expand(statesCuda, st, qst.stateNumber, hashtable, freeSlots,
                    usedSlots, bestState);
  }

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

  lock();

  int targetBlock = (blockIdx.x + threadIdx.x) % kBlocks;
  for (int i = threadIdx.x; i < usbIdx; i += kThreadsPerBlock) {
    const int stNum = usedSlotsBlock[i];

    State& newState = statesCuda[stNum];
    if (newState.isNull()) continue;

    QState queueEntry {
      .f = newState.f,
      .stateNumber = stNum,
    };

    queues.push(kThreadsPerBlock * targetBlock + threadIdx.x, queueEntry);

    targetBlock = (targetBlock + 1) % kBlocks;
  }

  unlock();
}

template<typename Problem, typename State, typename QState>
__device__ void Solver<Problem, State, QState>::findSolution() {
  grid_group grid = this_grid();
  int gti = threadIdx.x + blockIdx.x * blockDim.x;
  int ti = threadIdx.x;
  int bi = blockIdx.x;

  __shared__ int bestTargetStates[Problem::kThreadsPerBlock];
  __shared__ int endCondition[Problem::kThreadsPerBlock];
  int allocatedFreeSlots[Problem::kStatesUnrolledPerRound];

  for (int i = 0; i < Problem::kStatesUnrolledPerRound; i++) {
    allocatedFreeSlots[i] = -1;
  }

  bestTargetStates[ti] = -1;
  if (ti == 0) {
    bestBlockStatesCuda[bi] = -1;
  }

  while(*finishedCuda == 0) {
    extract(bestTargetStates[ti], allocatedFreeSlots);

    __syncthreads();

    if (ti == 0) {
      for (int i = 0; i < kThreadsPerBlock; i++) {
        int bestPerBlock = bestBlockStatesCuda[bi];
        if (bestPerBlock == -1 || (bestTargetStates[i] != -1
            && statesCuda[bestTargetStates[i]].f < statesCuda[bestPerBlock].f)) {
          bestBlockStatesCuda[bi] = bestTargetStates[i];
        }
      }
    }

    grid.sync();

    if (ti == 0 && bi == 0) {
      for (int i = 0; i < kBlocks; i++) {
        int bestPerBlock = bestBlockStatesCuda[i];
        if (bestPerBlock != -1 && (*bestStateCuda == -1
            || statesCuda[bestPerBlock].f < statesCuda[*bestStateCuda].f)) {
          *bestStateCuda = bestPerBlock;
        }
      }
    }

    grid.sync();

    if (*bestStateCuda != -1) {
      if (queues.empty(gti) || queues.top(gti).f >= statesCuda[*bestStateCuda].f) {
        endCondition[ti] = 1;
      }

      __syncthreads();

      if (ti == 0) {
        int finished = 1;
        for (int i = 0; i < kThreadsPerBlock; i++) {
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
        for (int i = 0; i < kBlocks; i++) {
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
      int finished = 1;
      for (int i = 0; i < kBlocks * kThreadsPerBlock; i++) {
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

  HANDLE_ERROR(cudaMalloc(&statesCuda, sizeof(State) * kTableSize));
  HANDLE_ERROR(cudaMalloc(&statesSizeCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&finishedCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&bestStateCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&bestBlockStatesCuda, sizeof(int) * kBlocks));
  HANDLE_ERROR(cudaMalloc(&endConditionCuda, sizeof(int) * kBlocks));

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  statesHost = (State*) HANDLE_NULLPTR(malloc(sizeof(State) * kTableSize));

  for (int i = 0; i < kTableSize; i++) {
    statesHost[i] = State();
  }

  State initState = problem->getInitState();
  QState initQState = problem->getInitQState();

  queues.init(initQState);
  hashtable.init();

  statesHost[0] = initState;

  HANDLE_ERROR(cudaMemcpy(statesCuda, statesHost, sizeof(State) * kTableSize,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(bestStateCuda, &bestState, sizeof(int),
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(endConditionCuda, 0, sizeof(int) * kBlocks));
  HANDLE_ERROR(cudaMemset(statesSizeCuda, 1, 1));
  HANDLE_ERROR(cudaMemset(finishedCuda, 0, sizeof(int)));


  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  void* kernelArgs[] = {(void*) this, (void*) problem};
  HANDLE_ERROR(cudaLaunchCooperativeKernel(
    (void*) kernel<Problem, State, QState>, kBlocks,
               kThreadsPerBlock, kernelArgs));

  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  HANDLE_ERROR(cudaMemcpy(statesHost, statesCuda, sizeof(State) * kTableSize,
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
  }

  problem->printSolution(statesHost, bestState, elapsedTime);
}
