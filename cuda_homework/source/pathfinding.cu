#include <assert.h>
#include <cooperative_groups.h>

#include "errors.h"
#include "pathfinding.cuh"
#include "hashtable.cuh"
#include "heap.cuh"

using namespace cooperative_groups;

__constant__ Coord endCuda;

Pathfinding::Pathfinding (const Config& config) : config(config) {
  FILE* input = fopen(config.input_data.c_str(), "r");

  fscanf(input, "%d,%d", &n, &m);
  fscanf(input, "%d,%d", &start.x, &start.y);
  fscanf(input, "%d,%d", &end.x, &end.y);

  gridHost = (int*) malloc(sizeof(int) * n * m);
  if (gridHost == nullptr) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(1);
  }

  for (int i = 0; i < n * m; i++) {
    gridHost[i] = 1;
  }

  int holes, non_ones;

  fscanf(input, "%d", &holes); 
  for (int i = 0; i < holes; i++) {
    int x, y;
    fscanf(input, "%d,%d", &x, &y);
    gridHost[getPosition(x, y)] = -1;
  }

  fscanf(input, "%d", &non_ones);
  for (int i = 0; i < non_ones; i++) {
    int x, y, val;
    fscanf(input, "%d,%d,%d", &x, &y, &val);
    gridHost[getPosition(x, y)] = val;
  }
}

Pathfinding::~Pathfinding() {
  if (gridHost != nullptr) {
    free(gridHost);
  }

  if (gridCuda != nullptr) {
    cudaFree(gridCuda);
  }

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

__device__ void Pathfinding::lock() {
  if (threadIdx.x == 0) {
    lockCuda.lock();
  }
  __syncthreads();
}

__device__ void Pathfinding::unlock() {
  if (threadIdx.x == 0) {
    lockCuda.unlock();
  }
  __syncthreads();
}

__device__ bool Pathfinding::inBounds(int x, int y) {
  return x >= 0 && x < n && y >= 0 && y < m;
}

__device__ void Pathfinding::expand(State& st, int stateIdx, int firstFreeSlot) {
  if (st.isNull()) {
    return;
  }

  // printf("Expanding state: ");
  st.print(n);

  int x = st.node % n;
  int y = st.node / n;
  
  int idx = firstFreeSlot;

  for (int i : {-1, 0, 1}) {
    for (int j : {-1, 0, 1}) {
      if (i == 0 && j == 0) continue;

      int nx = x + i;
      int ny = y + j;
      int newNode = getPosition(nx, ny);

      if (inBounds(nx, ny) && gridCuda[newNode] != -1) {
        // printf("Expanded node: %d, %d, index: %d\n", nx, ny, idx);
        statesCuda[idx].prev = stateIdx;
        statesCuda[idx].node = newNode;
        statesCuda[idx].g = st.g + gridCuda[newNode];
        statesCuda[idx].f = statesCuda[idx].g + abs(nx - endCuda.x) 
                            + abs(ny - endCuda.y);
      }

      idx++;
    }
  }

  // for (int i = 0; i <= 8; i++) {
    // printf("%d: ", i);
    // statesCuda[i].print(n);
  // }
}

__device__ void Pathfinding::extract() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int blockOffset = blockIdx.x * blockDim.x;

  __shared__ int offsets[THREADS_PER_BLOCK];

  lock();

  if (threadIdx.x == 0) {
    offsets[0] = *statesSizeCuda;
    for (int i = 1; i < THREADS_PER_BLOCK; i++) {
      offsets[i] = offsets[i-1] + 8 * min(8, queueSizesCuda[i - 1 + blockOffset]);
    }
    *statesSizeCuda = offsets[THREADS_PER_BLOCK - 1] +
                      8 * min(8, queueSizesCuda[THREADS_PER_BLOCK - 1 +
                          blockOffset]);
    // printf("blockIdx: %d: ", blockIdx.x);
    // for (int i = 0; i < THREADS_PER_BLOCK; i++) {
    //   printf("%d ", offsets[i]);
    // } 
    // printf("\n");
  }

  unlock();

  int firstFreeSlot = offsets[threadIdx.x];

  for (int i = 0; i < 8 && !empty(queueSizesCuda[idx]); i++) {
    QState qst = pop(queuesCuda + HEAP_SIZE * idx, queueSizesCuda[idx]);
    State st = statesCuda[qst.stateNumber];
    expand(st, qst.stateNumber, firstFreeSlot);
    firstFreeSlot += 8;
  }
}

__device__ void Pathfinding::findPath() {
  grid_group grid = this_grid();

  while(*finishedCuda == 0) {
    extract();
    grid.sync();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int finished = 1;
      for (int i = 0; i < BLOCKS * THREADS_PER_BLOCK; i++) {
        printf("i: %d, queueSize: %d\n", i, queueSizesCuda[i]);
        if (queueSizesCuda[i] > 0) {
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i <= 8; i++) {
      printf("%d: ", i);
      statesCuda[i].print(n);
    }
  }
}

__global__ void kernel(Pathfinding pathfinding) {
  pathfinding.findPath();
}


void Pathfinding::solve() {
  printf("n: %d, m: %d\n", n, m);
  printf("start: %d, %d\n", start.x, start.y);
  printf("end: %d, %d\n", end.x, end.y);
  printGrid();

  HANDLE_ERROR(cudaMalloc(&gridCuda, sizeof(State) * n * m));
  HANDLE_ERROR(cudaMalloc(&statesCuda, sizeof(State) * TABLE_SIZE));
  HANDLE_ERROR(cudaMalloc(&queuesCuda, sizeof(State) * BLOCKS * HEAP_SIZE));
  HANDLE_ERROR(cudaMalloc(&queueSizesCuda, sizeof(int) * BLOCKS *
        QUEUES_PER_BLOCK));
  HANDLE_ERROR(cudaMalloc(&hashtableCuda, sizeof(int) * TABLE_SIZE));
  HANDLE_ERROR(cudaMalloc(&statesSizeCuda, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&finishedCuda, sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(endCuda, &end, sizeof(Coord)));

  // TODO: handle errors
  statesHost = (State*) malloc(sizeof(State) * TABLE_SIZE);
  if (statesHost == nullptr) {
    printf("malloc failed!\n");
    exit(1);
  }

  for (int i = 0; i < TABLE_SIZE; i++) {
    statesHost[i] = State();
  }

  int startNode = getPosition(start.x, start.y);
  State initState = {
    .f = abs(start.x - end.x) + abs(start.y - end.y),
    .g = 0,
    .prev = startNode,
    .node = startNode,
  };
  QState initQState = {
    .f = 0,
    .stateNumber = 0,
  };

  statesHost[0] = initState;

  for (int i = 0; i <= 8; i++) {
    printf("%d: ", i);
    statesHost[i].print(n);
  }

  HANDLE_ERROR(cudaMemcpy(gridCuda, gridHost, sizeof(State) * n * m,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(statesCuda, statesHost, sizeof(State) * TABLE_SIZE,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(queuesCuda, &initQState, sizeof(QState),
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(queueSizesCuda, 0, sizeof(int) * BLOCKS *
        QUEUES_PER_BLOCK));
  HANDLE_ERROR(cudaMemset(statesSizeCuda, 1, 1));
  HANDLE_ERROR(cudaMemset(finishedCuda, 0, sizeof(int)));

  int one = 1;
  HANDLE_ERROR(cudaMemcpy(queueSizesCuda, &one, sizeof(int),
        cudaMemcpyHostToDevice));

  // TODO: add timing

  void* kernelArgs[] = {(void*) this};
  HANDLE_ERROR(cudaLaunchCooperativeKernel((void*) kernel, BLOCKS,
               THREADS_PER_BLOCK, kernelArgs));
  //kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(*this);

  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  printf("Computation finished!\n");
                              
  HANDLE_ERROR(cudaMemcpy(statesHost, statesCuda, sizeof(State) * TABLE_SIZE,
        cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaPeekAtLastError());

  for (int i = 0; i <= 8; i++) {
    printf("%d: ", i);
    statesHost[i].print(n);
  }
  // TODO: recreate the path
}


