#include <assert.h>

#include "pathfinding.cuh"
#include "hashtable.cuh"
#include "heap.cuh"

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

__global__ void kernel() {
  return;
}

void Pathfinding::solve() {
  printf("n: %d, m: %d\n", n, m);
  printf("start: %d, %d\n", start.x, start.y);
  printf("end: %d, %d\n", end.x, end.y);
  printGrid();

  // TODO: handle errors
  cudaMalloc(&gridCuda, sizeof(State) * n * m);
  cudaMalloc(&statesCuda, sizeof(State) * n * m);
  cudaMalloc(&queuesCuda, sizeof(State) * BLOCKS * HEAP_SIZE);
  cudaMalloc(&queueSizesCuda, sizeof(int) * BLOCKS * QUEUES_PER_BLOCK);
  cudaMalloc(&hashtableCuda, sizeof(int) * TABLE_SIZE);

  // TODO: handle errors
  statesHost = (State*) malloc(sizeof(State) * n * m);

  for (int i = 0; i < n * m; i++) {
    statesHost[i] = State();
  }

  int startNode = getPosition(start.x, start.y);
  State initState = {
    .f = 0,
    .g = 0,
    .prev = -1,
    .node = startNode,
  };
  QState initQState = {
    .f = 0,
    .stateNumber = 0,
  };

  statesHost[startNode] = initState;

  // TODO: handle errors
  cudaMemcpy(statesCuda, statesHost, sizeof(State) * n * m, cudaMemcpyHostToDevice);
  cudaMemcpy(queuesCuda, &initQState, sizeof(QState), cudaMemcpyHostToDevice);
  cudaMemset(queueSizesCuda, 0, sizeof(int) * BLOCKS * QUEUES_PER_BLOCK);

  int one = 1;
  cudaMemcpy(queueSizesCuda, &one, sizeof(int), cudaMemcpyHostToDevice);
}

