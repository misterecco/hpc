#include <assert.h>

#include "errors.h"
#include "pathfinding.cuh"
#include "memory.h"

static __constant__ Pathfinding::Coord endCuda;
static __constant__ int endNodeCuda;

Pathfinding::Pathfinding (const Config& config) : config(config) {
  FILE* input = fopen(config.input_data.c_str(), "r");

  fscanf(input, "%d,%d", &n, &m);
  fscanf(input, "%d,%d", &start.x, &start.y);
  fscanf(input, "%d,%d", &end.x, &end.y);

  gridHost = (int*) HANDLE_NULLPTR(malloc(sizeof(State) * n * m));

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

  printf("n: %d, m: %d\n", n, m);
  printf("start: %d, %d\n", start.x, start.y);
  printf("end: %d, %d\n", end.x, end.y);
  // printGrid();

  int endNode = getPosition(end.x, end.y);

  HANDLE_ERROR(cudaMalloc(&gridCuda, sizeof(State) * n * m));
  HANDLE_ERROR(cudaMemcpyToSymbol(endCuda, &end, sizeof(Coord)));
  HANDLE_ERROR(cudaMemcpyToSymbol(endNodeCuda, &endNode, sizeof(int)));

  HANDLE_ERROR(cudaMemcpy(gridCuda, gridHost, sizeof(State) * n * m,
        cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
}

Pathfinding::~Pathfinding() {
  maybeFree(gridHost);
  maybeCudaFree(gridCuda);
}

__device__ bool Pathfinding::inBounds(int x, int y) {
  return x >= 0 && x < n && y >= 0 && y < m;
}

__device__ void Pathfinding::expand(State* statesCuda, State& st, int stateIdx, int firstFreeSlot,
                                    int& bestState) {
  if (st.isNull()) {
    return;
  }

  // st.print(n);

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
        statesCuda[idx].f = statesCuda[idx].g +
                            max(abs(nx - endCuda.x), abs(ny - endCuda.y));
        // statesCuda[idx].print(n);

        if (newNode == endNodeCuda && (bestState == -1 ||
              statesCuda[bestState].f > statesCuda[idx].f)) {
          printf("Updating my bestState to: %d\n", idx);
          bestState = idx;
        }
      }

      idx++;
    }
  }
}

__device__ __host__ int Pathfinding::getPosition(int x, int y) const {
  return y * n + x;
}

void Pathfinding::printGrid() const {
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++) {
      printf("%d ", gridHost[getPosition(x, y)]);
    }
    printf("\n");
  }
}

Pathfinding::State Pathfinding::getInitState() {
  int startNode = getPosition(start.x, start.y);
  return {
    .f = max(abs(start.x - end.x), abs(start.y - end.y)),
    .g = 0,
    .prev = 0,
    .node = startNode,
  };
}

Pathfinding::QState Pathfinding::getInitQState() {
  return {
    .f = 0,
    .stateNumber = 0,
  };
}

void Pathfinding::expandSolution(State* statesHost, int bestState) {
  State& st = statesHost[bestState];
  int initNode = getPosition(start.x, start.y);

  printf("path:\n");

  // TODO: invert the list
  // TODO: write to file
  while(st.node != initNode) {
    printf("%d,%d\n", st.node % n, st.node / n);
    st = statesHost[abs(st.prev)];
  }
  printf("%d,%d\n", st.node % n, st.node / n);
}
