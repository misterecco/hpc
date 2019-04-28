#include <algorithm>
#include <assert.h>
#include <vector>

#include "errors.h"
#include "pathfinding.cuh"
#include "memory.h"
#include "hashtable.cuh"

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

__device__ void Pathfinding::expand(State* statesCuda, State& st, int stateIdx,
                                    const Hashtable<State>& hashtable, int freeSlots[8],
                                    int usedSlots[8], int& bestState) {
  if (st.isNull()) {
    return;
  }

  // st.print(n);

  int x = st.node % n;
  int y = st.node / n;

  int slotIdx = 0;

  for (int i : {-1, 0, 1}) {
    for (int j : {-1, 0, 1}) {
      if (i == 0 && j == 0) continue;

      int idx = freeSlots[slotIdx];

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
        if (hashtable.contains(statesCuda, idx)) {
          continue;
        }
        // statesCuda[idx].print(n);

        if (newNode == endNodeCuda && (bestState == -1 ||
              statesCuda[bestState].f > statesCuda[idx].f)) {
          printf("Updating my bestState to: %d\n", idx);
          bestState = idx;
        }

        usedSlots[slotIdx] = idx;
        freeSlots[slotIdx++] = -1;
      }
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
  int startNode = getPosition(start.x, start.y);
  return {
    .f = max(abs(start.x - end.x), abs(start.y - end.y)),
    .stateNumber = 0,
  };
}

void Pathfinding::printSolution(State* statesHost, int bestState) {
  State& st = statesHost[bestState];
  int initNode = getPosition(start.x, start.y);

  // TODO: write to file
  std::vector<int> path;
  while(st.node != initNode) {
    path.push_back(st.node);
    st = statesHost[abs(st.prev)];
  }
  path.push_back(st.node);
  std::reverse(path.begin(), path.end());

  for (int node : path) {
    printf("%d,%d\n", node % n, node / n);
  }
}
