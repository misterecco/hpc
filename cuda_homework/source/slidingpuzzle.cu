#include <algorithm>
#include <assert.h>
#include <vector>

#include "errors.h"
#include "slidingpuzzle.cuh"

static __constant__ SlidingPuzzle::PuzzleConfig endNodeCuda;
static __constant__ int numberToPositionCuda[25];

SlidingPuzzle::SlidingPuzzle (const Config& config) : config(config) {
  FILE* input = fopen(config.input_data.c_str(), "r");
  int next;

  for (int i = 0; i < 25; i++) {
    if(!fscanf(input, i == 24 ? "%d" : "%d,", &next)) {
        fscanf(input, "_,");
        next = 0;
    }
    startNode.board[i] = next;
    if (next == 0) startNode.zeroLoc = i;
  }

  for (int i = 0; i < 25; i++) {
    if(!fscanf(input, i == 24 ? "%d" : "%d,", &next)) {
        fscanf(input, "_,");
        next = 0;
    }
    endNode.board[i] = next;
    if (next == 0) endNode.zeroLoc = i;
  }

  printf("Start node: ");
  startNode.print();
  printf("\n");
  printf("End node: ");
  endNode.print();
  printf("\n");

  for (int i = 0; i < 25; i++) {
      int num = endNode.board[i];
      numberToPosition[num] = i;
  }

  for (int i = 0; i < 25; i++) {
    printf("%d ", numberToPosition[i]);
  }
  printf("\n");

  HANDLE_ERROR(cudaMemcpyToSymbol(numberToPositionCuda, numberToPosition, sizeof(int) * 25));
  HANDLE_ERROR(cudaMemcpyToSymbol(endNodeCuda, &endNode, sizeof(PuzzleConfig)));
  HANDLE_ERROR(cudaDeviceSynchronize());
}

  __device__ void SlidingPuzzle::expand(State* statesCuda, State& st, int stateIdx,
                         const Hashtable<State>& hashtable, int freeSlots[8],
                         int usedSlots[8], int& bestState) {
  if (st.isNull()) {
    return;
  }

  // st.print(0);

  int x = st.node.zeroLoc % 5;
  int y = st.node.zeroLoc / 5;

  int slotIdx = 0;

  for (int i : {-1, 0, 1}) {
    for (int j : {-1, 0, 1}) {
      if (i * j != 0 || (i == 0 && j == 0)) continue;

      int idx = freeSlots[slotIdx];

      if (x + i >= 0 && x + i < 5 && y + j >= 0 && y + j < 5) {
        PuzzleConfig newNode = st.node.swap(i, j);

        int h = 0;
        for (int i = 0; i < 25; i++) {
          int xp = i % 5;
          int yp = i / 5;
          int num = newNode.board[i];
          int tpos = numberToPositionCuda[num];
          int xt = tpos % 5;
          int yt = tpos % 5;
          h += num != 0 ? abs(xp - xt) + abs(yp - yt) : 0;
        }

        statesCuda[idx].prev = stateIdx;
        statesCuda[idx].node = newNode;
        statesCuda[idx].g = st.g + 1;
        statesCuda[idx].f = statesCuda[idx].g + h;

        if (hashtable.contains(statesCuda, idx)) {
          continue;
        }

        // printf("g: %d ", statesCuda[idx].g);
        // statesCuda[idx].print(0);

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

SlidingPuzzle::State SlidingPuzzle::getInitState() const {
  int h = 0;
  for (int i = 0; i < 25; i++) {
    int xp = i % 5;
    int yp = i / 5;
    int num = startNode.board[i];
    int tpos = numberToPosition[num];
    int xt = tpos % 5;
    int yt = tpos % 5;
    h += num != 0 ? abs(xp - xt) + abs(yp - yt) : 0;
  }

  return {
    .f = h,
    .g = 0,
    .prev = 0,
    .node = startNode,
  };
}

SlidingPuzzle::QState SlidingPuzzle::getInitQState() const {
  State initState = getInitState();
  return {
    .f = initState.f,
    .stateNumber = 0,
  };
}

void SlidingPuzzle::printSolution(State* statesHost, int bestState, float execTime) {
  FILE* output = fopen(config.output_data.c_str(), "w");
  State& st = statesHost[bestState];

  std::vector<PuzzleConfig> path;
  while(st.node != startNode) {
    path.push_back(st.node);
    st = statesHost[abs(st.prev)];
  }
  path.push_back(st.node);
  std::reverse(path.begin(), path.end());

  fprintf(output, "%.0f\n", execTime);

  for (PuzzleConfig& node : path) {
    node.print(output);
  }
}
