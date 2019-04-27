#include <assert.h>

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

__device__ void SlidingPuzzle::expand(State* statesCuda, State& st, int stateIdx, int firstFreeSlot,
                                    int& bestState) {
  if (st.isNull()) {
    return;
  }

  st.print(0);

  int idx = firstFreeSlot;
  int x = st.node.zeroLoc % 5;
  int y = st.node.zeroLoc / 5;

  for (int i : {-1, 0, 1}) {
    for (int j : {-1, 0, 1}) {
      if (i * j != 0 || (i == 0 && j == 0)) continue;

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
          h += abs(xp - xt) + abs(yp - yt);
        }

        statesCuda[idx].prev = stateIdx;
        statesCuda[idx].node = newNode;
        statesCuda[idx].g = st.g + 1;
        statesCuda[idx].f = statesCuda[idx].g + h;

        statesCuda[idx].print(0);

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

SlidingPuzzle::State SlidingPuzzle::getInitState() const {
  int h = 0;
  for (int i = 0; i < 25; i++) {
    int xp = i % 5;
    int yp = i / 5;
    int num = startNode.board[i];
    int tpos = numberToPosition[num];
    int xt = tpos % 5;
    int yt = tpos % 5;
    h += abs(xp - xt) + abs(yp - yt);
  }

  return {
    .f = h,
    .g = 0,
    .prev = 0,
    .node = startNode,
  };
}

// TODO: This doesn't change across different problems
SlidingPuzzle::QState SlidingPuzzle::getInitQState() const {
  return {
    .f = 0,
    .stateNumber = 0,
  };
}

void SlidingPuzzle::expandSolution(State* statesHost, int bestState) {
  State& st = statesHost[bestState];

  printf("path:\n");

  // TODO: invert the list
  // TODO: write to file
  while(st.node != startNode) {
    st.node.print();
    st = statesHost[abs(st.prev)];
  }
  st.node.print();
}
