#pragma once

#include <cstdio>
#include <limits>
#include <vector>

#include "config.h"
#include "lock.cuh"
#include "hashtable.cuh"

namespace slidingpuzzle {

// TODO: perhaps keep some hash or transform to integers
struct PuzzleConfig {
  char board[25] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
  char zeroLoc = 0;

  __device__ __host__ bool operator==(const PuzzleConfig& other) const {
    if (other.zeroLoc != zeroLoc) return false;

    for (int i = 0; i < 25; i++) {
      if (other.board[i] != board[i]) return false;
    }

    return true;
  };

  __device__ __host__ bool operator!=(const PuzzleConfig& other) const {
    return !(*this == other);
  };

  void print(FILE* file) const {
    for (int i = 0; i < 25; i++) {
      fprintf(file, i == 24 ? "%d\n" : "%d,", board[i]);
    }
  }

  __device__ __host__ void print() const {
    for (int i = 0; i < 25; i++) {
      printf(i == 24 ? "%d" : "%d,", board[i]);
    }
  }

  __device__ PuzzleConfig swap(int o_x, int o_y) {
    PuzzleConfig newConfig;
    memcpy(newConfig.board, board, sizeof(char) * 25);

    int newZeroLoc = zeroLoc + o_x + 5 * o_y;

    char tmp = newConfig.board[newZeroLoc];
    newConfig.board[newZeroLoc] = newConfig.board[zeroLoc];
    newConfig.board[zeroLoc] = tmp;

    newConfig.zeroLoc = newZeroLoc;

    return newConfig;
  }
};

struct QState {
  int f;
  int stateNumber;
};

struct State {
  int f = -1;
  int g = -1;
  int prev = -1;
  PuzzleConfig node;

  __device__ unsigned int hash(unsigned int a, int tableSize) const {
    unsigned int result = 0;
    for (int i = 0; i < 25; i++) {
      result = (result * a + node.board[i]) % tableSize;
    }
    return result;
  }

  __device__ __host__ bool isNull() const {
    return prev < 0;
  }

  __device__ __host__ bool equals(State& other) const {
    return other.node == node;
  }

  __device__ void clear() {
    prev = -1 * abs(prev);
  }

  __device__ __host__ void print(int) {
    printf("node: [ ");
    node.print();
    printf("], prev: %d, f: %d, g: %d\n", prev, f, g);
  }

  __device__ __host__ void printNode() {
    node.print();
  }
};

} // slidingpuzzle


class SlidingPuzzle {
 public:
  SlidingPuzzle(const Config& config);

  typedef slidingpuzzle::PuzzleConfig PuzzleConfig;
  typedef slidingpuzzle::QState QState;
  typedef slidingpuzzle::State State;

  State getInitState() const;
  QState getInitQState() const;
  void printSolution(State* statesHost, int bestState, float execTime);
  static const int statesUnrolledPerStep = 4;

  __device__ void expand(State* statesCuda, State& st, int stateIdx,
                         const Hashtable<State>& hashtable, int freeSlots[8],
                         int usedSlots[8], int& bestState);

  const Config config;
  PuzzleConfig startNode;
  PuzzleConfig endNode;
  int numberToPosition[25];
  int n = 0;
};
