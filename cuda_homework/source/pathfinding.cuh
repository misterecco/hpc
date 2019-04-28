#pragma once

#include <cstdio>
#include <limits>
#include <vector>

#include "config.h"
#include "lock.cuh"
#include "hashtable.cuh"

namespace pathfinding {

struct Coord {
  int x;
  int y;
};

struct QState {
  int f;
  int stateNumber;
};

struct State {
  int f = -1;
  int g = -1;
  int prev = -1;
  int node = -1; // y * n + x

  __device__ unsigned int hash(unsigned long long a, int tableSize) const {
    unsigned int result = (a * node) % tableSize;
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

  __device__ __host__ void print(int n) {
    printf("node.x: %d, node.y: %d, prev: %d, f: %d, g: %d\n",
        node == -1 ? -1 : node % n, node == -1 ? -1 : node / n, prev, f, g);
  }

  __device__ __host__ void printNode(int n) {
    printf("%d,%d\n", node == -1 ? -1 : node % n,
      node == -1 ? -1 : node / n);
  }
};

} // pathfinding


class Pathfinding {
 public:
  Pathfinding(const Config& config);
  ~Pathfinding();

  typedef pathfinding::Coord Coord;
  typedef pathfinding::QState QState;
  typedef pathfinding::State State;

  static constexpr int kStatesUnrolledPerStep = 8;
  static constexpr int kBlocks = 8;
  static constexpr int kThreadsPerBlock = 128;
  static constexpr int kQueueSize = 8 * 8192;
  static constexpr int kTableSize = 96 * 1024 * 1024;
  static constexpr int kHashTableSize = 32 * 1024 * 1024;
  static constexpr int kUnrollingRounds = 1;
  static constexpr int kStatesUnrolledPerRound =
                       kStatesUnrolledPerStep * kUnrollingRounds;

  State getInitState();
  QState getInitQState();
  void printSolution(State* statesHost, int bestState, float execTime);

__device__ void expand(State* statesCuda, State& st, int stateIdx,
                       const Hashtable<State>& hashtable, int freeSlots[8],
                       int usedSlots[8], int& bestState);
  __device__ __host__ int getPosition(int x, int y) const;
  __device__ bool inBounds(int x, int y);
  void printGrid() const;

  const Config config;
  int n; // x coords
  int m; // y coords
  Coord start;
  Coord end;
  int* gridHost = nullptr;
  int* gridCuda = nullptr;
};
