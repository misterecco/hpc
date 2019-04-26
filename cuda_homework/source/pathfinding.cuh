#include <cstdio>
#include <limits>
#include <vector>

#include "config.h"
#include "lock.cuh"

#define BLOCKS 2
#define THREADS_PER_BLOCK 8
#define QUEUES_PER_BLOCK 8
#define TABLE_SIZE 64 * 1024 * 1024
#define HASH_TABLE_SIZE 1024 * 1024

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
  int prev = -1; // y * n + x
  int node = -1; // y * n + x

  __device__ unsigned int hash(unsigned int a) const {
    unsigned int result = (a * node) % HASH_TABLE_SIZE;
    // result = ((result + f) * a) % TABLE_SIZE;
    // result = ((result + g) * a) % TABLE_SIZE;
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

  State getInitState();
  QState getInitQState();
  void expandSolution(State* statesHost, int bestState);

  __device__ void expand(State* statesCuda, State& st, int stateIdx, int firstFreeSlot, int& bestState);

  const Config config;
  int n; // x coords
  int m; // y coords
  Coord start;
  Coord end;
  int* gridHost = nullptr;
  int* gridCuda = nullptr;

  __device__ __host__ int getPosition(int x, int y) const;
  __device__ void expand(State& st, int stateIdx, int firstFreeSlot, int& bestState);
  __device__ bool inBounds(int x, int y);
  void printGrid() const;
};
