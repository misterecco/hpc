#include <cstdio>
#include <limits>
#include <vector>

#include "config.h"
#include "lock.cuh"

#define BLOCKS 2
#define THREADS_PER_BLOCK 4
#define QUEUES_PER_BLOCK 4
#define TABLE_SIZE 256 * 1024 * 1024

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

  typedef unsigned int Seed;

  static const std::vector<Seed> seeds;

  unsigned int hash(Seed a) const {
    return (a * node) % TABLE_SIZE;
  }

  __device__ __host__ bool isNull() const {
    return node == -1 || f == -1;
  }

  __device__ __host__ bool equals(State& other) const {
    return other.node == node;
  }

  __device__ __host__ bool equals(int nd) const {
    return node == nd;
  }

  void clear() {
    node = -1;
  }

  __device__ __host__ void print(int n) {
    printf("node.x: %d, node.y: %d, prev.x: %d, prev.y: %d, f: %d, g: %d\n", 
        node == -1 ? -1 : node % n,
        node == -1 ? -1 : node / n, 
        prev == -1 ? -1 : prev % n,
        prev == -1 ? -1 : prev / n,
        f, 
        g);
  }
};

static const std::vector<State::Seed> seeds = {100000007u, 350002487u};

class Pathfinding {
 public:
  Pathfinding(const Config& config);
  void solve();
  __device__ void findPath();
  void printGrid() const {
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < n; x++) {
        printf("%d ", gridHost[getPosition(x, y)]);
      }
      printf("\n");
    }
  }

  ~Pathfinding();

 private:
  const Config config;
  int n; // x coords
  int m; // y coords
  Coord start;
  Coord end;
  int* gridHost = nullptr;
  int* gridCuda = nullptr;
  State* statesHost = nullptr;
  State* statesCuda = nullptr;
  int* statesSizeCuda = nullptr;
  QState* queuesCuda = nullptr;
  int* queueSizesCuda = nullptr;
  int* hashtableCuda = nullptr;
  int* finishedCuda = nullptr;
  int* bestStateCuda = nullptr;
  int* bestBlockStatesCuda = nullptr;
  int* endConditionCuda = nullptr;
  int bestState = -1;
  Lock lockCuda;

  __device__ __host__ int getPosition(int x, int y) const {
    return y * n + x;
  }

  __device__ void expand(State& st, int stateIdx, int firstFreeSlot, int& bestState);
  __device__ bool inBounds(int x, int y);
  __device__ void lock();
  __device__ void unlock();
  __device__ void extract(int& bestState);
};

