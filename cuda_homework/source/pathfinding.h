#include <cstdio>
#include <limits>

#include "config.h"

#define BLOCKS 28 
#define THREADS_PER_BLOCK 32
#define QUEUES_PER_BLOCK 128
#define QUEUE_SIZE 8192


struct Coord {
  int x;
  int y;
};

struct State {
  int f = std::numeric_limits<int>::max();
  int g = std::numeric_limits<int>::max();
  int prev = -1; // y * n + x
  int node = -1; // y * n + x
};

class Pathfinding {
 public:
  Pathfinding(const Config& config);
  void solve();
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
  State* stateMapHost;
  State* stateMapCuda;
  State* queuesCuda;
  int* queueSizesCuda;

  size_t getPosition(int x, int y) const {
    return y * n + x;
  }
};

