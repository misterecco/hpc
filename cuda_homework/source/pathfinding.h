// #include <cuda.h>
#include <stdio.h>

#include "config.h"

using std::pair;

typedef pair<int, int> Coord;

class Pathfinding {
 public:
  Pathfinding (const Config& config) : config(config) {
    FILE* input = fopen(config.input_data.c_str(), "r");

    fscanf(input, "%d,%d", &n, &m);
    fscanf(input, "%d,%d", &start.first, &start.second);
    fscanf(input, "%d,%d", &end.first, &end.second);

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

  void solve();

  void printGrid() const {
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < n; x++) {
        printf("%d ", gridHost[getPosition(x, y)]);
      }
      printf("\n");
    }
  }

  ~Pathfinding() {
    if (gridHost != nullptr) {
      free(gridHost);
    }

    if (gridCuda != nullptr) {
      cudaFree(gridCuda);
    }
  }

 private:
  const Config config;
  int n; // x coords
  int m; // y coords
  Coord start;
  Coord end;
  int* gridHost = nullptr;
  int* gridCuda = nullptr;

  int getPosition(int x, int y) const {
    return y * n + x;
  }
};

