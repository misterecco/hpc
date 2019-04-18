#include "pathfinding.h"

void Pathfinding::solve() {
  printf("n: %d, m: %d\n", n, m);
  printf("start: %d, %d\n", start.first, start.second);
  printf("end: %d, %d\n", end.first, end.second);
  printGrid();
}

