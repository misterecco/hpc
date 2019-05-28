#pragma once

#include <functional>

#include "mpigroup.h"

template <typename T>
inline void append(std::vector<T>& lhs, std::vector<T>& rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}

inline void debugPrint(const MpiGroup& world, std::function<void()> debug) {
  for (int i = 0; i < world.size; i++) {
    if (world.rank == i) {
      printf("myRank: %d\n", world.rank);
      debug();
    }
    MPI_Barrier(world.comm);
  }
}
