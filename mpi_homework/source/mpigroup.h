#pragma once

#include <mpi.h>

struct MpiGroup {
  int rank;
  int size;
  int color;
  MPI_Comm comm;

  MpiGroup() {
    comm = MPI_COMM_WORLD;
    color = 0;
    init();
  }

  MpiGroup(int color, int worldRank) {
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &comm);
    this->color = color;
    init();
  }

 private:
  void init() {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  }
};