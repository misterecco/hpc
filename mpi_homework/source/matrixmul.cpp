#include <mpi.h>
#include <iostream>

#include "config.h"
#include "matrixmul.h"
#include "utils.h"

using std::vector;

void print_usage(char** argv) {
  printf(
      "Usage: %s -f sparse_matrix_file -s seed_for_dense_matrix -c "
      "repl_group_size -e exponent [-g ge_value] [-v] [-i] [-m]\n",
      argv[0]);
}

int main(int argc, char** argv) {
  MpiGroup world, replGroup, layer;

  MPI_Init(&argc, &argv);
  Config config(argc, argv);

  world.initWorld();

  if (!config.check()) {
    if (world.rank == 0)
      print_usage(argv);
    exit(EXIT_FAILURE);
  }

  SparseMatrixInfo myAInfo;
  SparseMatrixInfo myCInfo;
  SparseMatrix myA;
  DenseMatrix myB;
  DenseMatrix myC;

  replGroup.initCustom(world.rank / config.repl_group_size, world.rank);
  layer.initCustom(world.rank % config.repl_group_size, world.rank);

  initialize(myAInfo, myA, myCInfo, myC, config, world);

  replicate(myA, myAInfo, replGroup);

  multiply(myAInfo, myCInfo, myA, myB, myC, config, world, layer);

  if (config.verbose) {
    gatherC(myCInfo, myC, world);
  }

  if (config.print_ge) {
    countGe(myCInfo, myC, config.ge_value, world);
  }

  MPI_Finalize();

  return 0;
}