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
  MPI_Init(&argc, &argv);
  Config config(argc, argv);

  MpiGroup world;

  if (!config.check()) {
    if (world.rank == 0)
      print_usage(argv);
    exit(EXIT_FAILURE);
  }

  SparseMatrix myA;
  DenseMatrix myB;
  DenseMatrix myC;

  MpiGroup replGroup(world.rank / config.repl_group_size, world.rank);
  MpiGroup layer(world.rank % config.repl_group_size, world.rank);

  initialize(myA, myC, config, world);

  MatrixInfo cInfo = myA.getInfo();
  bool isCReducedToZeroLayer = false;

  replicate(myA, myC, config, world, replGroup, layer);

  multiply(myA, myB, myC, config, layer);

  if (config.verbose) {
    gatherC(cInfo, myC, world, replGroup, layer, config, isCReducedToZeroLayer);
  }

  if (config.print_ge) {
    countGe(myC, world, replGroup, layer, config, isCReducedToZeroLayer);
  }

  MPI_Finalize();

  return 0;
}

