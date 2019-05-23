#include <iostream>
#include <mkl_spblas.h>

#include "config.h"
#include "matrix.h"

void print_usage(char** argv) {
  printf("Usage: %s -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e exponent [-g ge_value] [-v] [-i] [-m]\n",
    argv[0]);
}

int main(int argc, char** argv) {
  Config config(argc, argv);

  if (!config.check()) {
    print_usage(argv);
    exit(EXIT_FAILURE);
  }

  config.print();

  int numProcesses = 3;

  SparseMatrix A(config.sparse_matrix_file);

  int n = A.rows;
  A.addPadding(numProcesses);

  auto info = A.getColumnDistributionInfo(numProcesses);

  auto dist = A.getColumnDistribution(numProcesses);

  for (auto d : dist) {
    d.print();
    std::cout << "============================" << std::endl;
  }

  A.print();

  // sparse_matrix_t mat;

  // mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, )

  return 0;
}