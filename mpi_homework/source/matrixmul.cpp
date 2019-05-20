#include <cstring>
#include <iostream>

#include "config.h"

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

  return 0;
}