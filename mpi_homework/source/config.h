#pragma once

#include <cstring>
#include <string>

struct Config {
  bool use_inner = false;
  bool use_mkl = false;
  bool verbose = false;
  bool print_ge = false;
  int repl_group_size = -1;
  int exponent = -1;
  int seed;
  double ge_value;
  std::string sparse_matrix_file = "";

  Config(int argc, char** argv) {
    int i = 1;

    while (i < argc) {
      if (!strcmp("-f", argv[i])) {
        sparse_matrix_file = argv[++i];
      } else if (!strcmp("-s", argv[i])) {
        seed = atoi(argv[++i]);
      } else if (!strcmp("-c", argv[i])) {
        repl_group_size = atoi(argv[++i]);
      } else if (!strcmp("-e", argv[i])) {
        exponent = atoi(argv[++i]);
      } else if (!strcmp("-g", argv[i])) {
        print_ge = true;
        ge_value = atof(argv[++i]);
      } else if (!strcmp("-v", argv[i])) {
        verbose = true;
      } else if (!strcmp("-i", argv[i])) {
        use_inner = true;
      } else if (!strcmp("-m", argv[i])) {
        use_mkl = true;
      }
      ++i;
    }
  }

  bool check() const {
    return sparse_matrix_file.size() > 0 && exponent > 0 && repl_group_size > 0;
  }

  void print(FILE* out) const {
    fprintf(out, "Config: \n");
    fprintf(out, "-f %s -s %i -c %d -e %d ", sparse_matrix_file.c_str(), seed,
            repl_group_size, exponent);
    if (print_ge) {
      fprintf(out, "-g %f ", ge_value);
    }
    if (verbose) {
      fprintf(out, "-v ");
    }
    if (use_inner) {
      fprintf(out, "-i ");
    }
    if (use_mkl) {
      fprintf(out, "-m ");
    }
    fprintf(out, "\n");
  }
};
