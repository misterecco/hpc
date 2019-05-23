#pragma once

#include <mkl.h>
#include <vector>

struct ProblemInfo {
  int rows;
  int cols;
  int nnz;

  void print() const {
    printf("rows: %d cols: %d nnz: %d\n", rows, cols, nnz);
  }
};

struct SparseMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int d = 0;

  std::vector<int> rows_start;
  std::vector<int> rows_end;
  std::vector<int> col_indx;
  std::vector<double> values;

  SparseMatrix() = default;
  SparseMatrix(std::string filePath);

  sparse_matrix_t toMklSparse();

  void addPadding(int numProcesses);
  void compact();

  std::vector<ProblemInfo> getColumnDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getColumnDistribution(int numProcesses) const;

  void print() const;
};

struct DenseMatrix {
  int rows;
  int cols;
  std::vector<double> values;
};
