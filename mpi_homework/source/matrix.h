#pragma once

#include <mkl.h>
#include <vector>

struct SparseMatrixInfo {
  int rows;
  int cols;
  int nnz;
  int d;

  static constexpr int size = 4;

  void print() const {
    printf("rows: %d cols: %d nnz: %d d: %d\n", rows, cols, nnz, d);
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
  void reserveSpace(SparseMatrixInfo& matrixInfo);

  std::vector<SparseMatrixInfo> getColumnDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getColumnDistribution(int numProcesses) const;

  void print() const;
};

struct DenseMatrix {
  int rows;
  int cols;
  std::vector<double> values;
};
