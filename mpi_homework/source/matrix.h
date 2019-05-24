#pragma once

#include <mkl.h>
#include <vector>

struct SparseMatrixInfo {
  int rows;
  int cols;
  int nnz;
  int d;
  int actualRows;
  int rank;

  // IMPORTANT: keep in sync with actual fields count
  static constexpr int size = 6;

  void print() const {
    printf("rows: %d cols: %d actualRows: %d nnz: %d d: %d rank: %d\n",
      rows, cols, actualRows, nnz, d, rank);
  }
};

struct SparseMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int d = 0; // TODO: get rid of it?
  int actualRows = 0;
  int rank = 0;

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
  int rank;

  std::vector<double> values;

  DenseMatrix() = default;
  DenseMatrix(SparseMatrixInfo& matrixInfo, int rank, int numProcesses, int seed);

  void compact();
  void print() const;
};
