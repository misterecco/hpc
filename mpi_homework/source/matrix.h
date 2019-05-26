#pragma once

#include <mkl.h>
#include <string>
#include <vector>

struct SparseMatrix;

struct SparseMatrixInfo {
  int rows;
  int cols;
  int nnz;
  int actualRows;
  int rank;

  // IMPORTANT: keep in sync with actual fields count
  static constexpr int size = 5;

  void print() const;

  void update(SparseMatrix& mat);
};

struct SparseMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int actualRows = 0;
  int rank = 0;

  std::vector<int> rows_start;
  std::vector<int> rows_end;
  std::vector<int> col_indx;
  std::vector<double> values;

  SparseMatrix() = default;
  SparseMatrix(std::string& filePath);

  sparse_matrix_t toMklSparse();

  void addPadding(int numProcesses);
  void compact();
  void reserveSpace(const SparseMatrixInfo& matrixInfo);

  std::vector<SparseMatrixInfo> getColumnDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getColumnDistribution(int numProcesses) const;
  void merge(const SparseMatrix& other);

  void print() const;
};

struct DenseMatrix {
  int rows;
  int cols;
  int rank;

  std::vector<double> values;

  DenseMatrix() = default;
  DenseMatrix(const SparseMatrixInfo& matrixInfo, int rank, int numProcesses, int seed);
  DenseMatrix(const SparseMatrixInfo& matrixInfo, int rank, int numProcesses);
  DenseMatrix(const SparseMatrixInfo& matrixInfo);

  void compact();
  void print() const;
  void print(int actualRows) const;
};
