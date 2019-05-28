#pragma once

#include <mkl.h>
#include <string>
#include <vector>

struct SparseMatrix;

struct MatrixInfo {
  int rows = -1;
  int cols = -1;
  int nnz = -1;
  int actualRows = -1;
  int rank = -1;

  // IMPORTANT: keep in sync with actual fields count
  static constexpr int size = 5;

  void print() const;

  void update(SparseMatrix& mat);

  bool check() const;
};

struct SparseMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int actualRows = 0;
  int rank = 0;

  std::vector<int> row_se;
  std::vector<int> col_indx;
  std::vector<double> values;

  SparseMatrix() = default;
  SparseMatrix(const std::string& filePath);

  sparse_matrix_t toMklSparse();

  void addPadding(int numProcesses);
  void compact();
  void reserveSpace(const MatrixInfo& matrixInfo);

  std::vector<MatrixInfo> getColumnDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getColumnDistribution(int numProcesses) const;
  std::vector<MatrixInfo> getRowDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getRowDistribution(int numProcesses) const;
  void merge(const SparseMatrix& other);

  void print() const;
};

struct DenseMatrix {
  int rows;
  int cols;
  int rank;

  std::vector<double> values;

  DenseMatrix() = default;
  DenseMatrix(const MatrixInfo& matrixInfo, int rank, int numProcesses,
              int seed);
  DenseMatrix(const MatrixInfo& matrixInfo, int rank, int numProcesses);
  DenseMatrix(const MatrixInfo& matrixInfo);

  void compact();
  void print() const;
  void print(int actualRows) const;

  void merge(const DenseMatrix& other);
  int countGreaterOrEqual(double g, int actualRows) const;
};
