#pragma once

#include <mkl.h>
#include <string>
#include <vector>

#include "mpigroup.h"

struct MatrixInfo {
  int rows = -1;
  int cols = -1;
  int nnz = -1;
  int actualRows = -1;
  int rank = 0;
  int firstCol = 0;

  // IMPORTANT: keep in sync with actual fields count
  static constexpr int size = 6;

  void print(FILE* file = stdout) const;

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
  SparseMatrix(const MatrixInfo& matrixInfo);

  MatrixInfo getInfo() const;

  sparse_matrix_t toMklSparse();

  void addPadding(int numProcesses);
  void compact();

  std::vector<MatrixInfo> getColumnDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getColumnDistribution(int numProcesses) const;
  std::vector<MatrixInfo> getRowDistributionInfo(int numProcesses) const;
  std::vector<SparseMatrix> getRowDistribution(int numProcesses) const;
  void merge(const SparseMatrix& other);

  void broadcast(const MpiGroup& replGroup, int sourceRank);

  void print(FILE* file = stdout) const;
};

struct DenseMatrix {
  int rows;
  int cols;
  int actualRows;
  int firstCol;

  std::vector<double> values;

  DenseMatrix() = default;
  DenseMatrix(const MatrixInfo& matrixInfo, int rank, int numProcesses,
              int seed);
  DenseMatrix(const MatrixInfo& matrixInfo);

  MatrixInfo getInfo() const;

  void compact();
  void printColMajor() const;
  void print(FILE* file = stdout) const;

  void broadcast(const MpiGroup& replGroup, int sourceRank);

  void merge(const DenseMatrix& other);
  int countGreaterOrEqual(double g) const;
};
