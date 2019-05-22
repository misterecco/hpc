#pragma once

#include <mkl.h>

struct SparseMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int d = 0;
  int *rows_start = nullptr;
  int *rows_end = nullptr;
  int *col_indx = nullptr;
  double *values = nullptr;

  SparseMatrix() = default;
  SparseMatrix(std::string filePath);

  ~SparseMatrix();

  // not const because there is no const version
  // of mkl_sparse_d_create_csr
  sparse_matrix_t toMklSparse();
  void print() const;
};

struct DenseMatrix {
  int rows;
  int cols;
  double *values;
};
