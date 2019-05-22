#include <fstream>
#include <iostream>

#include "matrix.h"
#include "memory.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::string;

SparseMatrix::SparseMatrix(string filePath) {
  ifstream input(filePath);

  if (!input.is_open()) {
    printf("Unable to read sparse matrix file\n");
    exit(EXIT_FAILURE);
  }

  input >> rows >> cols >> nnz >> d;

  rows_start = new int[rows];
  rows_end = new int[rows];
  col_indx = new int[nnz];
  values = new double[nnz];

  for (int i = 0; i < nnz; i++) {
    input >> values[i];
  }

  input >> rows_start[0];

  for (int i = 0; i < rows; i++) {
    input >> rows_end[i];
  }

  for (int i = 1; i < rows; i++) {
    rows_start[i] = rows_end[i-1];
  }

  for (int i = 0; i < nnz; i++) {
    input >> col_indx[i];
  }
}

SparseMatrix::~SparseMatrix() {
  delete[] rows_start;
  delete[] rows_end;
  delete[] col_indx;
  delete[] values;
}

sparse_matrix_t SparseMatrix::toMklSparse() {
  sparse_matrix_t mat;
  auto status = mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, rows, cols,
                          rows_start, rows_end, col_indx, values);
  if (status != SPARSE_STATUS_SUCCESS) {
    printf("Conversion to sparse mlk failed\n");
    exit(EXIT_FAILURE);
  }
  return mat;
}

void SparseMatrix::print() const {
  for (int i = 0; i < nnz; i++) {
    cout << values[i] << " ";
  }
  cout << endl;

  cout << 0 << " ";
  for (int i = 0; i < rows; i++) {
    cout << rows_end[i] << " ";
  }
  cout << endl;

  for (int i = 0; i < nnz; i++) {
    cout << col_indx[i] << " ";
  }
  cout << endl;
}