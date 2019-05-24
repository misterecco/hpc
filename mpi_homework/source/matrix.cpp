#include <fstream>
#include <iostream>

#include "matrix.h"
#include "memory.h"
#include "densematgen.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::min;
using std::string;
using std::vector;

SparseMatrix::SparseMatrix(string filePath) {
  ifstream input(filePath);

  if (!input.is_open()) {
    printf("Unable to read sparse matrix file\n");
    exit(EXIT_FAILURE);
  }

  input >> rows >> cols >> nnz >> d;

  actualRows = rows;

  rows_start.resize(rows);
  rows_end.resize(rows);
  col_indx.resize(nnz);
  values.resize(nnz);

  compact();

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

sparse_matrix_t SparseMatrix::toMklSparse() {
  sparse_matrix_t mat;
  auto status = mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, rows, cols,
              rows_start.data(), rows_end.data(), col_indx.data(), values.data());
  if (status != SPARSE_STATUS_SUCCESS) {
    printf("Conversion to sparse mlk failed\n");
    exit(EXIT_FAILURE);
  }
  return mat;
}

void SparseMatrix::print() const {
  cout << rows << " " << cols << " " << nnz << " " << d << endl;

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

void SparseMatrix::addPadding(int numProcesses) {
  int r = rows % numProcesses;
  int padding = r > 0 ? numProcesses - r : 0;

  int lastRowsStartElement = rows_start.back();
  int lastRowsEndElement = rows_end.back();

  rows += padding;
  cols += padding;

  rows_start.resize(rows, lastRowsStartElement);
  rows_end.resize(rows, lastRowsEndElement);

  compact();
}

vector<SparseMatrixInfo> SparseMatrix::getColumnDistributionInfo(int numProcesses) const {
  vector<SparseMatrixInfo> dist;

  int colsPerProcess = cols / numProcesses;
  vector<int> nnzs(numProcesses);

  for (int idx : col_indx) {
    nnzs[idx / colsPerProcess] += 1;
  }

  for (int i = 0; i < numProcesses; i++) {
    dist.push_back({
      .rows = rows,
      .cols = cols,
      .nnz = nnzs[i],
      .d = min(nnzs[i], colsPerProcess),
      .actualRows = actualRows,
      .rank = i,
    });
  }

  return dist;
}

vector<SparseMatrix> SparseMatrix::getColumnDistribution(int numProcesses) const {
  vector<SparseMatrix> dist(numProcesses);

  for (auto& frag: dist) {
    frag.rows = rows;
    frag.cols = cols;
  }

  int colsPerProcess = cols / numProcesses;

  for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
    int firstInd = rows_start[rowIdx];
    int lastInd = rows_end[rowIdx];

    for (int p = 0; p < numProcesses; p++) {
      auto& frag = dist[p];
      frag.rows_start.push_back(frag.values.size());
    }

    for (int i = firstInd; i < lastInd; i++) {
      int colIdx = col_indx[i];
      int p = colIdx / colsPerProcess;
      // TODO: should the colIdx be adjusted here? Probably not

      auto& frag = dist[p];
      frag.values.push_back(values[i]);
      frag.col_indx.push_back(colIdx);
    }

    for (int p = 0; p < numProcesses; p++) {
      auto& frag = dist[p];
      frag.rows_end.push_back(frag.values.size());
    }
  }

  for (auto& frag : dist) {
    frag.nnz = frag.values.size();
    frag.d = min(frag.nnz, colsPerProcess);
    frag.compact();
  }

  return dist;
}

void SparseMatrix::compact() {
  rows_start.shrink_to_fit();
  rows_end.shrink_to_fit();
  col_indx.shrink_to_fit();
  values.shrink_to_fit();
}

void SparseMatrix::reserveSpace(SparseMatrixInfo& matrixInfo) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols;
  nnz = matrixInfo.nnz;
  d = matrixInfo.d;
  actualRows = matrixInfo.actualRows;

  rows_start.resize(rows);
  rows_end.resize(rows);
  col_indx.resize(nnz);
  values.resize(nnz);

  compact();
}


DenseMatrix::DenseMatrix(SparseMatrixInfo& matrixInfo, int rank,
    int numProcesses, int seed) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols / numProcesses;
  rank = rank;

  values.resize(rows * cols);

  for (int col = 0; col < cols; col++) {
    int globalCol = cols * rank + col;
    if (globalCol >= matrixInfo.actualRows) {
      break;
    }

    for (int row = 0; row < matrixInfo.actualRows; row++) {
      values[col * rows + row] = generate_double(seed, row, globalCol);
    }
  }

  compact();
}

void DenseMatrix::compact() {
  values.shrink_to_fit();
}

void DenseMatrix::print() const {
  for (int col = 0; col < cols; col++) {
    for (int row = 0; row < rows; row++) {
      cout << values[col * rows + row] << " ";
    }
    cout << endl;
  }

}