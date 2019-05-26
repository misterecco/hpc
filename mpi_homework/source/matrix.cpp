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

void SparseMatrixInfo::print() const {
  printf("rows: %d cols: %d actualRows: %d nnz: %d rank: %d\n",
    rows, cols, actualRows, nnz, rank);
}

void SparseMatrixInfo::update(SparseMatrix& mat) {
  rows = mat.rows;
  cols = mat.cols;
  nnz = mat.nnz;
  actualRows = mat.actualRows;
  rank = mat.rank;
}

SparseMatrix::SparseMatrix(string& filePath) {
  ifstream input(filePath);

  if (!input.is_open()) {
    printf("Unable to read sparse matrix file\n");
    exit(EXIT_FAILURE);
  }

  int d;

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
  cout << rows << " " << cols << " " << nnz << " " << rank << endl;

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

  rows += padding;
  cols += padding;

  rows_start.resize(rows, nnz);
  rows_end.resize(rows, nnz);

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

    for (auto& frag : dist) {
      frag.rows_start.push_back(frag.values.size());
    }

    for (int i = firstInd; i < lastInd; i++) {
      int colIdx = col_indx[i];
      int p = colIdx / colsPerProcess;

      auto& frag = dist[p];
      frag.values.push_back(values[i]);
      frag.col_indx.push_back(colIdx);
    }

    for (auto& frag : dist) {
      frag.rows_end.push_back(frag.values.size());
    }
  }

  for (auto& frag : dist) {
    frag.nnz = frag.values.size();
    frag.compact();
  }

  return dist;
}

void SparseMatrix::merge(const SparseMatrix& other) {
  nnz += other.nnz;

  vector<int> newColIndx;
  vector<double> newValues;

  for (int row = 0; row < rows; row++) {
    int i = rows_start[row];
    int j = other.rows_start[row];

    while (i < rows_end[row] || j < other.rows_end[row]) {
      if (i == rows_end[row]
          || (j < other.rows_end[row] && other.col_indx[j] < col_indx[i])) {
        newValues.push_back(other.values[j]);
        newColIndx.push_back(other.col_indx[j]);
        j += 1;
      } else {
        newValues.push_back(values[i]);
        newColIndx.push_back(col_indx[i]);
        i += 1;
      }
    }
  }

  col_indx = newColIndx;
  values = newValues;

  for (int i = 0; i < rows; i++) {
    rows_start[i] += other.rows_start[i];
    rows_end[i] += other.rows_end[i];
  }

  compact();
}

void SparseMatrix::compact() {
  rows_start.shrink_to_fit();
  rows_end.shrink_to_fit();
  col_indx.shrink_to_fit();
  values.shrink_to_fit();
}

void SparseMatrix::reserveSpace(const SparseMatrixInfo& matrixInfo) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols;
  nnz = matrixInfo.nnz;
  actualRows = matrixInfo.actualRows;
  rank = matrixInfo.rank;

  rows_start.resize(rows);
  rows_end.resize(rows);
  col_indx.resize(nnz);
  values.resize(nnz);

  compact();
}


DenseMatrix::DenseMatrix(const SparseMatrixInfo& matrixInfo, int rank,
    int numProcesses, int seed) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols / numProcesses;
  this->rank = rank;

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

DenseMatrix::DenseMatrix(const SparseMatrixInfo& matrixInfo, int rank, int numProcesses) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols / numProcesses;
  this->rank = rank;

  values.resize(rows * cols);
  compact();
}

DenseMatrix::DenseMatrix(const SparseMatrixInfo& matrixInfo) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols;
  rank = 0;

  values.resize(rows * cols);
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

void DenseMatrix::print(int actualRows) const {
  cout << rows << " " << cols << endl;
  for (int row = 0; row < actualRows; row++) {
    for (int col = 0; col < actualRows; col++) {
      cout << values[col * rows + row] << " ";
    }
    cout << endl;
  }
}

int DenseMatrix::countGreaterOrEqual(double g, int actualRows) const {
  int count = 0;
  int firstCol = cols * rank;
  int lastCol = std::min(cols * (rank + 1), actualRows);

  for (int actualCol = firstCol; actualCol < lastCol; actualCol++) {
    int col = actualCol - firstCol;

    for (int row = 0; row < actualRows; row++) {
      if (values[col * rows + row] >= g) {
        count += 1;
      }
    }
  }

  return count;
}
