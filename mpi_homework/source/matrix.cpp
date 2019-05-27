#include <fstream>
#include <iostream>

#include "densematgen.h"
#include "matrix.h"
#include "memory.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::min;
using std::string;
using std::vector;

void SparseMatrixInfo::print() const {
  printf("rows: %d cols: %d actualRows: %d nnz: %d rank: %d\n", rows, cols,
         actualRows, nnz, rank);
}

void SparseMatrixInfo::update(SparseMatrix& mat) {
  rows = mat.rows;
  cols = mat.cols;
  nnz = mat.nnz;
  actualRows = mat.actualRows;
  rank = mat.rank;
}

bool SparseMatrixInfo::check() const {
  return rows >= 0 && cols >= 0 && nnz >= 0 && actualRows >= 0 && rank >= 0;
}

SparseMatrix::SparseMatrix(const string& filePath) {
  ifstream input(filePath);

  if (!input.is_open()) {
    printf("Unable to read sparse matrix file\n");
    exit(EXIT_FAILURE);
  }

  int d;

  input >> rows >> cols >> nnz >> d;

  actualRows = rows;

  row_se.resize(rows + 1);
  col_indx.resize(nnz);
  values.resize(nnz);

  compact();

  for (int i = 0; i < nnz; i++) {
    input >> values[i];
  }

  for (int i = 0; i < rows + 1; i++) {
    input >> row_se[i];
  }

  for (int i = 0; i < nnz; i++) {
    input >> col_indx[i];
  }
}

sparse_matrix_t SparseMatrix::toMklSparse() {
  sparse_matrix_t mat;
  auto status = mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, rows,
                                        cols, row_se.data(), row_se.data() + 1,
                                        col_indx.data(), values.data());
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

  for (int i = 0; i < rows + 1; i++) {
    cout << row_se[i] << " ";
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

  row_se.resize(rows + 1, nnz);

  compact();
}

vector<SparseMatrixInfo> SparseMatrix::getColumnDistributionInfo(
    int numProcesses) const {
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

vector<SparseMatrix> SparseMatrix::getColumnDistribution(
    int numProcesses) const {
  vector<SparseMatrix> dist(numProcesses);

  for (auto& frag : dist) {
    frag.rows = rows;
    frag.cols = cols;
    frag.row_se.push_back(0);
  }

  int colsPerProcess = cols / numProcesses;

  for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
    int firstInd = row_se[rowIdx];
    int lastInd = row_se[rowIdx + 1];

    for (int i = firstInd; i < lastInd; i++) {
      int colIdx = col_indx[i];
      int p = colIdx / colsPerProcess;

      auto& frag = dist[p];
      frag.values.push_back(values[i]);
      frag.col_indx.push_back(colIdx);
    }

    for (auto& frag : dist) {
      frag.row_se.push_back(frag.values.size());
    }
  }

  for (auto& frag : dist) {
    frag.nnz = frag.values.size();
    frag.compact();
  }

  return dist;
}

vector<SparseMatrixInfo> SparseMatrix::getRowDistributionInfo(
    int numProcesses) const {
  vector<SparseMatrixInfo> dist;

  int rowsPerProcess = rows / numProcesses;

  for (int i = 0; i < numProcesses; i++) {
    int startRow = i * rowsPerProcess;
    int endRow = (i + 1) * rowsPerProcess;

    dist.push_back({
        .rows = rows,
        .cols = cols,
        .nnz = row_se[endRow] - row_se[startRow],
        .actualRows = actualRows,
        .rank = i,
    });
  }

  return dist;
}

vector<SparseMatrix> SparseMatrix::getRowDistribution(int numProcesses) const {
  vector<SparseMatrix> dist(numProcesses);

  int i = 0;
  for (auto& frag : dist) {
    frag.rows = rows;
    frag.cols = cols;
    frag.row_se.push_back(0);
    frag.actualRows = actualRows;
    frag.rank = i++;
  }

  int rowsPerProcess = rows / numProcesses;

  for (int p = 0; p < numProcesses; p++) {
    int startRow = p * rowsPerProcess;
    int endRow = (p + 1) * rowsPerProcess;

    auto& frag = dist[p];

    for (int row = startRow; row < endRow; row++) {
      for (int i = row_se[row]; i < row_se[row+1]; i++) {
        frag.values.push_back(values[i]);
        frag.col_indx.push_back(col_indx[i]);
      }

      for (auto& frag : dist) {
        frag.row_se.push_back(frag.values.size());
      }
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
    int i = row_se[row];
    int j = other.row_se[row];

    while (i < row_se[row + 1] || j < other.row_se[row + 1]) {
      if (i == row_se[row + 1] ||
          (j < other.row_se[row + 1] && other.col_indx[j] < col_indx[i])) {
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

  for (int i = 0; i < rows + 1; i++) {
    row_se[i] += other.row_se[i];
  }

  compact();
}

void SparseMatrix::compact() {
  row_se.shrink_to_fit();
  col_indx.shrink_to_fit();
  values.shrink_to_fit();
}

void SparseMatrix::reserveSpace(const SparseMatrixInfo& matrixInfo) {
  if (!matrixInfo.check()) {
    exit(EXIT_FAILURE);
  }

  rows = matrixInfo.rows;
  cols = matrixInfo.cols;
  nnz = matrixInfo.nnz;
  actualRows = matrixInfo.actualRows;
  rank = matrixInfo.rank;

  row_se.resize(rows + 1);
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

DenseMatrix::DenseMatrix(const SparseMatrixInfo& matrixInfo, int rank,
                         int numProcesses) {
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

void DenseMatrix::compact() { values.shrink_to_fit(); }

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
