#include <fstream>
#include <iostream>

#include "densematgen.h"
#include "matrix.h"
#include "memory.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::max;
using std::min;
using std::string;
using std::vector;

void MatrixInfo::print(FILE* file) const {
  fprintf(file, "rows: %d cols: %d actualRows: %d nnz: %d rank: %d firstCol: %d\n",
         rows, cols, actualRows, nnz, rank, firstCol);
}

bool MatrixInfo::check() const {
  return rows >= 0 && cols >= 0 && nnz >= 0 && actualRows >= 0 && rank >= 0 &&
         firstCol >= 0;
}

SparseMatrix::SparseMatrix(const string& filePath) {
  ifstream input(filePath);

  if (!input.is_open()) {
    fprintf(stderr, "Unable to read sparse matrix file\n");
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

SparseMatrix::SparseMatrix(const MatrixInfo& matrixInfo) {
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

void SparseMatrix::print(FILE* file) const {
  fprintf(file, "%d %d %d %d\n", rows, cols, nnz, rank);

  for (int i = 0; i < nnz; i++) {
    fprintf(file, "%f ", values[i]);
  }
  fprintf(file, "\n");

  for (int i = 0; i < rows + 1; i++) {
    fprintf(file, "%d ", row_se[i]);
  }
  fprintf(file, "\n");

  for (int i = 0; i < nnz; i++) {
    fprintf(file, "%d ", col_indx[i]);
  }
  fprintf(file, "\n");
}

MatrixInfo SparseMatrix::getInfo() const {
  return {
      .rows = rows,
      .cols = cols,
      .nnz = nnz,
      .actualRows = actualRows,
      .rank = rank,
      .firstCol = 0,
  };
}

void SparseMatrix::addPadding(int numProcesses) {
  int r = rows % numProcesses;
  int padding = r > 0 ? numProcesses - r : 0;

  rows += padding;
  cols += padding;

  row_se.resize(rows + 1, nnz);

  compact();
}

vector<MatrixInfo> SparseMatrix::getColumnDistributionInfo(
    int numProcesses) const {
  vector<MatrixInfo> dist;

  int colsPerProcess = cols / numProcesses;
  vector<int> nnzs(numProcesses);

  for (int idx : col_indx) {
    nnzs[idx / colsPerProcess] += 1;
  }

  for (int i = 0; i < numProcesses; i++) {
    MatrixInfo mi = {
        .rows = rows,
        .cols = cols,
        .nnz = nnzs[i],
        .actualRows = actualRows,
        .rank = i,
    };
    dist.push_back(mi);
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

vector<MatrixInfo> SparseMatrix::getRowDistributionInfo(
    int numProcesses) const {
  vector<MatrixInfo> dist;

  int rowsPerProcess = rows / numProcesses;

  for (int i = 0; i < numProcesses; i++) {
    int startRow = i * rowsPerProcess;
    int endRow = (i + 1) * rowsPerProcess;

    MatrixInfo mi = {
        .rows = rows,
        .cols = cols,
        .nnz = row_se[endRow] - row_se[startRow],
        .actualRows = actualRows,
        .rank = i,
    };
    dist.push_back(mi);
  }

  return dist;
}

vector<SparseMatrix> SparseMatrix::getRowDistribution(int numProcesses) const {
  vector<SparseMatrix> dist(numProcesses);

  int r = 0;
  for (auto& frag : dist) {
    frag.rows = rows;
    frag.cols = cols;
    frag.row_se.push_back(0);
    frag.actualRows = actualRows;
    frag.rank = r++;
  }

  int rowsPerProcess = rows / numProcesses;

  for (int p = 0; p < numProcesses; p++) {
    int startRow = p * rowsPerProcess;
    int endRow = (p + 1) * rowsPerProcess;

    auto& frag = dist[p];

    for (int row = startRow; row < endRow; row++) {
      for (int i = row_se[row]; i < row_se[row + 1]; i++) {
        frag.values.push_back(values[i]);
        frag.col_indx.push_back(col_indx[i]);
      }

      for (auto& f: dist) {
        f.row_se.push_back(f.values.size());
      }
    }
  }

  for (auto& frag: dist) {
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

void SparseMatrix::broadcast(const MpiGroup& replGroup, int sourceRank) {
  MPI_Request requests[3];
  MPI_Ibcast(row_se.data(), rows + 1, MPI_INT, sourceRank, replGroup.comm,
             requests);
  if (nnz > 0) {
    MPI_Ibcast(col_indx.data(), nnz, MPI_INT, sourceRank, replGroup.comm,
               requests + 1);
    MPI_Ibcast(values.data(), nnz, MPI_DOUBLE, sourceRank, replGroup.comm,
               requests + 2);
  }
  MPI_Waitall(nnz > 0 ? 3 : 1, requests, MPI_STATUSES_IGNORE);
}

DenseMatrix::DenseMatrix(const MatrixInfo& matrixInfo, int rank,
                         int numProcesses, int seed) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols / numProcesses;
  actualRows = matrixInfo.actualRows;
  firstCol = rank * cols;

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

DenseMatrix::DenseMatrix(const MatrixInfo& matrixInfo) {
  rows = matrixInfo.rows;
  cols = matrixInfo.cols;
  actualRows = matrixInfo.actualRows;
  firstCol = matrixInfo.firstCol;

  values.resize(rows * cols);
  compact();
}

MatrixInfo DenseMatrix::getInfo() const {
  return {
      .rows = rows,
      .cols = cols,
      .nnz = rows * cols,
      .actualRows = actualRows,
      .rank = 0,
      .firstCol = firstCol,
  };
}

void DenseMatrix::broadcast(const MpiGroup& replGroup, int sourceRank) {
  MPI_Bcast(values.data(), rows * cols, MPI_DOUBLE, sourceRank, replGroup.comm);
}

void DenseMatrix::compact() { values.shrink_to_fit(); }

void DenseMatrix::printColMajor() const {
  printf("-------------------------------------\n");
  printf("rows: %d cols: %d actualRows: %d firstCol: %d\n", rows, cols,
         actualRows, firstCol);
  for (int col = 0; col < cols; col++) {
    for (int row = 0; row < rows; row++) {
      cout << values[col * rows + row] << " ";
    }
    cout << endl;
  }
  printf("-------------------------------------\n");
}

void DenseMatrix::print(FILE* file) const {
  fprintf(file, "%d %d\n", actualRows, actualRows);
  for (int row = 0; row < actualRows; row++) {
    for (int col = 0; col < actualRows; col++) {
      fprintf(file, "%f ", values[col * rows + row]);
    }
    fprintf(file, "\n");
  }
}

void DenseMatrix::merge(const DenseMatrix& other) {
  int newFirstCol = min(firstCol, other.firstCol);
  int newLastCol = max(firstCol + cols, other.firstCol + other.cols);
  int newCols = newLastCol - newFirstCol;

  vector<double> newValues(rows * newCols);

  for (int col = firstCol; col < firstCol + cols; col++) {
    for (int row = 0; row < actualRows; row++) {
      newValues[(col - newFirstCol) * rows + row] =
          values[(col - firstCol) * rows + row];
    }
  }

  for (int col = other.firstCol; col < other.firstCol + other.cols; col++) {
    for (int row = 0; row < actualRows; row++) {
      newValues[(col - newFirstCol) * rows + row] +=
          other.values[(col - other.firstCol) * rows + row];
    }
  }

  cols = newCols;
  firstCol = newFirstCol;
  values = newValues;
}

int DenseMatrix::countGreaterOrEqual(double g) const {
  int count = 0;
  int lastCol = std::min(firstCol + cols, actualRows);

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
