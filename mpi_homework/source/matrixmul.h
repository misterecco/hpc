#pragma once

#include <mpi.h>
#include <functional>

#include "config.h"
#include "matrix.h"

struct MpiGroup {
  int rank;
  int size;
  int color;
  MPI_Comm comm;

  MpiGroup() {
    comm = MPI_COMM_WORLD;
    color = 0;
    init();
  }

  MpiGroup(int color, int worldRank) {
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &comm);
    this->color = color;
    init();
  }

 private:
  void init() {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  }
};

void initialize(MatrixInfo& myAInfo, SparseMatrix& myA, MatrixInfo& myCInfo,
                DenseMatrix& myC, const Config& config, const MpiGroup& world);

void replicate(SparseMatrix& myA, MatrixInfo& myAInfo, const Config& config,
               const MpiGroup& world, const MpiGroup& replGroup,
               const MpiGroup& layer);

void shiftAandCompute(MatrixInfo& myAInfo, SparseMatrix& myA,
                      const MpiGroup& layer, int offset,
                      std::function<void()> computation);

void multiply(MatrixInfo& myAInfo, MatrixInfo& myCInfo, SparseMatrix& myA,
              DenseMatrix& myB, DenseMatrix& myC, const Config& config,
              const MpiGroup& world, const MpiGroup& layer);

void multiplyLocal(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C,
                   bool use_mkl);

void gatherC(const MatrixInfo& myCInfo, DenseMatrix& myC,
             const MpiGroup& world);

void countGe(const MatrixInfo& myCInfo, const DenseMatrix& myC, double g,
             const MpiGroup& world);
