#pragma once

#include <mpi.h>

#include "config.h"
#include "matrix.h"

struct MpiGroup {
  int rank;
  int size;
  MPI_Comm comm;

  void initWorld() {
    comm = MPI_COMM_WORLD;
    init();
  }

  void initCustom(int color, int worldRank) {
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &comm);
    init();
  }

 private:
  void init() {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  }
};

void initialize(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
                SparseMatrixInfo& myCInfo, DenseMatrix& myC,
                const Config& config, const MpiGroup& world);

void replicate(SparseMatrix& myA, SparseMatrixInfo& myAInfo,
               const MpiGroup& replGroup);

void multiply(SparseMatrixInfo& myAInfo, SparseMatrixInfo& myCInfo,
              SparseMatrix& myA, DenseMatrix& myB, DenseMatrix& myC,
              const Config& config, const MpiGroup& world,
              const MpiGroup& layer);

void multiplyLocal(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C,
                   bool use_mkl);

void gatherC(const SparseMatrixInfo& myCInfo, DenseMatrix& myC,
             const MpiGroup& world);

void countGe(const SparseMatrixInfo& myCInfo, const DenseMatrix& myC, double g,
             const MpiGroup& world);
