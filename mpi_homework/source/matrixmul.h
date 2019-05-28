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
               const Config& config, const MpiGroup& world,
               const MpiGroup& replGroup, const MpiGroup& layer);

void shiftAandCompute(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
                      const MpiGroup& layer, int offset,
                      std::function<void()> computation);

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
