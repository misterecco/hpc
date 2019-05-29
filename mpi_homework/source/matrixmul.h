#pragma once

#include <mpi.h>
#include <functional>

#include "config.h"
#include "matrix.h"
#include "mpigroup.h"

void initialize(MatrixInfo& myAInfo, SparseMatrix& myA, MatrixInfo& myCInfo,
                DenseMatrix& myC, const Config& config, const MpiGroup& world);

void replicate(SparseMatrix& myA, MatrixInfo& myAInfo, DenseMatrix& myC,
               MatrixInfo& myCInfo, const Config& config, const MpiGroup& world,
               const MpiGroup& replGroup, const MpiGroup& layer);

void shiftAandCompute(MatrixInfo& myAInfo, SparseMatrix& myA,
                      const MpiGroup& layer, int offset,
                      std::function<void()> computation);

void multiply(MatrixInfo& myAInfo, MatrixInfo& myCInfo, SparseMatrix& myA,
              DenseMatrix& myB, DenseMatrix& myC, const Config& config,
              const MpiGroup& world, const MpiGroup& layer);

void multiplyLocal(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C,
                   bool use_mkl);

void gatherC(const MatrixInfo& cInfo, DenseMatrix& myC, const MpiGroup& world,
             const MpiGroup& replGroup, const MpiGroup& layer,
             const Config& config, bool& isReducedToZeroLayer);

void countGe(const MatrixInfo& myCInfo, const DenseMatrix& myC,
             const MpiGroup& world, const MpiGroup& replGroup,
             const MpiGroup& layer, const Config& config,
             bool isReducedToZeroLayer);

template <typename T>
void broadcastMatrix(T& myMat, MatrixInfo& myMatInfo,
                     const MpiGroup& replGroup) {
  T myInitMat = myMat;

  for (int i = 0; i < replGroup.size; i++) {
    if (replGroup.rank == i) {
      MPI_Bcast(&myMatInfo, MatrixInfo::size, MPI_INT, i, replGroup.comm);

      myInitMat.broadcast(replGroup, i);
    } else {
      MatrixInfo otherInfo;
      MPI_Bcast(&otherInfo, MatrixInfo::size, MPI_INT, i, replGroup.comm);

      T otherMatrix(otherInfo);
      otherMatrix.broadcast(replGroup, i);

      myMat.merge(otherMatrix);
    }
  }

  myMatInfo = myMat.getInfo();
}
