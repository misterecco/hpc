#pragma once

#include <mpi.h>
#include <functional>

#include "config.h"
#include "matrix.h"
#include "mpigroup.h"

void initialize(SparseMatrix& myA, DenseMatrix& myC, const Config& config,
                const MpiGroup& world);

void replicate(SparseMatrix& myA, DenseMatrix& myC, const Config& config,
               const MpiGroup& world, const MpiGroup& replGroup,
               const MpiGroup& layer);

void shiftAandCompute(SparseMatrix& myA, const MpiGroup& layer, int offset,
                      std::function<void()> computation);

void multiply(SparseMatrix& myA, DenseMatrix& myB, DenseMatrix& myC,
              const Config& config, const MpiGroup& layer);

void gatherC(const MatrixInfo& cInfo, DenseMatrix& myC, const MpiGroup& world,
             const MpiGroup& replGroup, const MpiGroup& layer,
             const Config& config, bool& isCReducedToZeroLayer);

void countGe(DenseMatrix& myC, const MpiGroup& world, const MpiGroup& replGroup,
             const MpiGroup& layer, const Config& config,
             bool& isCReducedToZeroLayer);

template <typename T>
void broadcastMatrix(T& myMat, const MpiGroup& replGroup) {
  if (replGroup.size == 0)
    return;

  T myInitMat = myMat;
  MatrixInfo myMatInfo = myInitMat.getInfo();

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
}
