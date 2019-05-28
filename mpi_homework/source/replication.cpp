#include <mpi.h>

#include "matrixmul.h"

void replicate(SparseMatrix& myA, SparseMatrixInfo& myAInfo,
               const Config& config, const MpiGroup& world,
               const MpiGroup& replGroup, const MpiGroup& layer) {
  SparseMatrix myOrigA = myA;

  for (int i = 0; i < replGroup.size; i++) {
    if (replGroup.rank == i) {
      {
        MPI_Request request;
        MPI_Ibcast(&myAInfo, SparseMatrixInfo::size, MPI_INT, i, replGroup.comm,
                   &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }

      {
        MPI_Request requests[3];
        MPI_Ibcast(myOrigA.row_se.data(), myOrigA.rows + 1, MPI_INT, i,
                   replGroup.comm, requests);
        if (myOrigA.nnz > 0) {
          MPI_Ibcast(myOrigA.col_indx.data(), myOrigA.nnz, MPI_INT, i,
                     replGroup.comm, requests + 1);
          MPI_Ibcast(myOrigA.values.data(), myOrigA.nnz, MPI_DOUBLE, i,
                     replGroup.comm, requests + 2);
        }
        MPI_Waitall(myOrigA.nnz > 0 ? 3 : 1, requests, MPI_STATUSES_IGNORE);
      }
    } else {
      SparseMatrixInfo otherInfo;
      SparseMatrix otherMatrix;

      {
        MPI_Request request;
        MPI_Ibcast(&otherInfo, SparseMatrixInfo::size, MPI_INT, i,
                   replGroup.comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        // otherInfo.print();
      }

      otherMatrix.reserveSpace(otherInfo);

      {
        MPI_Request requests[3];
        MPI_Ibcast(otherMatrix.row_se.data(), otherInfo.rows + 1, MPI_INT, i,
                   replGroup.comm, requests);
        if (otherInfo.nnz > 0) {
          MPI_Ibcast(otherMatrix.col_indx.data(), otherInfo.nnz, MPI_INT, i,
                     replGroup.comm, requests + 1);
          MPI_Ibcast(otherMatrix.values.data(), otherInfo.nnz, MPI_DOUBLE, i,
                     replGroup.comm, requests + 2);
        }
        MPI_Waitall(otherInfo.nnz > 0 ? 3 : 1, requests, MPI_STATUSES_IGNORE);
      }

      myA.merge(otherMatrix);
    }
  }

  myAInfo.update(myA);

  myA.print();

  if (!config.use_inner)
    return;

  int c = config.repl_group_size;
  int q = world.size / (c * c);
  int offset = layer.color * q;

  shiftAandCompute(myAInfo, myA, layer, offset, []() {});

  myA.print();
}
