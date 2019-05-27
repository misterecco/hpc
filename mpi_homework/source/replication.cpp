#include <mpi.h>

#include "communication.h"

void replicate(SparseMatrix& myA, SparseMatrixInfo& myAInfo,
    MPI_Comm myReplGroup, int myGroupRank, int c) {
  SparseMatrix myOrigA = myA;

  for (int i = 0; i < c; i++) {
    if (myGroupRank == i) {
      {
        MPI_Request request;
        MPI_Ibcast(&myAInfo, SparseMatrixInfo::size, MPI_INT, i, myReplGroup,
          &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }

      {
        MPI_Request requests[3];
        MPI_Ibcast(myOrigA.row_se.data(), myOrigA.rows + 1, MPI_INT, i, myReplGroup,
          requests);
        if (myOrigA.nnz > 0) {
          MPI_Ibcast(myOrigA.col_indx.data(), myOrigA.nnz, MPI_INT, i, myReplGroup,
            requests + 1);
          MPI_Ibcast(myOrigA.values.data(), myOrigA.nnz, MPI_DOUBLE, i, myReplGroup,
            requests + 2);
        }
        MPI_Waitall(myOrigA.nnz > 0 ? 3 : 1, requests, MPI_STATUSES_IGNORE);
      }
    } else {
      SparseMatrixInfo otherInfo;
      SparseMatrix otherMatrix;

      {
        MPI_Request request;
        MPI_Ibcast(&otherInfo, SparseMatrixInfo::size, MPI_INT, i, myReplGroup,
          &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        // otherInfo.print();
      }

      otherMatrix.reserveSpace(otherInfo);

      {
        MPI_Request requests[3];
        MPI_Ibcast(otherMatrix.row_se.data(), otherInfo.rows + 1, MPI_INT, i, myReplGroup,
          requests);
        if (otherInfo.nnz > 0) {
          MPI_Ibcast(otherMatrix.col_indx.data(), otherInfo.nnz, MPI_INT, i, myReplGroup,
            requests + 1);
          MPI_Ibcast(otherMatrix.values.data(), otherInfo.nnz, MPI_DOUBLE, i, myReplGroup,
            requests + 2);
        }
        MPI_Waitall(otherInfo.nnz > 0 ? 3 : 1, requests, MPI_STATUSES_IGNORE);
      }

      myA.merge(otherMatrix);
    }
  }

  myAInfo.update(myA);
}
