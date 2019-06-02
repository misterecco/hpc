#include <functional>

#include "matrixmul.h"

using std::function;

void shiftAandCompute(SparseMatrix& myA, const MpiGroup& layer, int offset,
                      function<void()> computation) {
  MatrixInfo nextAInfo;
  MatrixInfo myAInfo = myA.getInfo();
  SparseMatrix nextA;
  int sendToLayerRank = (layer.rank + offset) % layer.size;
  int recvFromLayerRank = (layer.rank >= offset)
                              ? layer.rank - offset
                              : (layer.rank + layer.size) - offset;

  if (layer.size > 1 && offset > 0) {
    if (layer.rank == 0) {
      MPI_Recv(&nextAInfo, MatrixInfo::size, MPI_INT, recvFromLayerRank, 0,
               layer.comm, MPI_STATUS_IGNORE);
      MPI_Send(&myAInfo, MatrixInfo::size, MPI_INT, sendToLayerRank, 0,
               layer.comm);
    } else {
      MPI_Send(&myAInfo, MatrixInfo::size, MPI_INT, sendToLayerRank, 0,
               layer.comm);
      MPI_Recv(&nextAInfo, MatrixInfo::size, MPI_INT, recvFromLayerRank, 0,
               layer.comm, MPI_STATUS_IGNORE);
    }
    nextA = SparseMatrix(nextAInfo);
  }

  MPI_Request sendRequests[3];
  MPI_Request recvRequests[3];

  if (layer.size > 1 && offset > 0) {
    MPI_Isend(myA.row_se.data(), myA.rows + 1, MPI_INT, sendToLayerRank, 1,
              layer.comm, sendRequests);
    if (myA.nnz > 0) {
      MPI_Isend(myA.col_indx.data(), myA.nnz, MPI_INT, sendToLayerRank, 2,
                layer.comm, sendRequests + 1);
      MPI_Isend(myA.values.data(), myA.nnz, MPI_DOUBLE, sendToLayerRank, 3,
                layer.comm, sendRequests + 2);
    }

    MPI_Irecv(nextA.row_se.data(), nextAInfo.rows + 1, MPI_INT,
              recvFromLayerRank, 1, layer.comm, recvRequests);
    if (nextAInfo.nnz > 0) {
      MPI_Irecv(nextA.col_indx.data(), nextAInfo.nnz, MPI_INT,
                recvFromLayerRank, 2, layer.comm, recvRequests + 1);
      MPI_Irecv(nextA.values.data(), nextAInfo.nnz, MPI_DOUBLE,
                recvFromLayerRank, 3, layer.comm, recvRequests + 2);
    }
  }

  computation();

  if (layer.size > 1 && offset > 0) {
    MPI_Waitall(myA.nnz > 0 ? 3 : 1, sendRequests, MPI_STATUSES_IGNORE);
    MPI_Waitall(nextAInfo.nnz > 0 ? 3 : 1, recvRequests, MPI_STATUSES_IGNORE);

    myA = nextA;
  }
}
