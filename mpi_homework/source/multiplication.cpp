#include <mkl.h>

#include "matrixmul.h"

void multiplyLocal(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C,
                   bool use_mkl) {
  if (A.nnz == 0)
    return;

  if (use_mkl) {
    sparse_matrix_t myAMkl = A.toMklSparse();

    struct matrix_descr mType {
      .type = SPARSE_MATRIX_TYPE_GENERAL,
      // not relevant, but compiler complains without them
          .mode = SPARSE_FILL_MODE_FULL, .diag = SPARSE_DIAG_NON_UNIT,
    };

    // C := alpha*op(A)*B + beta*C
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, myAMkl, mType,
                    SPARSE_LAYOUT_COLUMN_MAJOR, B.values.data(), C.cols, B.rows,
                    1.0, C.values.data(), C.rows);
  } else {
    for (int i = 0; i < B.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        int firstK = A.row_se[i];
        int lastK = A.row_se[i + 1];

        for (int k = firstK; k < lastK; k++) {
          int col = A.col_indx[k];
          C.values[C.rows * j + i] += A.values[k] * B.values[B.rows * j + col];
        }
      }
    }
  }
}

void multiply(SparseMatrixInfo& myAInfo, SparseMatrixInfo& myCInfo,
              SparseMatrix& myA, DenseMatrix& myB, DenseMatrix& myC,
              const Config& config, const MpiGroup& world,
              const MpiGroup& layer) {
  for (int e = 0; e < config.exponent; e++) {
    myB = myC;
    myC = DenseMatrix(myCInfo, world.rank, world.size);

    for (int i = 0; i < layer.size; i++) {
      SparseMatrixInfo nextAInfo;
      SparseMatrix nextA;
      int sendToGroupRank = layer.rank > 0 ? (layer.rank - 1) : layer.size - 1;
      int recvFromGroupRank = (layer.rank + 1) % layer.size;

      if (layer.size > 1) {
        if (layer.rank == 0) {
          MPI_Recv(&nextAInfo, SparseMatrixInfo::size, MPI_INT,
                   recvFromGroupRank, 0, layer.comm, MPI_STATUS_IGNORE);
          MPI_Send(&myAInfo, SparseMatrixInfo::size, MPI_INT, sendToGroupRank,
                   0, layer.comm);
        } else {
          MPI_Send(&myAInfo, SparseMatrixInfo::size, MPI_INT, sendToGroupRank,
                   0, layer.comm);
          MPI_Recv(&nextAInfo, SparseMatrixInfo::size, MPI_INT,
                   recvFromGroupRank, 0, layer.comm, MPI_STATUS_IGNORE);
        }
        nextA.reserveSpace(nextAInfo);
      }

      MPI_Request sendRequests[3];
      MPI_Request recvRequests[3];

      if (layer.size > 1) {
        MPI_Isend(myA.row_se.data(), myA.rows + 1, MPI_INT, sendToGroupRank, 1,
                  layer.comm, sendRequests);
        if (myA.nnz > 0) {
          MPI_Isend(myA.col_indx.data(), myA.nnz, MPI_INT, sendToGroupRank, 2,
                    layer.comm, sendRequests + 1);
          MPI_Isend(myA.values.data(), myA.nnz, MPI_DOUBLE, sendToGroupRank, 3,
                    layer.comm, sendRequests + 2);
        }

        MPI_Irecv(nextA.row_se.data(), nextAInfo.rows + 1, MPI_INT,
                  recvFromGroupRank, 1, layer.comm, recvRequests);
        if (nextAInfo.nnz > 0) {
          MPI_Irecv(nextA.col_indx.data(), nextAInfo.nnz, MPI_INT,
                    recvFromGroupRank, 2, layer.comm, recvRequests + 1);
          MPI_Irecv(nextA.values.data(), nextAInfo.nnz, MPI_DOUBLE,
                    recvFromGroupRank, 3, layer.comm, recvRequests + 2);
        }
      }

      multiplyLocal(myA, myB, myC, config.use_mkl);

      if (layer.size > 1) {
        MPI_Waitall(myA.nnz > 0 ? 3 : 1, sendRequests, MPI_STATUSES_IGNORE);
        MPI_Waitall(nextAInfo.nnz > 0 ? 3 : 1, recvRequests,
                    MPI_STATUSES_IGNORE);

        myA = nextA;
        myAInfo = nextAInfo;
      }
    }
  }
}
