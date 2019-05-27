#include <mpi.h>
#include <iostream>

#include "communication.h"
#include "config.h"
#include "math.h"
#include "matrix.h"
#include "utils.h"

using std::vector;

void print_usage(char** argv) {
  printf(
      "Usage: %s -f sparse_matrix_file -s seed_for_dense_matrix -c "
      "repl_group_size -e exponent [-g ge_value] [-v] [-i] [-m]\n",
      argv[0]);
}

int main(int argc, char** argv) {
  MpiGroup world, replGroup, layer;

  MPI_Init(&argc, &argv);
  Config config(argc, argv);

  world.initWorld();

  if (!config.check()) {
    if (world.rank == 0)
      print_usage(argv);
    exit(EXIT_FAILURE);
  }

  SparseMatrixInfo myAInfo;
  SparseMatrixInfo myCInfo;
  SparseMatrix myA;
  DenseMatrix myB;
  DenseMatrix myC;

  replGroup.initCustom(world.rank / config.repl_group_size, world.rank);
  layer.initCustom(world.rank % config.repl_group_size, world.rank);

  initialize(myAInfo, myA, myCInfo, myC, config, world);

  replicate(myA, myAInfo, replGroup);

  // myA.print();

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
      }

      nextA.reserveSpace(nextAInfo);

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

      multiply(myA, myB, myC, config.use_mkl);

      if (layer.size > 1) {
        MPI_Waitall(myA.nnz > 0 ? 3 : 1, sendRequests, MPI_STATUSES_IGNORE);
        MPI_Waitall(nextAInfo.nnz > 0 ? 3 : 1, recvRequests,
                    MPI_STATUSES_IGNORE);

        myA = nextA;
        myAInfo = nextAInfo;
      }
    }
  }

  // RESULT GATHERING
  // TODO: different scheme for InnerABC
  if (config.verbose) {
    if (world.rank == 0) {
      DenseMatrix C(myAInfo);

      MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
                 C.values.data(), myC.rows * myC.cols, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

      C.print(myAInfo.actualRows);
    } else {
      MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE, nullptr,
                 myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  // TODO: different scheme for InnerABC
  if (config.print_ge) {
    int myCount = myC.countGreaterOrEqual(config.ge_value, myAInfo.actualRows);
    int totalCount = 0;

    MPI_Reduce(&myCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
      std::cout << totalCount << std::endl;
    }
  }

  MPI_Finalize();

  return 0;
}