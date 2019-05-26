#include <iostream>
#include <mkl.h>
#include <mpi.h>

#include "config.h"
#include "math.h"
#include "matrix.h"
#include "utils.h"

using std::vector;

void print_usage(char** argv) {
  printf("Usage: %s -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e exponent [-g ge_value] [-v] [-i] [-m]\n",
    argv[0]);
}

int main(int argc, char** argv) {
  int numProcesses, myRank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  Config config(argc, argv);

  if (!config.check()) {
    if (myRank == 0) print_usage(argv);
    exit(EXIT_FAILURE);
  }

  SparseMatrixInfo myAInfo;
  SparseMatrixInfo myCInfo;
  SparseMatrix myA;
  DenseMatrix myB;
  DenseMatrix myC;

  // INITIAL DISTRIBUTION
  // TODO: factor these blocks out
  if (myRank == 0) {
    config.print();

    SparseMatrix A(config.sparse_matrix_file);
    A.addPadding(numProcesses);

    // TODO: row distibution for InnerABC
    auto info = A.getColumnDistributionInfo(numProcesses);

    {
      MPI_Request request;
      MPI_Iscatter(info.data(), SparseMatrixInfo::size,
        MPI_INT, &myAInfo, SparseMatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD, &request);

      // myAInfo.print();
      myA.reserveSpace(myAInfo);
    }

    // TODO: row distibution for InnerABC
    auto frags = A.getColumnDistribution(numProcesses);

    {
      vector<int> allRowsStart;
      for (auto& frag : frags) {
        append(allRowsStart, frag.rows_start);
      }
      MPI_Request request;
      MPI_Iscatter(allRowsStart.data(), myAInfo.rows, MPI_INT, myA.rows_start.data(),
        myAInfo.rows, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      vector<int> allRowsEnd;
      for (auto& frag : frags) {
        append(allRowsEnd, frag.rows_end);
      }
      MPI_Request request;
      MPI_Iscatter(allRowsEnd.data(), myAInfo.rows, MPI_INT, myA.rows_end.data(),
        myAInfo.rows, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      vector<int> allNnz;
      vector<int> allDisp;

      {
        vector<int> allColIdx;
        for (auto& frag : frags) {
          append(allColIdx, frag.col_indx);
          allNnz.push_back(frag.nnz);
        }

        allDisp.push_back(0);
        for (int i = 0; i < numProcesses - 1; i++) {
          allDisp.push_back(allDisp.back() + allNnz[i]);
        }

        MPI_Request request;
        MPI_Iscatterv(allColIdx.data(), allNnz.data(), allDisp.data(), MPI_INT,
          myA.col_indx.data(), myAInfo.nnz, MPI_INT, 0, MPI_COMM_WORLD, &request);
      }

      {
        vector<double> allValues;
        for (auto& frag : frags) {
          append(allValues, frag.values);
        }

        MPI_Request request;
        MPI_Iscatterv(allValues.data(), allNnz.data(), allDisp.data(), MPI_DOUBLE,
          myA.values.data(), myAInfo.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
      }
    }
  } else {
    {
      MPI_Request request;
      MPI_Iscatter(nullptr, SparseMatrixInfo::size,
        MPI_INT, &myAInfo, SparseMatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      // myAInfo.print();
      myA.reserveSpace(myAInfo);
    }

    {
      MPI_Request request;
      MPI_Iscatter(nullptr, myAInfo.rows, MPI_INT, myA.rows_start.data(),
        myAInfo.rows, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      MPI_Request request;
      MPI_Iscatter(nullptr, myAInfo.rows, MPI_INT, myA.rows_end.data(),
        myAInfo.rows, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      MPI_Request request;
      MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_INT,
        myA.col_indx.data(), myAInfo.nnz, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      MPI_Request request;
      MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
        myA.values.data(), myAInfo.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    }
  }

  myC = DenseMatrix(myAInfo, myRank, numProcesses, config.seed);
  // myB.print();
  MPI_Barrier(MPI_COMM_WORLD);
  // myA.print();

  myCInfo = myAInfo;

  // REPLICATION
  MPI_Comm myReplGroup;
  // int groupCount = numProcesses / config.repl_group_size;
  MPI_Comm_split(MPI_COMM_WORLD, myRank / config.repl_group_size,
      myRank % config.repl_group_size, &myReplGroup);
  int myGroupRank;
  MPI_Comm_rank(myReplGroup, &myGroupRank);

  {
    SparseMatrix myOrigA = myA;

    for (int i = 0; i < config.repl_group_size; i++) {
      if (myGroupRank == i) {
        {
          MPI_Request request;
          MPI_Ibcast(&myAInfo, SparseMatrixInfo::size, MPI_INT, i, myReplGroup,
            &request);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        {
          MPI_Request requests[4];
          MPI_Ibcast(myOrigA.rows_start.data(), myOrigA.rows, MPI_INT, i, myReplGroup,
            requests);
          MPI_Ibcast(myOrigA.rows_end.data(), myOrigA.rows, MPI_INT, i, myReplGroup,
            requests + 1);
          if (myOrigA.nnz > 0) {
            MPI_Ibcast(myOrigA.col_indx.data(), myOrigA.nnz, MPI_INT, i, myReplGroup,
              requests + 2);
            MPI_Ibcast(myOrigA.values.data(), myOrigA.nnz, MPI_DOUBLE, i, myReplGroup,
              requests + 3);
          }
          MPI_Waitall(myOrigA.nnz > 0 ? 4 : 2, requests, MPI_STATUSES_IGNORE);
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
          MPI_Request requests[4];
          MPI_Ibcast(otherMatrix.rows_start.data(), otherInfo.rows, MPI_INT, i, myReplGroup,
            requests);
          MPI_Ibcast(otherMatrix.rows_end.data(), otherInfo.rows, MPI_INT, i, myReplGroup,
            requests + 1);
          if (otherInfo.nnz > 0) {
            MPI_Ibcast(otherMatrix.col_indx.data(), otherInfo.nnz, MPI_INT, i, myReplGroup,
              requests + 2);
            MPI_Ibcast(otherMatrix.values.data(), otherInfo.nnz, MPI_DOUBLE, i, myReplGroup,
              requests + 3);
          }
          MPI_Waitall(otherInfo.nnz > 0 ? 4 : 2, requests, MPI_STATUSES_IGNORE);
        }

        myA.merge(otherMatrix);
      }
    }
  }

  // myA.print();
  myAInfo.update(myA);

  MPI_Comm myLayer;
  int myLayerRank;
  int layerSize = numProcesses / config.repl_group_size;
  MPI_Comm_split(MPI_COMM_WORLD, myRank % config.repl_group_size,
      myRank, &myLayer);
  MPI_Comm_rank(myLayer, &myLayerRank);

  // MULTIPLICATION
  for (int e = 0; e < config.exponent; e++) {
    myB = myC;
    myC = DenseMatrix(myCInfo, myRank, numProcesses);

    for (int i = 0; i < layerSize; i++) {
      SparseMatrixInfo nextAInfo;
      SparseMatrix nextA;
      int sendToGroupRank = myLayerRank > 0 ? (myLayerRank - 1) : layerSize - 1;
      int recvFromGroupRank = (myLayerRank + 1) % layerSize;

      if (layerSize > 1) {
        if (myLayerRank == 0) {
          MPI_Recv(&nextAInfo, SparseMatrixInfo::size, MPI_INT,
            recvFromGroupRank, 0, myLayer, MPI_STATUS_IGNORE);
          MPI_Send(&myAInfo, SparseMatrixInfo::size, MPI_INT,
            sendToGroupRank, 0, myLayer);
        } else {
          MPI_Send(&myAInfo, SparseMatrixInfo::size, MPI_INT,
            sendToGroupRank, 0, myLayer);
          MPI_Recv(&nextAInfo, SparseMatrixInfo::size, MPI_INT,
            recvFromGroupRank, 0, myLayer, MPI_STATUS_IGNORE);
        }
      }

      nextA.reserveSpace(nextAInfo);

      MPI_Request sendRequests[4];
      MPI_Request recvRequests[4];

      if (layerSize > 1) {
        MPI_Isend(myA.rows_start.data(), myA.rows, MPI_INT,
          sendToGroupRank, 1, myLayer, sendRequests);
        MPI_Isend(myA.rows_end.data(), myA.rows, MPI_INT,
          sendToGroupRank, 2, myLayer, sendRequests + 1);
        if (myA.nnz > 0) {
          MPI_Isend(myA.col_indx.data(), myA.nnz, MPI_INT,
            sendToGroupRank, 3, myLayer, sendRequests + 2);
          MPI_Isend(myA.values.data(), myA.nnz, MPI_DOUBLE,
            sendToGroupRank, 4, myLayer, sendRequests + 3);
        }

        MPI_Irecv(nextA.rows_start.data(), nextAInfo.rows, MPI_INT,
          recvFromGroupRank, 1, myLayer, recvRequests);
        MPI_Irecv(nextA.rows_end.data(), nextAInfo.rows, MPI_INT,
          recvFromGroupRank, 2, myLayer, recvRequests + 1);
        if (nextAInfo.nnz > 0) {
          MPI_Irecv(nextA.col_indx.data(), nextAInfo.nnz, MPI_INT,
            recvFromGroupRank, 3, myLayer, recvRequests + 2);
          MPI_Irecv(nextA.values.data(), nextAInfo.nnz, MPI_DOUBLE,
            recvFromGroupRank, 4, myLayer, recvRequests + 3);
        }
      }

      multiply(myA, myB, myC, config.use_mkl);

      if (layerSize > 1) {
        MPI_Waitall(myA.nnz > 0 ? 4 : 2, sendRequests, MPI_STATUSES_IGNORE);
        MPI_Waitall(nextAInfo.nnz > 0 ? 4 : 2, recvRequests, MPI_STATUSES_IGNORE);

        myA = nextA;
        myAInfo = nextAInfo;
      }
    }
  }


  // RESULT GATHERING
  // TODO: different scheme for InnerABC
  if (config.verbose) {
    if (myRank == 0) {
      DenseMatrix C(myAInfo);

      MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
        C.values.data(), myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      C.print(myAInfo.actualRows);
    } else {
      MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
        nullptr, myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();

  return 0;
}