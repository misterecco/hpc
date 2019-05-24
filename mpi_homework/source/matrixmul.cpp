#include <iostream>
#include <mkl.h>
#include <mpi.h>

#include "config.h"
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

    myB = DenseMatrix(myAInfo, myRank, numProcesses, config.seed);
    myC = DenseMatrix(myAInfo, myRank, numProcesses);
    // myB.print();

    MPI_Barrier(MPI_COMM_WORLD);

    // myA.print();

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

    myB = DenseMatrix(myAInfo, myRank, numProcesses, config.seed);
    myC = DenseMatrix(myAInfo, myRank, numProcesses);
    // myB.print();

    MPI_Barrier(MPI_COMM_WORLD);

    // myA.print();
  }

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

  myA.print();

  // sparse_matrix_t mat;

  // mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, )

  // RESULT GATHERING
  // TODO: send back C instead of B when there is something to send
  // TODO: different scheme for InnerABC
  if (config.verbose) {
    if (myRank == 0) {
      DenseMatrix C(myAInfo);

      MPI_Gather(myB.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
        C.values.data(), myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      C.print(myAInfo.actualRows);
    } else {
      MPI_Gather(myB.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
        nullptr, myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();

  return 0;
}