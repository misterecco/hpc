#include <mpi.h>

#include "communication.h"
#include "utils.h"

using std::vector;

void initialDistibution(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
    const Config& config, const int myRank, const int numProcesses) {
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
}
