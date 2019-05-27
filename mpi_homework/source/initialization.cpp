#include <mpi.h>

#include "matrixmul.h"
#include "utils.h"

using std::vector;

void initialize(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
                SparseMatrixInfo& myCInfo, DenseMatrix& myC,
                const Config& config, const MpiGroup& world) {
  if (world.rank == 0) {
    config.print();

    SparseMatrix A(config.sparse_matrix_file);
    A.addPadding(world.size);

    // TODO: row distibution for InnerABC
    auto info = A.getColumnDistributionInfo(world.size);

    {
      MPI_Request request;
      MPI_Iscatter(info.data(), SparseMatrixInfo::size, MPI_INT, &myAInfo,
                   SparseMatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD,
                   &request);

      myA.reserveSpace(myAInfo);
    }

    // TODO: row distibution for InnerABC
    auto frags = A.getColumnDistribution(world.size);

    {
      vector<int> allRowSe;
      for (auto& frag : frags) {
        append(allRowSe, frag.row_se);
      }
      MPI_Request request;
      MPI_Iscatter(allRowSe.data(), myAInfo.rows + 1, MPI_INT,
                   myA.row_se.data(), myAInfo.rows + 1, MPI_INT, 0,
                   MPI_COMM_WORLD, &request);
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
        for (int i = 0; i < world.size - 1; i++) {
          allDisp.push_back(allDisp.back() + allNnz[i]);
        }

        MPI_Request request;
        MPI_Iscatterv(allColIdx.data(), allNnz.data(), allDisp.data(), MPI_INT,
                      myA.col_indx.data(), myAInfo.nnz, MPI_INT, 0,
                      MPI_COMM_WORLD, &request);
      }

      {
        vector<double> allValues;
        for (auto& frag : frags) {
          append(allValues, frag.values);
        }

        MPI_Request request;
        MPI_Iscatterv(allValues.data(), allNnz.data(), allDisp.data(),
                      MPI_DOUBLE, myA.values.data(), myAInfo.nnz, MPI_DOUBLE, 0,
                      MPI_COMM_WORLD, &request);
      }
    }
  } else {
    {
      MPI_Request request;
      MPI_Iscatter(nullptr, SparseMatrixInfo::size, MPI_INT, &myAInfo,
                   SparseMatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD,
                   &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      myA.reserveSpace(myAInfo);
    }

    {
      MPI_Request request;
      MPI_Iscatter(nullptr, myAInfo.rows + 1, MPI_INT, myA.row_se.data(),
                   myAInfo.rows + 1, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      MPI_Request request;
      MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_INT, myA.col_indx.data(),
                    myAInfo.nnz, MPI_INT, 0, MPI_COMM_WORLD, &request);
    }

    {
      MPI_Request request;
      MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, myA.values.data(),
                    myAInfo.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    }
  }

  myC = DenseMatrix(myAInfo, world.rank, world.size, config.seed);

  MPI_Barrier(MPI_COMM_WORLD);

  myCInfo = myAInfo;
}
