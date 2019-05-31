#include <mpi.h>

#include "matrixmul.h"
#include "utils.h"

using std::vector;

// TODO: fix case when there is only one process
void initialize(SparseMatrix& myA, DenseMatrix& myC, const Config& config,
                const MpiGroup& world) {
  MatrixInfo myAInfo;

  MPI_Request requests[3];

  if (world.rank == 0) {
    config.print(stderr);

    SparseMatrix A(config.sparse_matrix_file);

    A.addPadding(world.size);

    vector<MatrixInfo> info;
    if (config.use_inner) {
      info = A.getRowDistributionInfo(world.size);
    } else {
      info = A.getColumnDistributionInfo(world.size);
    }

    {
      MPI_Request request;
      MPI_Iscatter(info.data(), MatrixInfo::size, MPI_INT, &myAInfo,
                   MatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      myA = SparseMatrix(myAInfo);
    }

    vector<SparseMatrix> frags;
    if (config.use_inner) {
      frags = A.getRowDistribution(world.size);
    } else {
      frags = A.getColumnDistribution(world.size);
    }

    vector<int> allRowSe;
    for (auto& frag : frags) {
      append(allRowSe, frag.row_se);
    }

    MPI_Iscatter(allRowSe.data(), myAInfo.rows + 1, MPI_INT,
                  myA.row_se.data(), myAInfo.rows + 1, MPI_INT, 0,
                  MPI_COMM_WORLD, requests);

    vector<int> allNnz;
    vector<int> allColIdx;
    vector<int> allDisp;

    for (auto& frag : frags) {
      append(allColIdx, frag.col_indx);
      allNnz.push_back(frag.nnz);
    }

    allDisp.push_back(0);
    for (int i = 0; i < world.size - 1; i++) {
      allDisp.push_back(allDisp.back() + allNnz[i]);
    }

    MPI_Iscatterv(allColIdx.data(), allNnz.data(), allDisp.data(), MPI_INT,
                  myA.col_indx.data(), myAInfo.nnz, MPI_INT, 0,
                  MPI_COMM_WORLD, requests + 1);

    vector<double> allValues;
    for (auto& frag : frags) {
      append(allValues, frag.values);
    }

    MPI_Iscatterv(allValues.data(), allNnz.data(), allDisp.data(),
                  MPI_DOUBLE, myA.values.data(), myAInfo.nnz, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD, requests + 2);

    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
  } else {
    {
      MPI_Request request;
      MPI_Iscatter(nullptr, MatrixInfo::size, MPI_INT, &myAInfo,
                   MatrixInfo::size, MPI_INT, 0, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      myA = SparseMatrix(myAInfo);
    }

    MPI_Iscatter(nullptr, myAInfo.rows + 1, MPI_INT, myA.row_se.data(),
                  myAInfo.rows + 1, MPI_INT, 0, MPI_COMM_WORLD, requests);

    MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_INT, myA.col_indx.data(),
                  myAInfo.nnz, MPI_INT, 0, MPI_COMM_WORLD, requests + 1);

    MPI_Iscatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, myA.values.data(),
                  myAInfo.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, requests + 2);

    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
  }

  myC = DenseMatrix(myAInfo, world.rank, world.size, config.seed);
}
