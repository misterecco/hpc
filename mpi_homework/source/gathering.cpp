#include "matrixmul.h"

// TODO: for InnerABC first reduce inside repl groups, then send to coordinator

void gatherC(const MatrixInfo& myCInfo, DenseMatrix& myC,
             const MpiGroup& world) {
  if (world.rank == 0) {
    DenseMatrix C(myCInfo);

    MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
               C.values.data(), myC.rows * myC.cols, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    C.print(myCInfo.actualRows);
  } else {
    MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE, nullptr,
               myC.rows * myC.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

void countGe(const MatrixInfo& myCInfo, const DenseMatrix& myC, double g,
             const MpiGroup& world) {
  int myCount = myC.countGreaterOrEqual(g, myCInfo.actualRows);
  int totalCount = 0;

  MPI_Reduce(&myCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world.rank == 0) {
    std::cout << totalCount << std::endl;
  }
}