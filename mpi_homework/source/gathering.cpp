#include "matrixmul.h"

void gatherAndPrintMatrix(const DenseMatrix& myC, const MatrixInfo cInfo,
                          const MpiGroup& group) {
  if (group.rank == 0) {
    DenseMatrix C(cInfo);

    MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE,
               C.values.data(), myC.rows * myC.cols, MPI_DOUBLE, 0, group.comm);

    C.print();
  } else {
    MPI_Gather(myC.values.data(), myC.rows * myC.cols, MPI_DOUBLE, nullptr,
               myC.rows * myC.cols, MPI_DOUBLE, 0, group.comm);
  }
}

int reduceCountGe(const DenseMatrix& myC, double g, const MpiGroup& group) {
  int myCount = myC.countGreaterOrEqual(g);
  int totalCount = 0;

  MPI_Reduce(&myCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, group.comm);

  return totalCount;

  if (group.rank == 0) {
    std::cout << totalCount << std::endl;
  }
}

void gatherC(const MatrixInfo& cInfo, DenseMatrix& myC, const MpiGroup& world,
             const MpiGroup& replGroup, const MpiGroup& layer,
             const Config& config, bool& isReducedToZeroLayer) {
  if (config.use_inner) {
    if (replGroup.rank == 0) {
      DenseMatrix myInitC = myC;
      MPI_Reduce(myInitC.values.data(), myC.values.data(), myC.rows * myC.cols,
                 MPI_DOUBLE, MPI_SUM, 0, replGroup.comm);
    } else {
      MPI_Reduce(myC.values.data(), nullptr, myC.rows * myC.cols, MPI_DOUBLE,
                 MPI_SUM, 0, replGroup.comm);
    }

    if (layer.color == 0) {
      gatherAndPrintMatrix(myC, cInfo, layer);
    }

    isReducedToZeroLayer = true;
  } else {
    gatherAndPrintMatrix(myC, cInfo, world);
  }
}

// TODO: for InnerABC first reduce inside repl groups, then send to coordinator
void countGe(const MatrixInfo& myCInfo, const DenseMatrix& myC,
             const MpiGroup& world, const MpiGroup& replGroup,
             const MpiGroup& layer, const Config& config,
             bool isReducedToZeroLayer) {
  if (config.use_inner) {
    if (isReducedToZeroLayer) {
      if (layer.color == 0) {
        int totalCount = reduceCountGe(myC, config.ge_value, layer);

        if (layer.rank == 0) {
          printf("%d\n", totalCount);
        }
      }
    } else {
      int replGroupCount = reduceCountGe(myC, config.ge_value, replGroup);

      if (layer.color == 0) {
        int totalCount = 0;

        MPI_Reduce(&replGroupCount, &totalCount, 1, MPI_INT, MPI_SUM, 0,
                   layer.comm);

        if (layer.rank == 0) {
          printf("%d\n", totalCount);
        }
      }
    }
  } else {
    int totalCount = reduceCountGe(myC, config.ge_value, world);

    if (world.rank == 0) {
      printf("%d\n", totalCount);
    }
  }
}
