#include <mpi.h>

#include "matrixmul.h"
#include "utils.h"

void replicate(SparseMatrix& myA, MatrixInfo& myAInfo, DenseMatrix& myC,
               MatrixInfo& myCInfo, const Config& config, const MpiGroup& world,
               const MpiGroup& replGroup, const MpiGroup& layer) {
  broadcastMatrix(myA, myAInfo, replGroup);

  if (!config.use_inner)
    return;

  int c = config.repl_group_size;
  int q = world.size / (c * c);
  int offset = layer.color * q;

  shiftAandCompute(myAInfo, myA, layer, offset, []() {});

  // IMPORTANT NOTE: be careful with C info
  broadcastMatrix(myC, myCInfo, replGroup);
}
