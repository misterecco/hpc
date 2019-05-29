#include <mpi.h>

#include "matrixmul.h"
#include "utils.h"

void replicate(SparseMatrix& myA, DenseMatrix& myC, const Config& config,
               const MpiGroup& world, const MpiGroup& replGroup,
               const MpiGroup& layer) {
  broadcastMatrix(myA, replGroup);

  if (!config.use_inner)
    return;

  int c = config.repl_group_size;
  int q = world.size / (c * c);
  int offset = layer.color * q;

  shiftAandCompute(myA, layer, offset, []() {});

  broadcastMatrix(myC, replGroup);
}
