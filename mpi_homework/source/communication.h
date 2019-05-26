#pragma once

#include "config.h"
#include "matrix.h"

void initialDistibution(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
    const Config& config, const int myRank, const int numProcesses);
