#pragma once

#include "config.h"
#include "matrix.h"

void initialize(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
    const Config& config, const int myRank, const int numProcesses);
