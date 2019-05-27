#pragma once

#include "config.h"
#include "matrix.h"

void initialize(SparseMatrixInfo& myAInfo, SparseMatrix& myA,
    SparseMatrixInfo& myCInfo, DenseMatrix& myC,
    const Config& config, const int myRank, const int numProcesses);

void replicate(SparseMatrix& myA, SparseMatrixInfo& myAInfo,
    MPI_Comm myReplGroup, int myGroupRank, int c);
