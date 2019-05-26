#pragma once

#include "matrix.h"

void multiply(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C, bool use_mkl);
