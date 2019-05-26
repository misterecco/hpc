#include <mkl.h>

#include "math.h"

void multiply(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C, bool use_mkl) {
  if (A.nnz == 0) return;

  if (use_mkl) {
    sparse_matrix_t myAMkl = A.toMklSparse();

    struct matrix_descr mType {
        .type = SPARSE_MATRIX_TYPE_GENERAL,
        // not relevant, but compiler complains without them
        .mode = SPARSE_FILL_MODE_FULL,
        .diag = SPARSE_DIAG_NON_UNIT,
    };

    // C := alpha*op(A)*B + beta*C
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, myAMkl,
        mType, SPARSE_LAYOUT_COLUMN_MAJOR, B.values.data(),
        C.cols, B.rows, 1.0, C.values.data(), C.rows);
  } else {
    for (int i = 0; i < B.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        int firstK = A.rows_start[i];
        int lastK = A.rows_end[i];

        for (int k = firstK; k < lastK; k++) {
          int col = A.col_indx[k];
          C.values[C.rows * j + i] += A.values[k] * B.values[B.rows * j + col];
        }
      }
    }
  }
}
