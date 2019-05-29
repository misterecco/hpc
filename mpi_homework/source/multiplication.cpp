#include <mkl.h>

#include "matrixmul.h"
#include "utils.h"

void multiplyLocal(SparseMatrix& A, const DenseMatrix& B, DenseMatrix& C,
                   bool use_mkl) {
  if (A.nnz == 0)
    return;

  if (use_mkl) {
    sparse_matrix_t myAMkl = A.toMklSparse();

    struct matrix_descr mType {
      .type = SPARSE_MATRIX_TYPE_GENERAL,
      // not relevant, but compiler complains without them
          .mode = SPARSE_FILL_MODE_FULL, .diag = SPARSE_DIAG_NON_UNIT,
    };

    // C := alpha*op(A)*B + beta*C
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, myAMkl, mType,
                    SPARSE_LAYOUT_COLUMN_MAJOR, B.values.data(), C.cols, B.rows,
                    1.0, C.values.data(), C.rows);
  } else {
    for (int i = 0; i < B.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        int firstK = A.row_se[i];
        int lastK = A.row_se[i + 1];

        for (int k = firstK; k < lastK; k++) {
          int col = A.col_indx[k];
          C.values[C.rows * j + i] += A.values[k] * B.values[B.rows * j + col];
        }
      }
    }
  }
}

void multiply(SparseMatrix& myA, DenseMatrix& myB, DenseMatrix& myC,
              const Config& config, const MpiGroup& layer) {
  for (int e = 0; e < config.exponent; e++) {
    MatrixInfo myCInfo = myC.getInfo();
    myB = myC;
    myC = DenseMatrix(myCInfo);

    int rounds =
        config.use_inner ? layer.size / config.repl_group_size : layer.size;

    for (int i = 0; i < rounds; i++) {
      shiftAandCompute(myA, layer, 1,
                       [&]() { multiplyLocal(myA, myB, myC, config.use_mkl); });
    }
  }
}
