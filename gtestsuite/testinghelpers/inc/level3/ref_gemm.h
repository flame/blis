#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 *  GEMM performs one of the matrix-matrix operations
 *     C := alpha*op( A )*op( B ) + beta*C,
 *  where  op( A ) is one of
 *     op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H,
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
   ==========================================================================
*/

namespace testinghelpers {

template <typename T>
void ref_gemm (
    char storage, char trnsa, char trnsb,
    gtint_t m, gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers