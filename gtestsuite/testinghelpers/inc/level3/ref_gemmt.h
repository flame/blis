#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * GEMMT performs one of the matrix-matrix operations
 *    C := alpha*op( A )*op( B ) + beta*C,
 * where  op( X ) is one of
 *    op( X ) = X   or   op( X ) = A**T   or   op( X ) = X**H,
 * alpha and beta are scalars, and A, B and C are matrices, with op( A )
 * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 * Only accesses and updates the upper or the lower triangular part.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_gemmt (
    char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers