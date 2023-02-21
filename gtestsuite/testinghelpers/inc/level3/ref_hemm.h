#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * For BLIS-typed interface HEMM performs one of the matrix-matrix operations
 *    C := alpha*conj( A )*trans( B ) + beta*C, if side is left
 * or C := alpha*trans( B )*conj( A ) + beta*C, if side is right
 * alpha and beta are scalars, and A is Hermitian, B and C are matrices, with conj( A )
 * an m by m matrix, and trans( B ) and C m by n matrices.
 *
 * For BLAS/CBLAS interface HEMM performs one of the matrix-matrix operations
 *    C := alpha*A*B + beta*C, if side is left
 * or C := alpha*B*A + beta*C, if side is right
 * alpha and beta are scalars, and A is Hermitian, B and C are matrices, with A
 * an m by m matrix, and B and C m by n matrices.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_hemm(
    char storage, char side, char uplo, char conja, char transb,
    gtint_t m, gtint_t n,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers