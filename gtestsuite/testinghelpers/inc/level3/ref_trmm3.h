#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * TRMM3  performs one of the matrix-matrix operations
 *    C := beta * C_orig + alpha * transa(A) * transb(B)
 * or
 *    C := beta * C_orig + alpha * transb(B) * transa(A)
 * where alpha and beta are scalars, A is an triangular matrix
 * and  B and C are m by n matrices.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trmm3( char storage, char side, char uploa, char transa, char diaga,
                char transb, gtint_t m, gtint_t n, T alpha, T *ap, gtint_t lda,
                T *bp, gtint_t ldb, T beta, T *c, gtint_t ldc );

} //end of namespace testinghelpers