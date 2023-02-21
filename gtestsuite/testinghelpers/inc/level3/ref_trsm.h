#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 *  TRSM  solves one of the matrix equations
 *     op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
 *  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 *  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *     op( A ) = A   or   op( A ) = A**T.
 *  The matrix X is overwritten on B.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trsm( char storage, char side, char uploa, char transa, char diaga,
    gtint_t m, gtint_t n, T alpha, T *ap, gtint_t lda, T *bp, gtint_t ldb );

} //end of namespace testinghelpers