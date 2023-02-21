#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * TRMM  performs one of the matrix-matrix operations
 *    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
 * where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
 * non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trmm( char storage, char side, char uploa, char transa, char diaga,
    gtint_t m, gtint_t n, T alpha, T *ap, gtint_t lda, T *bp, gtint_t ldb );

} //end of namespace testinghelpers