#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * TRMV  performs one of the matrix-vector operations
 *    x := alpha * transa(A) * x
 * where x is an n element vector and  A is an n by n unit, or non-unit,
 * upper or lower triangular matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trmv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx );

} //end of namespace testinghelpers