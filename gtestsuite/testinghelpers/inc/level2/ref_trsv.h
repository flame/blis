#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * TRSV Solves a triangular system of equations with a single value for the
 *        right side
 *    b := alpha * inv(transa(A)) * x_orig
 * where b and x are n element vectors and A is an n by n unit, or non-unit,
 * upper or lower triangular matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trsv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx );

} //end of namespace testinghelpers