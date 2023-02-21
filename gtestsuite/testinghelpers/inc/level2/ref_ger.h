#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * GER performs the rank 1 operation
 *    A := alpha*x*y**T + A,
 * where alpha is a scalar, x is an m element vector, y is an n element
 * vector and A is an m by n matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
    T alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda );

} //end of namespace testinghelpers