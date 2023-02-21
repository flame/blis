#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * SYR2  performs the symmetric rank 2 operation
 *    A := alpha*x*y**T + alpha*y*x**T + A,
 * where alpha is a scalar, x and y are n element vectors and A is an n
 * by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr2( char storage, char uploa, char conjx, char conjy, gtint_t n,
    T alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda );

} //end of namespace testinghelpers