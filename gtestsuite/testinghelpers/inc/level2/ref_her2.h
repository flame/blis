#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * HER2  performs the hermitian rank 2 operation
 *    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
 * where alpha is a scalar, x and y are n element vectors and A is an n
 * by n hermitian matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_her2( char storage, char uploa, char conjx, char conjy, gtint_t n,
    T* alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda );

} //end of namespace testinghelpers