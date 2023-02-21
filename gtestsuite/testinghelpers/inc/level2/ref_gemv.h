#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * GEMV performs one of the matrix-vector operations
 *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
 *    y := alpha*A**H*x + beta*y,
 * ==========================================================================
*/

namespace testinghelpers {

template <typename T>
void ref_gemv(  char storage, char trans, char conjx, gtint_t m, gtint_t n,
    T alpha, T *ap, gtint_t lda, T *xp, gtint_t incx, T beta,
    T *yp, gtint_t incy );

} //end of namespace testinghelpers