#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * SYMV performs the matrix-vector  operation
 *    y := alpha*A*x + beta*y
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T* alpha, T *ap, gtint_t lda, T *xp, gtint_t incx, T* beta,
    T *yp, gtint_t incy );

} //end of namespace testinghelpers