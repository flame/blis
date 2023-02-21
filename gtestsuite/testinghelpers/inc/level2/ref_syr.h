#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * SYR performs the symmetric rank 1 operation
 *    A := alpha*x*x**T + A,
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr( char storage, char uploa, char conjx, gtint_t n,
             T alpha, T *xp, gtint_t incx, T *ap, gtint_t lda );

} //end of namespace testinghelpers