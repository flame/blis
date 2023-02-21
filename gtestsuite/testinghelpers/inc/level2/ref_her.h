#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * HER performs the hermitian rank 1 operation
 *    A := alpha*x*x**H + A
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n hermitian matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T, typename Tr>
void ref_her( char storage, char uploa, char conjx, gtint_t n,
    Tr alpha, T *xp, gtint_t incx, T *ap, gtint_t lda );

} //end of namespace testinghelpers