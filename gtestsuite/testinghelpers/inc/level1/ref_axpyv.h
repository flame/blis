#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  AXPYV performs vector operations
 *     y := y + alpha * conjx(x)
 *     where x and y are vectors of length n, and alpha is a scalar
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_axpyv(char conjx, gtint_t len, const T alpha,
                        const T* xp, gtint_t incx, T* yp, gtint_t incy);

} //end of namespace testinghelpers