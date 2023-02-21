#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  AXPYV performs vector operations
 *     y := beta * y + conjx(x)
 *     where x and y are vectors of length n, and beta is a scalar
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_xpbyv(char conjx, gtint_t len,
                        const T* xp, gtint_t incx, const T beta, T* yp, gtint_t incy);

} //end of namespace testinghelpers