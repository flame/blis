#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  AXPYV performs vector operations
 *     y := beta * y + alpha * conjx(x)
 *     where x and y are vectors of length n, and alpha, beta are scalars
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_axpbyv(char conjx, gtint_t len, const T alpha,
                        const T* xp, gtint_t incx, const T beta, T* yp, gtint_t incy);

} //end of namespace testinghelpers