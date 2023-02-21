#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  SCAL2V performs a vector operation
 *     y := alpha * conj(x) (BLIS interface only)
 *  where x and y are vectors of length n, and alpha is a scalar
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_scal2v(char conjx, gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy);

} //end of namespace testinghelpers