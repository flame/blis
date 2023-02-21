#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  SCALV performs a vector operation
 *     x := alpha * x
 *  or x := conjalpha(alpha)*x (BLIS interface only)
 *  where x is a vector of length n, and alpha is a scalar
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_scalv(char conjalpha, gtint_t len, T alpha, T* x, gtint_t incx);

} //end of namespace testinghelpers