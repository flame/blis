#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  DOTXV performs vector operations
 *     rho := beta * rho + alpha * conjx(x)^T * conjy(y)
 *     where x and y are vectors of length n, and alpha, beta, and rho are scalars.
 *  ==========================================================================
 */

namespace testinghelpers {

template<typename T>
void ref_dotxv(char conjx, char conjy, gtint_t n, const T alpha,
  const T* x, gtint_t incx, const T* y, gtint_t incy, const T beta, T* rho);

} //end of namespace testinghelpers