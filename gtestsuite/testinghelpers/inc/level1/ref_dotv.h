#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  DOTV performs vector operations
 *     rho := conjx(x)^T * conjy(y)
 *     where x and y are vectors of length n, and rho is a scalar.
 *  ==========================================================================
 */

namespace testinghelpers {
template<typename T>
void ref_dotv(gtint_t n, const T* x, gtint_t incx, const T* y,
                                          gtint_t incy, T* rho);

template<typename T>
void ref_dotv(char conjx, char conjy, gtint_t n, const T* x,
                  gtint_t incx, const T* y, gtint_t incy, T* rho);

} //end of namespace testinghelpers
