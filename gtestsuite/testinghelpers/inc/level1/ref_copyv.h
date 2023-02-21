#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  COPYV performs vector operations
 *     y := conjx(x)
 *     where x and y are vectors of length n.
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_copyv(char conjx, gtint_t n, const T* x, gtint_t incx,
                                            T* y, gtint_t incy);

} //end of namespace testinghelpers