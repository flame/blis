#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  SUBV performs vector operations
 *     y := y - conjx(x)
 *     where x and y are vectors of length n
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
void ref_subv(char conjx, gtint_t len, const T* X, gtint_t incx, T* Y, gtint_t incy);

} //end of namespace testinghelpers