#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  Given a vector of length n, return the zero-based index of
 *  the element of vector x that contains the largest absolute value
 *  (or, in the complex domain, the largest complex modulus).
 *  ==========================================================================
**/

namespace testinghelpers {

template<typename T>
gtint_t ref_amaxv(gtint_t n, const T* x, gtint_t incx);

} //end of namespace testinghelpers