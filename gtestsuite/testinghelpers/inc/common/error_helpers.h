/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#pragma once

#include <limits>
#include "common/type_info.h"

namespace testinghelpers {

/**
 * @brief Returns the value of machine epsilon depending on the type.
 *        For scomplex and dcomplex, returns the value of machine epsilon
 *        for float and double, respectively.
 *        Epsilon is used as the basis for setting the threshold used for
 *        SUCCESS or FAILURE of tests.
 */
template<typename T>
double getEpsilon()
{
    using RT = typename testinghelpers::type_info<T>::real_type;
    double eps = std::numeric_limits<RT>::epsilon();
    return eps;
}

/**
 * @brief Returns the relative error. Relative error is used in most cases since
 *        it takes into account the magnitude of the exact and approx.
 *        For the cases where we are comparing very small values, that is values
 *        which are approximately zero, division with zero will cause inf/NaN.
 *        For example, if exact=0 and approx=0, getRelativeError() would return -NaN.
 */
template<typename T>
double getRelativeError(T exact, T approx)
{
    double rel_err;
    rel_err = std::abs(exact - approx)/std::abs(exact);
    return rel_err;
}

/**
 * @brief Returns the absolute error. Absolute error is used for the cases where
 *        we are comparing very small values, where relative error cannot be used.
 *        For example, on the example above where exact=0 and approx=0,
 *        getAbsoluteError() would return 0.
 *
 *        Absolute error doesn't take into account magnitude which means that for
 *        large values this could give false negatives.
 *        For example, if T is float, exact=598320.943 and approx=598320.9431,
 *        getAbsoluteError() would return 0.0001, compared to the relative error of ~2e-10.
 */
template<typename T>
double getAbsoluteError(T exact, T approx)
{
    double abs_err;
    abs_err = std::abs(exact - approx);
    return abs_err;
}

template<typename T>
double getError(T exact, T approx)
{
    if ( std::abs(exact) > 1 )
        return getRelativeError(exact, approx);
    else
        return getAbsoluteError(exact, approx);
}


} // end of testinghelpers namespace
