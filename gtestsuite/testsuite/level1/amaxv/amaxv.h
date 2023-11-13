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

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Finds the index of the first element that has the maximum absolute value.
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 *
 * If n < 1 or incx <= 0, return 0.
 */

template<typename T>
static gtint_t amaxv_(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
        idx = isamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        idx = idamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        idx = icamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        idx = izamax_( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in amaxv_().");

    // Since we are comparing against CBLAS which is 0-based and BLAS is 1-based,
    // we need to use -1 here.
    return (idx-1);
}

template<typename T>
static gtint_t cblas_amaxv(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
      idx = cblas_isamax( n, x, incx );
    else if constexpr (std::is_same<T, double>::value)
      idx = cblas_idamax( n, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
      idx = cblas_icamax( n, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
      idx = cblas_izamax( n, x, incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in cblas_amaxv().");

    return idx;
}

template<typename T>
static gtint_t typed_amaxv(gtint_t n, T* x, gtint_t incx)
{
    gtint_t idx = 0;
    if constexpr (std::is_same<T, float>::value)
        bli_samaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, double>::value)
        bli_damaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_camaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zamaxv( n, x, incx, &idx );
    else
        throw std::runtime_error("Error in testsuite/level1/amaxddv.h: Invalid typename in typed_amaxv().");

    return idx;
}

template<typename T>
static gtint_t amaxv(gtint_t n, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    return amaxv_<T>(n, x, incx);
#elif TEST_CBLAS
    return cblas_amaxv<T>(n, x, incx);
#elif TEST_BLIS_TYPED
    return typed_amaxv(n, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/amaxv.h: No interfaces are set to be tested.");
#endif
}
