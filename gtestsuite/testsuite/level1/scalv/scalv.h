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
 * @brief Performs the operation:
 *             y := alpha * y
 *          or y := conj(alpha) * y (for BLIS interface only)
 * @param[in] conjalpha denotes if alpha or conj(alpha) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar
 * @param[in,out] x pointer which points to the first element of x
 * @param[in] incx increment of x
 */

template<typename T>
static void scalv_(gtint_t n, T alpha, T* x, gtint_t incx)
{
    if constexpr (std::is_same<T, float>::value)
        sscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zscal_( &n, &alpha, x, &incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in scalv_().");
}

template<typename T>
static void cblas_scalv(gtint_t n, T alpha, T* x, gtint_t incx)
{
    if constexpr (std::is_same<T, float>::value)
        cblas_sscal( n, alpha, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dscal( n, alpha, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cscal( n, &alpha, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zscal( n, &alpha, x, incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in cblas_scalv().");
}

template<typename T>
static void typed_scalv(char conj_alpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
    conj_t conjalpha;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_alpha, &conjalpha );
    if constexpr (std::is_same<T, float>::value)
        bli_sscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zscalv( conjalpha, n, &alpha, x, incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in typed_scalv().");
}


template<typename T>
static void scalv(char conj_alpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    scalv_<T>( n, alpha, x, incx );
#elif TEST_CBLAS
    cblas_scalv<T>( n, alpha, x, incx );
#elif TEST_BLIS_TYPED
    typed_scalv<T>( conj_alpha, n, alpha, x, incx );
#else
    throw std::runtime_error("Error in testsuite/level1/scalv.h: No interfaces are set to be tested.");
#endif
}
