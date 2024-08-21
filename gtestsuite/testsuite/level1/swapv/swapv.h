/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "inc/check_error.h"

/**
 * @brief Performs the operation:
 *     x <=> y
 * @param[in] n vector length of x and y
 * @param[in,out] x pointer which points to the first element of x
 * @param[in,out] y pointer which points to the first element of y
 * @param[in] incx increment of x
 * @param[in] incy increment of y
 */

template<typename T>
static void swapv_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{

    if constexpr (std::is_same<T, float>::value)
        sswap_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dswap_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cswap_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zswap_( &n, x, &incx, y, &incy );
    else
        throw std::runtime_error("Error in testsuite/level1/swapv.h: Invalid typename in swapv_().");
}

template<typename T>
static void swapv_blis_impl(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{

    if constexpr (std::is_same<T, float>::value)
        sswap_blis_impl( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dswap_blis_impl( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cswap_blis_impl( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zswap_blis_impl( &n, x, &incx, y, &incy );
    else
        throw std::runtime_error("Error in testsuite/level1/swapv.h: Invalid typename in swapv_blis_impl().");
}

template<typename T>
static void cblas_swapv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{

    if constexpr (std::is_same<T, float>::value)
        cblas_sswap( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dswap( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cswap( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zswap( n, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/swapv.h: Invalid typename in cblas_swapv().");
}

template<typename T>
static void typed_swapv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
    if constexpr (std::is_same<T, float>::value)
        bli_sswapv( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_dswapv( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cswapv( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zswapv( n, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/swapv.h: Invalid typename in typed_swapv().");

}

template<typename T>
static void swapv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    gtint_t n_cpy = n;
    gtint_t incx_cpy = incx;
    gtint_t incy_cpy = incy;
#endif

#ifdef TEST_BLAS
    swapv_<T>( n, x, incx, y, incy );
#elif TEST_BLAS_BLIS_IMPL
    swapv_blis_impl<T>( n, x, incx, y, incy );
#elif TEST_CBLAS
    cblas_swapv<T>( n, x, incx, y, incy );
#elif TEST_BLIS_TYPED
    typed_swapv<T>( n, x, incx, y, incy );
#else
    throw std::runtime_error("Error in testsuite/level1/swapv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );
#endif
}

