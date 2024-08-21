/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
 *             rho := conjx(x)^T * conjy(y)
 *          or rho := conjx(x)^T * conjy(y) (BLIS_TYPED only)
 * @param[in] conjx denotes if x or conj(x) will be used for this operation (BLIS API specific)
 * @param[in] conjy denotes if y or conj(y) will be used for this operation (BLIS API specific)
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 * @param[in,out] rho is a scalar
 */

template<typename T>
static void dotv_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, float>::value)
        *rho = sdot_(&n, x, &incx, y, &incy);
    else if constexpr (std::is_same<T, double>::value)
        *rho = ddot_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotu_(&n, x, &incx, y, &incy);
    #else
        cdotu_(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotu_(&n, x, &incx, y, &incy);
    #else
        zdotu_(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotv_().");
}

template<typename T>
static void dotu_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotu_(&n, x, &incx, y, &incy);
    #else
        cdotu_(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotu_(&n, x, &incx, y, &incy);
    #else
        zdotu_(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotu_().");
}

template<typename T>
static void dotc_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotc_(&n, x, &incx, y, &incy);
    #else
        cdotc_(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotc_(&n, x, &incx, y, &incy);
    #else
        zdotc_(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotc_().");
}

template<typename T>
static void dotv_blis_impl(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, float>::value)
        *rho = sdot_blis_impl(&n, x, &incx, y, &incy);
    else if constexpr (std::is_same<T, double>::value)
        *rho = ddot_blis_impl( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotu_blis_impl(&n, x, &incx, y, &incy);
    #else
        cdotu_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotu_blis_impl(&n, x, &incx, y, &incy);
    #else
        zdotu_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotv_blis_impl().");
}

template<typename T>
static void dotu_blis_impl(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotu_blis_impl(&n, x, &incx, y, &incy);
    #else
        cdotu_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotu_blis_impl(&n, x, &incx, y, &incy);
    #else
        zdotu_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotu_blis_impl().");
}

template<typename T>
static void dotc_blis_impl(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = cdotc_blis_impl(&n, x, &incx, y, &incy);
    #else
        cdotc_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else if constexpr (std::is_same<T, dcomplex>::value)
    #ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
        *rho = zdotc_blis_impl(&n, x, &incx, y, &incy);
    #else
        zdotc_blis_impl(rho, &n, x, &incx, y, &incy);
    #endif
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotc_blis_impl().");
}

template<typename T>
static void cblas_dotv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, float>::value)
        *rho = cblas_sdot( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        *rho = cblas_ddot( n, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in cblas_dotv().");
}

template<typename T>
static void cblas_dotu(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
        cblas_cdotu_sub( n, x, incx, y, incy, rho );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zdotu_sub( n, x, incx, y, incy, rho );
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in cblas_dotu().");
}

template<typename T>
static void cblas_dotc(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {
    if constexpr (std::is_same<T, scomplex>::value)
        cblas_cdotc_sub( n, x, incx, y, incy, rho );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zdotc_sub( n, x, incx, y, incy, rho );
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in cblas_dotc().");
}

template<typename T>
static void typed_dotv(char conj_x, char conj_y, gtint_t n,
  T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {

  conj_t conjx, conjy;
  // Map parameter characters to BLIS constants.
  testinghelpers::char_to_blis_conj( conj_x, &conjx );
  testinghelpers::char_to_blis_conj( conj_y, &conjy );
  if constexpr (std::is_same<T, float>::value)
    bli_sdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, double>::value)
    bli_ddotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, scomplex>::value)
    bli_cdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, dcomplex>::value)
    bli_zdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in typed_dotv().");
}

template<typename T>
static void dotv(char conjx, char conjy, gtint_t n,
  T* x, gtint_t incx, T* y, gtint_t incy, T* rho)
{

#ifdef TEST_UPPERCASE_ARGS
    conjx = static_cast<char>(std::toupper(static_cast<unsigned char>(conjx)));
    conjy = static_cast<char>(std::toupper(static_cast<unsigned char>(conjy)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char conjx_cpy = conjx;
    char conjy_cpy = conjy;
    gtint_t n_cpy = n;
    gtint_t incx_cpy = incx;
    gtint_t incy_cpy = incy;

    // Create copy of input arrays so we can check that they are not altered.
    T* x_cpy = nullptr;
    gtint_t size_x = testinghelpers::buff_dim( n, incx );
    if (x && size_x > 0)
    {
        x_cpy = new T[size_x];
        memcpy( x_cpy, x, size_x * sizeof( T ) );
    }
    T* y_cpy = nullptr;
    gtint_t size_y = testinghelpers::buff_dim( n, incy );
    if (y && size_y > 0)
    {
        y_cpy = new T[size_y];
        memcpy( y_cpy, y, size_y * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if constexpr ( testinghelpers::type_info<T>::is_real )
        dotv_<T>(n, x, incx, y, incy, rho);
    else if constexpr ( testinghelpers::type_info<T>::is_complex )
    {
        if ( testinghelpers::chkconj(conjx) )
            dotc_<T>(n, x, incx, y, incy, rho);
        else
            dotu_<T>(n, x, incx, y, incy, rho);
    }
#elif TEST_BLAS_BLIS_IMPL
    if constexpr ( testinghelpers::type_info<T>::is_real )
        dotv_blis_impl<T>(n, x, incx, y, incy, rho);
    else if constexpr ( testinghelpers::type_info<T>::is_complex )
    {
        if ( testinghelpers::chkconj(conjx) )
            dotc_blis_impl<T>(n, x, incx, y, incy, rho);
        else
            dotu_blis_impl<T>(n, x, incx, y, incy, rho);
    }
#elif TEST_CBLAS
    if constexpr ( testinghelpers::type_info<T>::is_real )
        cblas_dotv<T>(n, x, incx, y, incy, rho);
    else if constexpr ( testinghelpers::type_info<T>::is_complex )
    {
        if ( testinghelpers::chkconj(conjx) )
            cblas_dotc<T>(n, x, incx, y, incy, rho);
        else
            cblas_dotu<T>(n, x, incx, y, incy, rho);
    }
#elif TEST_BLIS_TYPED
    typed_dotv<T>(conjx, conjy, n, x, incx, y, incy, rho);
#else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "conjx", conjx, conjx_cpy );
    computediff<char>( "conjy", conjy, conjy_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (x && size_x > 0)
    {
        computediff<T>( "x", n, x, x_cpy, incx, true );
        delete[] x_cpy;
    }
    if (y && size_y > 0)
    {
        computediff<T>( "y", n, y, y_cpy, incy, true );
        delete[] y_cpy;
    }
#endif
}
