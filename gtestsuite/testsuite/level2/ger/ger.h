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
 *          A := alpha*x*y**T + A,
 *       or A := A + alpha * conjx(x) * conjy(y)^T (BLIS_TYPED only)
 * @param[in]     conjy  specifies the form of xp to be used in
                         the vector multiplication (BLIS_TYPED only)
 * @param[in]     conjy  specifies the form of yp to be used in
                         the vector multiplication (BLIS_TYPED only)
 * @param[in]     m      specifies  the number  of rows  of the  matrix A
 * @param[in]     n      specifies the number  of columns of the matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 * @param[in,out] ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 */

template<typename T>
static void ger_( char conjy, gtint_t m, gtint_t n, T* alpha,
    T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    if constexpr (std::is_same<T, float>::value)
        sger_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, double>::value)
        dger_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, scomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        cgerc_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
      else
        cgeru_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    }
    else if constexpr (std::is_same<T, dcomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        zgerc_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
      else
        zgeru_( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    }
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: Invalid typename in ger_().");
}

template<typename T>
static void ger_blis_impl( char conjy, gtint_t m, gtint_t n, T* alpha,
    T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    if constexpr (std::is_same<T, float>::value)
        sger_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, double>::value)
        dger_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, scomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        cgerc_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
      else
        cgeru_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    }
    else if constexpr (std::is_same<T, dcomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        zgerc_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
      else
        zgeru_blis_impl( &m, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    }
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: Invalid typename in ger_blis_impl().");
}

template<typename T>
static void cblas_ger( char storage, char conjy, gtint_t m, gtint_t n,
    T* alpha, T* xp, gtint_t incx,T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    enum CBLAS_ORDER cblas_order;
    testinghelpers::char_to_cblas_order( storage, &cblas_order );

    if constexpr (std::is_same<T, float>::value)
        cblas_sger( cblas_order, m, n, *alpha, xp, incx, yp, incy, ap, lda );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dger( cblas_order, m, n, *alpha, xp, incx, yp, incy, ap, lda );
    else if constexpr (std::is_same<T, scomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        cblas_cgerc( cblas_order, m, n, alpha, xp, incx, yp, incy, ap, lda );
      else
        cblas_cgeru( cblas_order, m, n, alpha, xp, incx, yp, incy, ap, lda );
    }
    else if constexpr (std::is_same<T, dcomplex>::value) {
      if( testinghelpers::chkconj( conjy ) )
        cblas_zgerc( cblas_order, m, n, alpha, xp, incx, yp, incy, ap, lda );
      else
        cblas_zgeru( cblas_order, m, n, alpha, xp, incx, yp, incy, ap, lda );
    }
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: Invalid typename in cblas_ger().");
}

template<typename T>
static void typed_ger(char storage, char conj_x, char conj_y, gtint_t m, gtint_t n,
         T* alpha, T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    conj_t  conjx;
    conj_t  conjy;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj ( conj_x, &conjx );
    testinghelpers::char_to_blis_conj ( conj_y, &conjy );

    dim_t rsa,csa;

    rsa=csa=1;
    /* a = m x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else if( (storage == 'r') || (storage == 'R') )
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_sger( conjx, conjy, m, n, alpha, xp, incx, yp, incy, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dger( conjx, conjy, m, n, alpha, xp, incx, yp, incy, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cger( conjx, conjy, m, n, alpha, xp, incx, yp, incy, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zger( conjx, conjy, m, n, alpha, xp, incx, yp, incy, ap, rsa, csa );
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: Invalid typename in typed_ger().");
}

template<typename T>
static void ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
    T* alpha, T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    conjx = static_cast<char>(std::toupper(static_cast<unsigned char>(conjx)));
    conjy = static_cast<char>(std::toupper(static_cast<unsigned char>(conjy)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char conjx_cpy = conjx;
    char conjy_cpy = conjy;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t incx_cpy = incx;
    gtint_t incy_cpy = incy;
    gtint_t lda_cpy = lda;

    // Create copy of input arrays so we can check that they are not altered.
    T* xp_cpy = nullptr;
    gtint_t size_xp;
    size_xp = testinghelpers::buff_dim( m, incx );
    {
        xp_cpy = new T[size_xp];
        memcpy( xp_cpy, xp, size_xp * sizeof( T ) );
    }
    T* yp_cpy = nullptr;
    gtint_t size_yp;
    size_yp = testinghelpers::buff_dim( n, incy );
    {
        yp_cpy = new T[size_yp];
        memcpy( yp_cpy, yp, size_yp * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        ger_<T>( conjy, m, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        ger_blis_impl<T>( conjy, m, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/ger.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_ger<T>( storage, conjy, m, n, alpha, xp, incx, yp, incy, ap, lda );
#elif TEST_BLIS_TYPED
    typed_ger<T>( storage, conjx, conjy, m, n, alpha, xp, incx, yp, incy, ap, lda );
#else
    throw std::runtime_error("Error in testsuite/level2/ger.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "conjx", conjx, conjx_cpy );
    computediff<char>( "conjy", conjy, conjy_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (xp && size_xp > 0)
    {
        computediff<T>( "x", m, xp, xp_cpy, incx, true );
        delete[] xp_cpy;
    }

    if (yp && size_yp > 0)
    {
        computediff<T>( "y", n, yp, yp_cpy, incy, true );
        delete[] yp_cpy;
    }
#endif
}
