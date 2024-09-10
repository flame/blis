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
 *              A := alpha*x*y**T + alpha*y*x**T + A,
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 * @param[in,out] ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 */

template<typename T>
static void syr2_( char uploa, gtint_t n, T* alpha, T* xp, gtint_t incx,
                              T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    if constexpr (std::is_same<T, float>::value)
        ssyr2_( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, double>::value)
        dsyr2_( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: Invalid typename in syr2_().");
}

template<typename T>
static void syr2_blis_impl( char uploa, gtint_t n, T* alpha, T* xp, gtint_t incx,
                              T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    if constexpr (std::is_same<T, float>::value)
        ssyr2_blis_impl( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, double>::value)
        dsyr2_blis_impl( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: Invalid typename in syr2_blis_impl().");
}

template<typename T>
static void cblas_syr2( char storage, char uploa, gtint_t n, T* alpha,
       T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uplo;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uplo );

    if constexpr (std::is_same<T, float>::value)
        cblas_ssyr2( cblas_order, cblas_uplo, n, *alpha, xp, incx, yp, incy, ap, lda );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dsyr2( cblas_order, cblas_uplo, n, *alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: Invalid typename in cblas_syr2().");
}

template<typename T>
static void typed_syr2( char storage, char uplo, char conj_x, char conj_y,
    gtint_t n, T* alpha, T* x, gtint_t incx, T* y, gtint_t incy,
    T* a, gtint_t lda )
{
    uplo_t uploa;
    conj_t conjx;
    conj_t conjy;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_conj ( conj_x, &conjx );
    testinghelpers::char_to_blis_conj ( conj_y, &conjy );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else if( (storage == 'r') || (storage == 'R') )
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_ssyr2( uploa, conjx, conjy, n, alpha, x, incx, y, incy, a, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dsyr2( uploa, conjx, conjy, n, alpha, x, incx, y, incy, a, rsa, csa );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: Invalid typename in typed_syr2().");
}

template<typename T>
static void syr2( char storage, char uploa, char conj_x, char conj_y, gtint_t n,
      T* alpha, T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
    conj_x = static_cast<char>(std::toupper(static_cast<unsigned char>(conj_x)));
    conj_y = static_cast<char>(std::toupper(static_cast<unsigned char>(conj_y)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char uploa_cpy = uploa;
    char conj_x_cpy = conj_x;
    char conj_y_cpy = conj_y;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t incx_cpy = incx;
    gtint_t incy_cpy = incy;
    gtint_t lda_cpy = lda;

    // Create copy of input arrays so we can check that they are not altered.
    T* xp_cpy = nullptr;
    gtint_t size_xp;
    size_xp = testinghelpers::buff_dim( n, incx );
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
        syr2_<T>( uploa, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        syr2_blis_impl<T>( uploa, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/syr2.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_syr2<T>( storage, uploa, n, alpha, xp, incx, yp, incy, ap, lda );
#elif TEST_BLIS_TYPED
    typed_syr2<T>( storage, uploa, conj_x, conj_y, n, alpha, xp, incx, yp, incy, ap, lda );
#else
    throw std::runtime_error("Error in testsuite/level2/syr2.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "uploa", uploa, uploa_cpy );
    computediff<char>( "conj_x", conj_x, conj_x_cpy );
    computediff<char>( "conj_y", conj_y, conj_y_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (xp && size_xp > 0)
    {
        computediff<T>( "x", n, xp, xp_cpy, incx, true );
        delete[] xp_cpy;
    }

    if (yp && size_yp > 0)
    {
        computediff<T>( "y", n, yp, yp_cpy, incy, true );
        delete[] yp_cpy;
    }
#endif
}
