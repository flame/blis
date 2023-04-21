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
 *        y := alpha*A*x + beta*y
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 */

template<typename T>
static void symv_( char uploa, gtint_t n, T* alpha, T* ap, gtint_t lda,
                    T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, float>::value)
        ssymv_( &uploa, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dsymv_( &uploa, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/symv.h: Invalid typename in symv_().");
}

template<typename T>
static void cblas_symv( char storage, char uploa, gtint_t n, T* alpha,
    T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uplo;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uplo );

    if constexpr (std::is_same<T, float>::value)
        cblas_ssymv( cblas_order, cblas_uplo, n, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dsymv( cblas_order, cblas_uplo, n, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/symv.h: Invalid typename in cblas_symv().");
}

template<typename T>
static void typed_symv( char storage, char uplo, char conj_a, char conj_x,
    gtint_t n, T* alpha, T* a, gtint_t lda, T* x, gtint_t incx, T* beta,
    T* y, gtint_t incy )
{
    uplo_t uploa;
    conj_t conja;
    conj_t conjx;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_conj ( conj_a, &conja );
    testinghelpers::char_to_blis_conj ( conj_x, &conjx );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else if( (storage == 'r') || (storage == 'R') )
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_ssymv( uploa, conja, conjx, n, alpha, a, rsa, csa, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_dsymv( uploa, conja, conjx, n, alpha, a, rsa, csa, x, incx, beta, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/symv.h: Invalid typename in typed_symv().");
}

template<typename T>
static void symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T* alpha, T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp,
    gtint_t incy )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        symv_<T>( uploa, n, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/symv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_symv<T>( storage, uploa, n, alpha, ap, lda, xp, incx, beta, yp, incy );
#elif TEST_BLIS_TYPED
    typed_symv<T>( storage, uploa, conja, conjx, n, alpha, ap, lda, xp, incx, beta, yp, incy );
#else
    throw std::runtime_error("Error in testsuite/level2/symv.h: No interfaces are set to be tested.");
#endif
}
