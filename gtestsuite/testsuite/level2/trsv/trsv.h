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
  *    x := alpha * inv(transa(A)) * x_orig
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     transa specifies the form of op( A ) to be used in matrix multiplication
 * @param[in]     diaga  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 * @param[in,out] xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.

 */

template<typename T>
static void trsv_( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        strsv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtrsv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrsv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrsv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trsv.h: Invalid typename in trsv_().");
}

template<typename T>
static void cblas_trsv( char storage, char uploa, char transa, char diaga,
                      gtint_t n, T *ap, gtint_t lda, T *xp, gtint_t incx )
{

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uploa;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_DIAG cblas_diaga;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uploa );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_diag( diaga, &cblas_diaga );

    if constexpr (std::is_same<T, float>::value)
        cblas_strsv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dtrsv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_ctrsv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_ztrsv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trsv.h: Invalid typename in cblas_trsv().");
}

template<typename T>
static void typed_trsv( char storage, char uplo, char trans, char diag,
            gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    uplo_t  uploa;
    trans_t transa;
    diag_t  diaga;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_trans( trans, &transa );
    testinghelpers::char_to_blis_diag ( diag, &diaga );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else if( (storage == 'r') || (storage == 'R') )
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_strsv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dtrsv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_ctrsv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_ztrsv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else

        throw std::runtime_error("Error in testsuite/level2/trsv.h: Invalid typename in typed_trsv().");
}

template<typename T>
static void trsv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
#if (defined TEST_BLAS || defined  TEST_CBLAS)
    T one;
    testinghelpers::initone(one);
#endif

#ifdef TEST_BLAS
    if(( storage == 'c' || storage == 'C' ))
        if( *alpha == one )
            trsv_<T>( uploa, transa, diaga, n, ap, lda, xp, incx );
        else
            throw std::runtime_error("Error in testsuite/level2/trsv.h: BLAS interface cannot be tested for alpha != one.");
    else
        throw std::runtime_error("Error in testsuite/level2/trsv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    if( *alpha == one )
        cblas_trsv<T>( storage, uploa, transa, diaga, n, ap, lda, xp, incx );
    else
      throw std::runtime_error("Error in testsuite/level2/trsv.h: CBLAS interface cannot be tested for alpha != one.");
#elif TEST_BLIS_TYPED
    typed_trsv<T>( storage, uploa, transa, diaga, n, alpha, ap, lda, xp, incx );
#else
    throw std::runtime_error("Error in testsuite/level2/trsv.h: No interfaces are set to be tested.");
#endif
}
