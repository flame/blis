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
 * For BLIS-typed API:
 *        C := alpha*A*A**T + beta*C
 *     or C := alpha*A**T*A + beta*C
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     uplo   specifies if the upper or lower triangular part of C is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     n      specifies the number of rows and cols of C
 * @param[in]     k      specifies the number of rows of A, in case of transa = 'C',
 *                       and the columns of A otherwise.
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T>
static void syrk_(char uplo, char transa, gtint_t m, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        ssyrk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dsyrk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        csyrk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zsyrk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/syrk.h: Invalid typename in syrk_().");
}

template<typename T>
static void cblas_syrk(char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uplo;
    enum CBLAS_TRANSPOSE cblas_transa;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uplo, &cblas_uplo );
    testinghelpers::char_to_cblas_trans( trnsa, &cblas_transa );

    if constexpr (std::is_same<T, float>::value)
        cblas_ssyrk( cblas_order, cblas_uplo, cblas_transa, m, k, *alpha, ap, lda, *beta, cp, ldc );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dsyrk( cblas_order, cblas_uplo, cblas_transa, m, k, *alpha, ap, lda, *beta, cp, ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_csyrk( cblas_order, cblas_uplo, cblas_transa, m, k, alpha, ap, lda, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zsyrk( cblas_order, cblas_uplo, cblas_transa, m, k, alpha, ap, lda, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/syrk.h: Invalid typename in cblas_syrk().");
}

template<typename T>
static void typed_syrk(char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* beta, T* cp, gtint_t ldc)
{
    trans_t transa;
    uplo_t blis_uplo;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_uplo( uplo, &blis_uplo );
    dim_t rsa,csa;
    dim_t rsc,csc;

    rsa=rsc=1;
    csa=csc=1;
    /* a = m x k   c = m x m    */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csc = ldc ;
    }
    else if( (storage == 'r') || (storage == 'R') ) {
        rsa = lda ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_ssyrk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dsyrk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_csyrk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zsyrk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/syrk.h: Invalid typename in typed_syrk().");
}

template<typename T>
static void syrk( char storage, char uplo, char transa, gtint_t m, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        syrk_<T>( uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/syrk.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_syrk<T>( storage, uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_syrk<T>( storage, uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/syrk.h: No interfaces are set to be tested.");
#endif
}
