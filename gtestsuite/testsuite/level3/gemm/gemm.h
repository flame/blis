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
 *        C := alpha*op( A )*op( B ) + beta*C,
 * where  op( A ) is one of
 *        op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H,
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies  the number  of rows  of the  matrix
                         op( A )  and of the  matrix  C
 * @param[in]     n      specifies the number  of columns of the matrix
                         op( B ) and the number of columns of the matrix C
 * @param[in]     k      specifies  the number of columns of the matrix
                         op( A ) and the number of rows of the matrix op( B ).
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     bp     specifies pointer which points to the first element of bp
 * @param[in]     rsb    specifies row increment of bp.
 * @param[in]     csb    specifies column increment of bp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T>
static void gemm_(char transa, char transb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in gemm_().");
}

template<typename T>
static void gemm_blis_impl(char transa, char transb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemm_blis_impl( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemm_blis_impl( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemm_blis_impl( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemm_blis_impl( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in gemm_blis_impl().");
}

template<typename T>
static void cblas_gemm(char storage, char transa, char transb,
    gtint_t m, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_trans( transb, &cblas_transb );

    if constexpr (std::is_same<T, float>::value)
        cblas_sgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in cblas_gemm().");
}

template<typename T>
static void typed_gemm(char storage, char trnsa, char trnsb,
    gtint_t m, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    trans_t transa, transb;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );

    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    rsa=rsb=rsc=1;
    csa=csb=csc=1;
    /* a = m x k       b = k x n       c = m x n    */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csb = ldb ;
        csc = ldc ;
    }
    else if( (storage == 'r') || (storage == 'R') ) {
        rsa = lda ;
        rsb = ldb ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_sgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in typed_gemm().");
}

template<typename T>
static void gemm( char storage, char transa, char transb, gtint_t m, gtint_t n, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    transb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char transa_cpy = transa;
    char transb_cpy = transb;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    gtint_t k_cpy = k;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t ldb_cpy = ldb;
    T* beta_cpy = beta;
    gtint_t ldc_cpy = ldc;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( storage, transa, m, k, lda );
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
    T* bp_cpy = nullptr;
    gtint_t size_bp = testinghelpers::matsize( storage, transb, k, n, ldb );
    if (bp && size_bp > 0)
    {
        bp_cpy = new T[size_bp];
        memcpy( bp_cpy, bp, size_bp * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        gemm_<T>( transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        gemm_blis_impl<T>( transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_gemm<T>( storage, transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_gemm<T>( storage, transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/gemm.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "transb", transb, transb_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "k", k, k_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "ldb", ldb, ldb_cpy );
    if (beta) computediff<T>( "beta", *beta, *beta_cpy );
    computediff<gtint_t>( "ldc", ldc, ldc_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        if(( transa == 'n' ) || ( transa == 'N' ))
            computediff<T>( "A", storage, m, k, ap, ap_cpy, lda, true );
        else
            computediff<T>( "A", storage, k, m, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }

    if (bp && size_bp > 0)
    {
        if(( transb == 'n' ) || ( transb == 'N' ))
            computediff<T>( "B", storage, k, n, bp, bp_cpy, ldb, true );
        else
            computediff<T>( "B", storage, n, k, bp, bp_cpy, ldb, true );
        delete[] bp_cpy;
    }
#endif
}
