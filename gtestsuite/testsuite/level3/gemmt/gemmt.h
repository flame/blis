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
 *        op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
 * Only accesses and updates the upper or the lower triangular part.
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the symmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
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
static void gemmt_(char uplo, char transa, char transb, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in gemmt_().");
}

template<typename T>
static void gemmt_blis_impl(char uplo, char transa, char transb, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemmt_blis_impl( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemmt_blis_impl( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemmt_blis_impl( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemmt_blis_impl( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in gemmt_blis_impl().");
}

template<typename T>
static void cblas_gemmt(char storage, char uplo, char transa, char transb,
    gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;
    enum CBLAS_UPLO cblas_uplo;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_trans( transb, &cblas_transb );
    testinghelpers::char_to_cblas_uplo( uplo, &cblas_uplo );

    if constexpr (std::is_same<T, float>::value)
        cblas_sgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in cblas_gemmt().");
}

#ifdef TEST_BLIS_TYPED
template<typename T>
static void typed_gemmt(char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    trans_t transa, transb;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );

    uplo_t blis_uplo;
    testinghelpers::char_to_blis_uplo( uplo, &blis_uplo );

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
        bli_sgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in typed_gemmt().");
}
#endif

template<typename T>
static void gemmt( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    uplo = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    transb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char uplo_cpy = uplo;
    char transa_cpy = transa;
    char transb_cpy = transb;
    gtint_t n_cpy = n;
    gtint_t k_cpy = k;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t ldb_cpy = ldb;
    T* beta_cpy = beta;
    gtint_t ldc_cpy = ldc;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( storage, transa, n, k, lda );
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
        gemmt_<T>( uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        gemmt_blis_impl<T>( uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_gemmt<T>( storage, uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    //typed_gemmt<T>( storage, uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    throw std::runtime_error("Error in testsuite/level3/gemmt.h: BLIS-typed interface cannot be tested tested.");
#else
    throw std::runtime_error("Error in testsuite/level3/gemmt.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "uplo", uplo, uplo_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "transb", transb, transb_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "k", k, k_cpy );
    computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "ldb", ldb, ldb_cpy );
    computediff<T>( "beta", *beta, *beta_cpy );
    computediff<gtint_t>( "ldc", ldc, ldc_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        if(( transa == 'n' ) || ( transa == 'N' ))
            computediff<T>( "A", storage, n, k, ap, ap_cpy, lda, true );
        else
            computediff<T>( "A", storage, k, n, ap, ap_cpy, lda, true );
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
