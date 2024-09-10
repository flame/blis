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
 *        B := alpha*op( A )*B,   or   B := alpha*B*op( A )
 * where  op( A ) is one of
 *        op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H,
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the symmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     uplo   specifies if the upper or lower triangular part of A is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     diaga  specifies whether upper or lower triangular part of the matrix A
 * @param[in]     m      specifies the number of rows and cols of the  matrix
                         op( A ) and rows of the matrix C and B
 * @param[in]     n      specifies the number of columns of the matrix
                         op( B ) and the number of columns of the matrix C
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in,out] bp     specifies pointer which points to the first element of bp
 * @param[in]     rsb    specifies row increment of bp.
 * @param[in]     csb    specifies column increment of bp.
 */

template<typename T>
static void trmm_( char side, char uploa, char transa, char diaga, gtint_t m,
               gtint_t n, T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb )
{
    if constexpr (std::is_same<T, float>::value)
        strmm_( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, double>::value)
        dtrmm_( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrmm_( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrmm_( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: Invalid typename in trmm_().");
}

template<typename T>
static void trmm_blis_impl( char side, char uploa, char transa, char diaga, gtint_t m,
               gtint_t n, T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb )
{
    if constexpr (std::is_same<T, float>::value)
        strmm_blis_impl( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, double>::value)
        dtrmm_blis_impl( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrmm_blis_impl( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrmm_blis_impl( &side, &uploa, &transa, &diaga, &m, &n, alpha, ap, &lda, bp, &ldb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: Invalid typename in trmm_blis_impl().");
}

template<typename T>
static void cblas_trmm( char storage, char side, char uploa, char transa,
    char diaga, gtint_t m, gtint_t n, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_SIDE cblas_side;
    enum CBLAS_UPLO cblas_uploa;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_DIAG cblas_diaga;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_side( side, &cblas_side );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uploa );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_diag( diaga, &cblas_diaga );

    if constexpr (std::is_same<T, float>::value)
        cblas_strmm( cblas_order, cblas_side, cblas_uploa, cblas_transa, cblas_diaga, m, n, *alpha, ap, lda, bp, ldb );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dtrmm( cblas_order, cblas_side, cblas_uploa, cblas_transa, cblas_diaga, m, n, *alpha, ap, lda, bp, ldb );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_ctrmm( cblas_order, cblas_side, cblas_uploa, cblas_transa, cblas_diaga, m, n, alpha, ap, lda, bp, ldb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_ztrmm( cblas_order, cblas_side, cblas_uploa, cblas_transa, cblas_diaga, m, n, alpha, ap, lda, bp, ldb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: Invalid typename in cblas_trmm().");
}

template<typename T>
static void typed_trmm( char storage, char side, char uplo, char trans,
    char diag, gtint_t m, gtint_t n, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb )
{
    side_t  sidea;
    uplo_t  uploa;
    trans_t transa;
    diag_t  diaga;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_side( side, &sidea );
    testinghelpers::char_to_blis_uplo( uplo, &uploa );
    testinghelpers::char_to_blis_trans( trans, &transa );
    testinghelpers::char_to_blis_diag( diag, &diaga );

    dim_t rsa,csa;
    dim_t rsb,csb;

    rsa=rsb=1;
    csa=csb=1;
    /* a = m x m       b = m x n  */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csb = ldb ;
    }
    else if( (storage == 'r') || (storage == 'R') ) {
        rsa = lda ;
        rsb = ldb ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_strmm( sidea, uploa, transa, diaga, m, n, alpha, ap, rsa, csa, bp, rsb, csb );
    else if constexpr (std::is_same<T, double>::value)
        bli_dtrmm( sidea, uploa, transa, diaga, m, n, alpha, ap, rsa, csa, bp, rsb, csb );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_ctrmm( sidea, uploa, transa, diaga, m, n, alpha, ap, rsa, csa, bp, rsb, csb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_ztrmm( sidea, uploa, transa, diaga, m, n, alpha, ap, rsa, csa, bp, rsb, csb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: Invalid typename in typed_trmm().");
}

template<typename T>
static void trmm( char storage, char side, char uploa, char transa, char diaga,
    gtint_t m, gtint_t n, T *alpha, T *ap, gtint_t lda, T *bp, gtint_t ldb )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    side = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    diaga = static_cast<char>(std::toupper(static_cast<unsigned char>(diaga)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char side_cpy = side;
    char uploa_cpy = uploa;
    char transa_cpy = transa;
    char diaga_cpy = diaga;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t ldb_cpy = ldb;

    gtint_t mn;
    testinghelpers::set_dim_with_side( side, m, n, &mn );
    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( storage, transa, mn, mn, lda );
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        trmm_<T>( side, uploa, transa, diaga, m, n, alpha, ap, lda, bp, ldb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        trmm_blis_impl<T>( side, uploa, transa, diaga, m, n, alpha, ap, lda, bp, ldb );
    else
        throw std::runtime_error("Error in testsuite/level3/trmm.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_trmm<T>( storage, side, uploa, transa, diaga, m, n, alpha, ap, lda, bp, ldb );
#elif TEST_BLIS_TYPED
    typed_trmm<T>( storage, side, uploa, transa, diaga, m, n, alpha, ap, lda, bp, ldb );
#else
    throw std::runtime_error("Error in testsuite/level3/trmm.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "side", side, side_cpy );
    computediff<char>( "uploa", uploa, uploa_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "diaga", diaga, diaga_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "ldb", ldb, ldb_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        computediff<T>( "A", storage, mn, mn, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }
#endif
}
