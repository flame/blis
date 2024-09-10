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
 *        C := op( A )*op( B ) + beta*C,
 * where  op( A ) is one of
 *        op( A ) = alpha * A   or   op( A ) = alpha * A**T
 *        op( A ) = A           or   op( A ) = A**T
 *        op( B ) is one of
 *        op( B ) = alpha * B   or   op( B ) = alpha * B**T
 *        op( B ) = B           or   op( B ) = B**T
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication.
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication.
 * @param[in]     packa  specifies whether to reorder op( A ).
 * @param[in]     packb  specifies whether to reorder op( B ).
 * @param[in]     m      specifies  the number  of rows  of the  matrix
                         op( A )  and of the  matrix  C.
 * @param[in]     n      specifies the number  of columns of the matrix
                         op( B ) and the number of columns of the matrix C.
 * @param[in]     k      specifies  the number of columns of the matrix
                         op( A ) and the number of rows of the matrix op( B ).
 * @param[in]     ap     specifies pointer which points to the first element of ap.
 * @param[in]     lda    specifies the leading dimension of ap.
 * @param[in]     bp     specifies pointer which points to the first element of bp.
 * @param[in]     ldb    specifies the leading dimension of bp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp.
 * @param[in]     ldc    specifies the leading dimension of cp.
 */

#ifdef TEST_BLAS
template<typename T>
static void gemm_compute_(char transa, char transb, char packa, char packb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    T unit_alpha = 1.0;
    err_t err = BLIS_SUCCESS;
    if constexpr (std::is_same<T, float>::value)
    {
        if ( ( packa == 'P' || packa == 'p' ) && ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = sgemm_pack_get_size_( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            float* aBuffer = (float*) bli_malloc_user( bufSizeA, &err );
            sgemm_pack_( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         &unit_alpha,
                         ap,
                         &lda,
                         aBuffer );

            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = sgemm_pack_get_size_( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            float* bBuffer = (float*) bli_malloc_user( bufSizeB, &err );
            sgemm_pack_( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            sgemm_compute_( &packa, &packb, &m, &n, &k, aBuffer, &lda, bBuffer, &ldb, beta, cp, &ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( ( packa == 'P' || packa == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = sgemm_pack_get_size_( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            float* aBuffer = (float*) bli_malloc_user( bufSizeA, &err );
            sgemm_pack_( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         alpha,
                         ap,
                         &lda,
                         aBuffer );

            sgemm_compute_( &packa, &transb, &m, &n, &k, aBuffer, &lda, bp, &ldb, beta, cp, &ldc );
            bli_free_user( aBuffer );
        }
        else if ( ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = sgemm_pack_get_size_( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            float* bBuffer = (float*) bli_malloc_user( bufSizeB, &err );
            sgemm_pack_( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            sgemm_compute_( &transa, &packb, &m, &n, &k, ap, &lda, bBuffer, &ldb, beta, cp, &ldc );
            bli_free_user( bBuffer );
        }
        else
        {
            sgemm_compute_( &transa, &transb, &m, &n, &k, ap, &lda, bp, &ldb, beta, cp, &ldc );
        }
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        if ( ( packa == 'P' || packa == 'p' ) && ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = dgemm_pack_get_size_( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            double* aBuffer = (double*) bli_malloc_user( bufSizeA, &err );
            dgemm_pack_( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         &unit_alpha,
                         ap,
                         &lda,
                         aBuffer );

            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = dgemm_pack_get_size_( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            double* bBuffer = (double*) bli_malloc_user( bufSizeB, &err );
            dgemm_pack_( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            dgemm_compute_( &packa, &packb, &m, &n, &k, aBuffer, &lda, bBuffer, &ldb, beta, cp, &ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( ( packa == 'P' || packa == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = dgemm_pack_get_size_( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            double* aBuffer = (double*) bli_malloc_user( bufSizeA, &err );
            dgemm_pack_( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         alpha,
                         ap,
                         &lda,
                         aBuffer );

            dgemm_compute_( &packa, &transb, &m, &n, &k, aBuffer, &lda, bp, &ldb, beta, cp, &ldc );
            bli_free_user( aBuffer );
        }
        else if ( ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = dgemm_pack_get_size_( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            double* bBuffer = (double*) bli_malloc_user( bufSizeB, &err );
            dgemm_pack_( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            dgemm_compute_( &transa, &packb, &m, &n, &k, ap, &lda, bBuffer, &ldb, beta, cp, &ldc );
            bli_free_user( bBuffer );
        }
        else
        {
            dgemm_compute_( &transa, &transb, &m, &n, &k, ap, &lda, bp, &ldb, beta, cp, &ldc );
        }
    }
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in gemm_compute_().");
}
#endif

template<typename T>
static void gemm_compute_blis_impl(char transa, char transb, char packa, char packb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    T unit_alpha = 1.0;
    err_t err = BLIS_SUCCESS;
    if constexpr (std::is_same<T, float>::value)
    {
        if ( ( packa == 'P' || packa == 'p' ) && ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = sgemm_pack_get_size_blis_impl( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            float* aBuffer = (float*) bli_malloc_user( bufSizeA, &err );
            sgemm_pack_blis_impl( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         &unit_alpha,
                         ap,
                         &lda,
                         aBuffer );

            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = sgemm_pack_get_size_blis_impl( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            float* bBuffer = (float*) bli_malloc_user( bufSizeB, &err );
            sgemm_pack_blis_impl( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            sgemm_compute_blis_impl( &packa, &packb, &m, &n, &k, aBuffer, &lda, bBuffer, &ldb, beta, cp, &ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( ( packa == 'P' || packa == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = sgemm_pack_get_size_blis_impl( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            float* aBuffer = (float*) bli_malloc_user( bufSizeA, &err );
            sgemm_pack_blis_impl( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         alpha,
                         ap,
                         &lda,
                         aBuffer );

            sgemm_compute_blis_impl( &packa, &transb, &m, &n, &k, aBuffer, &lda, bp, &ldb, beta, cp, &ldc );
            bli_free_user( aBuffer );
        }
        else if ( ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = sgemm_pack_get_size_blis_impl( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            float* bBuffer = (float*) bli_malloc_user( bufSizeB, &err );
            sgemm_pack_blis_impl( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            sgemm_compute_blis_impl( &transa, &packb, &m, &n, &k, ap, &lda, bBuffer, &ldb, beta, cp, &ldc );
            bli_free_user( bBuffer );
        }
        else
        {
            sgemm_compute_blis_impl( &transa, &transb, &m, &n, &k, ap, &lda, bp, &ldb, beta, cp, &ldc );
        }
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        if ( ( packa == 'P' || packa == 'p' ) && ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = dgemm_pack_get_size_blis_impl( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            double* aBuffer = (double*) bli_malloc_user( bufSizeA, &err );
            dgemm_pack_blis_impl( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         &unit_alpha,
                         ap,
                         &lda,
                         aBuffer );

            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = dgemm_pack_get_size_blis_impl( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            double* bBuffer = (double*) bli_malloc_user( bufSizeB, &err );
            dgemm_pack_blis_impl( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            dgemm_compute_blis_impl( &packa, &packb, &m, &n, &k, aBuffer, &lda, bBuffer, &ldb, beta, cp, &ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( ( packa == 'P' || packa == 'p' ) )
        {
            // Reorder A
            char identifierA = 'A';
            gtint_t bufSizeA = dgemm_pack_get_size_blis_impl( &identifierA,
                                                     &m,
                                                     &n,
                                                     &k );

            double* aBuffer = (double*) bli_malloc_user( bufSizeA, &err );
            dgemm_pack_blis_impl( &identifierA,
                         &transa,
                         &m,
                         &n,
                         &k,
                         alpha,
                         ap,
                         &lda,
                         aBuffer );

            dgemm_compute_blis_impl( &packa, &transb, &m, &n, &k, aBuffer, &lda, bp, &ldb, beta, cp, &ldc );
            bli_free_user( aBuffer );
        }
        else if ( ( packb == 'P' || packb == 'p' ) )
        {
            // Reorder B
            char identifierB = 'B';
            gtint_t bufSizeB = dgemm_pack_get_size_blis_impl( &identifierB,
                                                     &m,
                                                     &n,
                                                     &k );

            double* bBuffer = (double*) bli_malloc_user( bufSizeB, &err );
            dgemm_pack_blis_impl( &identifierB,
                         &transb,
                         &m,
                         &n,
                         &k,
                         alpha,
                         bp,
                         &ldb,
                         bBuffer );

            dgemm_compute_blis_impl( &transa, &packb, &m, &n, &k, ap, &lda, bBuffer, &ldb, beta, cp, &ldc );
            bli_free_user( bBuffer );
        }
        else
        {
            dgemm_compute_blis_impl( &transa, &transb, &m, &n, &k, ap, &lda, bp, &ldb, beta, cp, &ldc );
        }
    }
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in gemm_compute_blis_impl().");
}

template<typename T>
static void cblas_gemm_compute(char storage, char transa, char transb, char pcka, char pckb,
    gtint_t m, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_trans( transb, &cblas_transb );

    T unit_alpha = 1.0;
    CBLAS_IDENTIFIER cblas_identifierA = CblasAMatrix;
    CBLAS_IDENTIFIER cblas_identifierB = CblasBMatrix;
    CBLAS_STORAGE cblas_packed = CblasPacked;

    err_t err = BLIS_SUCCESS;

    if constexpr (std::is_same<T, float>::value)
    {
        if ( ( pcka == 'p' || pcka == 'P' ) && ( pckb == 'p' || pckb == 'P' ) )
        {
            gtint_t bufSizeA = cblas_sgemm_pack_get_size( cblas_identifierA,
                                                         m,
                                                         n,
                                                         k );

            T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

            cblas_sgemm_pack( cblas_order, cblas_identifierA, cblas_transa,
                m, n, k, *alpha, ap, lda, aBuffer );

            gtint_t bufSizeB = cblas_sgemm_pack_get_size( cblas_identifierB,
                                                         m,
                                                         n,
                                                         k );

            T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

            cblas_sgemm_pack( cblas_order, cblas_identifierB, cblas_transb,
                m, n, k, unit_alpha, bp, ldb, bBuffer );

            cblas_sgemm_compute( cblas_order, cblas_packed, cblas_packed,
                m, n, k, aBuffer, lda, bBuffer, ldb, *beta, cp, ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( pcka == 'p' || pcka == 'P' )
        {
            gtint_t bufSizeA = cblas_sgemm_pack_get_size( cblas_identifierA,
                                                         m,
                                                         n,
                                                         k );

            T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

            cblas_sgemm_pack( cblas_order, cblas_identifierA, cblas_transa,
                m, n, k, *alpha, ap, lda, aBuffer );


            cblas_sgemm_compute( cblas_order, cblas_packed, cblas_transb,
                m, n, k, aBuffer, lda, bp, ldb, *beta, cp, ldc );

            bli_free_user( aBuffer );
        }
        else if ( pckb == 'p' || pckb == 'P' )
        {
            gtint_t bufSizeB = cblas_sgemm_pack_get_size( cblas_identifierB,
                                                         m,
                                                         n,
                                                         k );

            T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

            cblas_sgemm_pack( cblas_order, cblas_identifierB, cblas_transb,
                m, n, k, *alpha, bp, ldb, bBuffer );

            cblas_sgemm_compute( cblas_order, cblas_transa, cblas_packed,
                m, n, k, ap, lda, bBuffer, ldb, *beta, cp, ldc );

            bli_free_user( bBuffer );
        }
        else
        {
            cblas_sgemm_compute( cblas_order, cblas_transa, cblas_transb,
                m, n, k, ap, lda, bp, ldb, *beta, cp, ldc );
        }
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        if ( ( pcka == 'p' || pcka == 'P' ) && ( pckb == 'p' || pckb == 'P' ) )
        {
            gtint_t bufSizeA = cblas_dgemm_pack_get_size( cblas_identifierA,
                                                         m,
                                                         n,
                                                         k );

            T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

            cblas_dgemm_pack( cblas_order, cblas_identifierA, cblas_transa,
                m, n, k, *alpha, ap, lda, aBuffer );

            gtint_t bufSizeB = cblas_dgemm_pack_get_size( cblas_identifierB,
                                                         m,
                                                         n,
                                                         k );

            T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

            cblas_dgemm_pack( cblas_order, cblas_identifierB, cblas_transb,
                m, n, k, unit_alpha, bp, ldb, bBuffer );

            cblas_dgemm_compute( cblas_order, cblas_packed, cblas_packed,
                m, n, k, aBuffer, lda, bBuffer, ldb, *beta, cp, ldc );

            bli_free_user( aBuffer );
            bli_free_user( bBuffer );
        }
        else if ( pcka == 'p' || pcka == 'P' )
        {
            gtint_t bufSizeA = cblas_dgemm_pack_get_size( cblas_identifierA,
                                                         m,
                                                         n,
                                                         k );

            T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

            cblas_dgemm_pack( cblas_order, cblas_identifierA, cblas_transa,
                m, n, k, *alpha, ap, lda, aBuffer );


            cblas_dgemm_compute( cblas_order, cblas_packed, cblas_transb,
                m, n, k, aBuffer, lda, bp, ldb, *beta, cp, ldc );

            bli_free_user( aBuffer );
        }
        else if ( pckb == 'p' || pckb == 'P' )
        {
            gtint_t bufSizeB = cblas_dgemm_pack_get_size( cblas_identifierB,
                                                         m,
                                                         n,
                                                         k );

            T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

            cblas_dgemm_pack( cblas_order, cblas_identifierB, cblas_transb,
                m, n, k, *alpha, bp, ldb, bBuffer );

            cblas_dgemm_compute( cblas_order, cblas_transa, cblas_packed,
                m, n, k, ap, lda, bBuffer, ldb, *beta, cp, ldc );

            bli_free_user( bBuffer );
        }
        else
        {
            cblas_dgemm_compute( cblas_order, cblas_transa, cblas_transb,
                m, n, k, ap, lda, bp, ldb, *beta, cp, ldc );
        }
    }
    else
    {
        throw std::runtime_error("Error in testsuite/level3/gemm_compute.h: Invalid typename in cblas_gemm_compute().");
    }
}

template<typename T>
static void gemm_compute( char storage, char transa, char transb, char packa, char packb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
    T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    transb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
    packa = static_cast<char>(std::toupper(static_cast<unsigned char>(packa)));
    packb = static_cast<char>(std::toupper(static_cast<unsigned char>(packb)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char transa_cpy = transa;
    char transb_cpy = transb;
    char packa_cpy = packa;
    char packb_cpy = packb;
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
        gemm_compute_<T>( transa, transb, packa, packb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm_compute.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        gemm_compute_blis_impl<T>( transa, transb, packa, packb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm_compute.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_gemm_compute<T>( storage, transa, transb, packa, packb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    throw std::runtime_error("Error in testsuite/level3/gemm_compute.h: BLIS interfaces not yet implemented for pack and compute BLAS extensions.");
#else
    throw std::runtime_error("Error in testsuite/level3/gemm_compute.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "transb", transb, transb_cpy );
    computediff<char>( "packa", packa, packa_cpy );
    computediff<char>( "packb", packb, packb_cpy );
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
