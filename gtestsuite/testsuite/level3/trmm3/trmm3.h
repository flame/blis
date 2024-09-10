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
 *        C := beta * C_orig + alpha * transa(A) * transb(B)
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the symmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     uplo   specifies if the upper or lower triangular part of A is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     diaga  specifies whether upper or lower triangular part of the matrix A
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies the number of rows and cols of the  matrix
                         op( A ) and rows of the matrix C and B
 * @param[in]     n      specifies the number of columns of the matrix
                         op( B ) and the number of columns of the matrix C
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
static void typed_trmm3( char storage, char side, char uplo, char trnsa,
    char diag, char trnsb, gtint_t m, gtint_t n, T *alpha, T *a, gtint_t lda,
    T *b, gtint_t ldb, T *beta, T *c, gtint_t ldc )
{
    side_t  sidea;
    uplo_t  uploa;
    trans_t transa;
    trans_t transb;
    diag_t  diaga;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_side( side, &sidea );
    testinghelpers::char_to_blis_uplo( uplo, &uploa );
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );
    testinghelpers::char_to_blis_diag( diag, &diaga );

    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    rsa=rsb=rsc=1;
    csa=csb=csc=1;
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

    if constexpr (std::is_same<T, float>::value) {
        bli_strmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, double>::value) {
        bli_dtrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, scomplex>::value)  {
        bli_ctrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, dcomplex>::value)  {
        bli_ztrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else
        throw std::runtime_error("Error in testsuite/level3/trmm3.h: Invalid typename in typed_trmm3().");
}

template<typename T>
static void trmm3( char storage, char side, char uploa, char transa, char diaga,
                  char transb, gtint_t m, gtint_t n, T *alpha, T *ap, gtint_t lda,
                  T *bp, gtint_t ldb, T *beta, T *c, gtint_t ldc )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    side = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    diaga = static_cast<char>(std::toupper(static_cast<unsigned char>(diaga)));
    transb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char side_cpy = side;
    char uploa_cpy = uploa;
    char transa_cpy = transa;
    char diaga_cpy = diaga;
    char transb_cpy = transb;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t ldb_cpy = ldb;
    T* beta_cpy = beta;
    gtint_t ldc_cpy = ldc;

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
    T* bp_cpy = nullptr;
    gtint_t size_bp = testinghelpers::matsize( storage, transb, m, n, ldb );
    if (bp && size_bp > 0)
    {
        bp_cpy = new T[size_bp];
        memcpy( bp_cpy, bp, size_bp * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: BLAS interface is not available.");
#elif TEST_BLAS_BLIS_IMPL
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: BLAS_BLIS_IMPL interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha,
                                            ap, lda, bp, ldb, beta, c, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: No interfaces are set to be tested.");
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
    computediff<char>( "transb", transb, transb_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
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
        computediff<T>( "A", storage, mn, mn, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }

    if (bp && size_bp > 0)
    {
        if(( transb == 'n' ) || ( transb == 'N' ))
            computediff<T>( "B", storage, m, n, bp, bp_cpy, ldb, true );
        else
            computediff<T>( "B", storage, n, m, bp, bp_cpy, ldb, true );
        delete[] bp_cpy;
    }
#endif
}
