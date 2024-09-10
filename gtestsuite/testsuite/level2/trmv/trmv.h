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
 * @brief Performs the operation
  *    x := alpha * transa(A) * x
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
static void trmv_( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        strmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in trmv_().");
}

template<typename T>
static void trmv_blis_impl( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        strmv_blis_impl( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtrmv_blis_impl( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrmv_blis_impl( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrmv_blis_impl( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in trmv_blis_impl().");
}

template<typename T>
static void cblas_trmv( char storage, char uploa, char transa, char diaga,
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
        cblas_strmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dtrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_ctrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_ztrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in cblas_trmv().");
}

template<typename T>
static void typed_trmv( char storage, char uplo, char trans, char diag,
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
        bli_strmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dtrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_ctrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_ztrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else

        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in typed_trmv().");
}

template<typename T>
static void trmv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
#if (defined TEST_BLAS_LIKE || defined TEST_CBLAS)
    T one;
    testinghelpers::initone(one);
#endif

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    diaga = static_cast<char>(std::toupper(static_cast<unsigned char>(diaga)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char uploa_cpy = uploa;
    char transa_cpy = transa;
    char diaga_cpy = diaga;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t incx_cpy = incx;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( storage, transa, n, n, lda );
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if(( storage == 'c' || storage == 'C' ))
        if( *alpha == one )
            trmv_<T>( uploa, transa, diaga, n, ap, lda, xp, incx );
        else
            throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS interface cannot be tested for alpha != one.");
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if(( storage == 'c' || storage == 'C' ))
        if( *alpha == one )
            trmv_blis_impl<T>( uploa, transa, diaga, n, ap, lda, xp, incx );
        else
            throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS_BLIS_IMPL interface cannot be tested for alpha != one.");
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    if( *alpha == one )
        cblas_trmv<T>( storage, uploa, transa, diaga, n, ap, lda, xp, incx );
    else
      throw std::runtime_error("Error in testsuite/level2/trmv.h: CBLAS interface cannot be tested for alpha != one.");
#elif TEST_BLIS_TYPED
    typed_trmv<T>( storage, uploa, transa, diaga, n, alpha, ap, lda, xp, incx );
#else
    throw std::runtime_error("Error in testsuite/level2/trmv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "uploa", uploa, uploa_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "diaga", diaga, diaga_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        computediff<T>( "A", storage, n, n, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }
#endif
}
