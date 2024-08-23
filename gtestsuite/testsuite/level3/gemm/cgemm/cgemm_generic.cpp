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

#include <gtest/gtest.h>
#include "level3/gemm/test_gemm.h"

class cgemmGeneric :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   scomplex,   // alpha
                                                   scomplex,   // beta
                                                   gtint_t,    // inc to the lda
                                                   gtint_t,    // inc to the ldb
                                                   gtint_t     // inc to the ldc
                                                   >> {};
TEST_P( cgemmGeneric, API )
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // matrix size k
    gtint_t k  = std::get<5>(GetParam());
    // specifies alpha value
    T alpha = std::get<6>(GetParam());
    // specifies beta value
    T beta = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());
    gtint_t ldc_inc = std::get<10>(GetParam());
    // Set the threshold for the errors:

    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}

/********************************************************************/
/* Testing SUP and Native implementation of cgemm API               */
/********************************************************************/
/************************** SCALM************************************/
/* Scaling of C matrix for below conditions                         */
/* 1. When alpha is zero                                            */
/* 2. When Matrix A or Matrix B has zero dimension                  */
/* Scale Matrix C by Beta and return                                */
/********************************************************************/
/************************** SUP *************************************/
/* Current SUP implmentation does not support below parameters      */
/* 1. General Stride                                                */
/* 2. Conjugate                                                     */
/* 3. Input dimensions greater than below thresholds                */
/*    m > 380 ||  n > 256 || k > 220                                */
/* SUP implementations is suitable for Skinny Matrices              */
/* List of API's:                                                   */
/*  1. bli_cgemmsup_rv_zen_asm_3x8m: M preferred kernel             */
/*  2. bli_cgemmsup_rv_zen_asm_3x8n: N preferred kernel             */
/********************************************************************/
/************************** NATIVE***********************************/
/*  When SUP method does not support given input arguments,         */
/*  Native implmentation will be invoked, it is well suited for     */
/*  square, large sizes                                             */
/* API Name: bli_cgemm_haswell_asm_3x8                              */
/********************************************************************/

INSTANTIATE_TEST_SUITE_P(
        Alpha_zero,
        cgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(300), gtint_t(17)),                   // m
            ::testing::Values(gtint_t(200), gtint_t(18)),                   // n
            ::testing::Values(gtint_t(150), gtint_t(19)),                   // k
            ::testing::Values(scomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{12.9, 12.3}, scomplex{0.0, 1.9},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.2, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(5)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(3))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix,
        cgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(300), gtint_t(320)),                  // m
            ::testing::Values(gtint_t(200), gtint_t(220)),                  // n
            ::testing::Values(gtint_t(150), gtint_t(160)),                  // k
            ::testing::Values(scomplex{-1.0, -2.0}),                        // alpha
            ::testing::Values(scomplex{12.0, 2.3}),                         // beta
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(3))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Alpha_Beta,
        cgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(300), gtint_t(304)),                  // m
            ::testing::Values(gtint_t(200), gtint_t(209)),                  // n
            ::testing::Values(gtint_t(150)),                                // k
            ::testing::Values(scomplex{0.0, -30.0},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{0.0, 1.3},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.0, 0.0}, scomplex{0.0, 0.0}),      // beta
            ::testing::Values(gtint_t(0), gtint_t(5)),                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(2)),                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(6))                    // increment to the leading dim of c
        ),
        ::gemmGenericPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix,
        cgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(400), gtint_t(700)),                  // m
            ::testing::Values(gtint_t(380), gtint_t(1000)),                 // n
            ::testing::Values(gtint_t(270), gtint_t(280)),                  // k
            ::testing::Values(scomplex{1.5, 3.5}),                          // alpha
            ::testing::Values(scomplex{2.0, 4.1}),                          // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmGenericPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Alpha_Beta,
        cgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(400), gtint_t(700)),                  // m
            ::testing::Values(gtint_t(380), gtint_t(1000)),                 // n
            ::testing::Values(gtint_t(270)),                                // k
            ::testing::Values(scomplex{0.0, -10.0},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{2.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{0.0, 3.4},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{3.3, 0.0}, scomplex{0.0, 0.0}),      // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmGenericPrint<scomplex>()
    );
