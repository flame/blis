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
#include "test_gemm_compute.h"

class dgemmComputeGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   double,
                                                   double,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P( dgemmComputeGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t
    char transa = std::get<1>(GetParam());
    // denotes whether matrix b is n,c,t
    char transb = std::get<2>(GetParam());
    // denotes whether matrix a is packed (p) or unpacked (u)
    char packa = std::get<3>(GetParam());
    // denotes whether matrix b is packed (p) or unpacked (u)
    char packb = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // matrix size k
    gtint_t k  = std::get<7>(GetParam());
    // specifies alpha value
    T alpha = std::get<8>(GetParam());
    // specifies beta value
    T beta = std::get<9>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<10>(GetParam());
    gtint_t ldb_inc = std::get<11>(GetParam());
    gtint_t ldc_inc = std::get<12>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();
        //thresh = (7*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm_compute<T>( storage, transa, transb, packa, packb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        dgemmComputeGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                   // storage format
            ::testing::Values('n', 't', 'c'),                    // transa
            ::testing::Values('n', 't', 'c'),                    // transb
            ::testing::Values('u', 'p'),                         // packa
            ::testing::Values('u', 'p'),                         // packb
            ::testing::Range(gtint_t(10), gtint_t(31), 10),      // m
            ::testing::Range(gtint_t(10), gtint_t(31), 10),      // n
            ::testing::Range(gtint_t(10), gtint_t(31), 10),      // k
            ::testing::Values(0.0, 1.0, -1.2, 2.1),              // alpha
            ::testing::Values(0.0, 1.0, -1.2, 2.1),              // beta
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                        // increment to the leading dim of c
        ),
        ::gemm_computeGeneticPrint<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        TinySizes,
        dgemmComputeGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                   // storage format
            ::testing::Values('n', 't', 'c'),                    // transa
            ::testing::Values('n', 't', 'c'),                    // transb
            ::testing::Values('u', 'p'),                         // packa
            ::testing::Values('u', 'p'),                         // packb
            ::testing::Range(gtint_t(1), gtint_t(3), 1),         // m
            ::testing::Range(gtint_t(1), gtint_t(3), 1),         // n
            ::testing::Range(gtint_t(1), gtint_t(3), 1),         // k
            ::testing::Values(0.0, 1.0, -1.2, 2.1),              // alpha
            ::testing::Values(0.0, 1.0, -1.2, 2.1),              // beta
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                        // increment to the leading dim of c
        ),
        ::gemm_computeGeneticPrint<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        DimensionsGtBlocksizes,                                  // Dimensions > SUP Blocksizes
        dgemmComputeGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                   // storage format
            ::testing::Values('n'),                              // transa
            ::testing::Values('n'),                              // transb
            ::testing::Values('u', 'p'),                         // packa
            ::testing::Values('u', 'p'),                         // packb
            ::testing::Values(71, 73),                           // m (MC - 1, MC + 1)
            ::testing::Values(4079, 4081),                       // n (NC - 1, NC + 1)
            ::testing::Values(255, 257),                         // k (KC - 1, KC + 1)
            ::testing::Values(1.0),                              // alpha
            ::testing::Values(1.0),                              // beta
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                       // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                        // increment to the leading dim of c
        ),
        ::gemm_computeGeneticPrint<double>()
    );
