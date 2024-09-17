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

class sgemmGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   float,
                                                   float,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

//matrix storage format, transA, transB, m, n, k, alpha, beta, lda, ldb, ldc

TEST_P( sgemmGeneric, API )
{
    using T = float;
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
        //thresh = (24*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}

INSTANTIATE_TEST_SUITE_P(
        expect_sgemv_path,
        sgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                  // storage format
            ::testing::Values('n','t'),                         // transa
            ::testing::Values('n','t'),                         // transb
            ::testing::Range(gtint_t(1), gtint_t(7), 1),        // m
            ::testing::Range(gtint_t(1), gtint_t(7), 1),        // n
            ::testing::Range(gtint_t(1), gtint_t(7), 1),        // k
            ::testing::Values(5.3, -1.0, 1.0),                  // alpha
            ::testing::Values(6.4, 1.0, -1.0, 0.0),             // beta
            ::testing::Values(0, 13),                           // increment to the leading dim of a
            ::testing::Values(0, 15),                           // increment to the leading dim of b
            ::testing::Values(0, 17)                            // increment to the leading dim of c
        ),
        ::gemmGenericPrint<float>()
    );

//----------------------------- sgemmGeneric_small kernel ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sgemmGeneric_small_path,
        sgemmGeneric,
        ::testing::Combine(
            // Test both storage types
            ::testing::Values('c'),                                        // storage format
            // Covers all possible combinations of storage schemes
            ::testing::Values('n', 't'),                                   // transa
            ::testing::Values('n', 't'),                                   // transb
            ::testing::Values(5, 19, 20, 24, 28, 32, 48, 44, 40, 36, 35),  // m
            ::testing::Range(gtint_t(25), gtint_t(43), gtint_t(1)),        // n
            // k-unroll factor = KR = 1
            ::testing::Range(gtint_t(2), gtint_t(25), 1),                  // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.0, 1.7),                        // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                        // beta
            ::testing::Values(0, 13),                                      // increment to the leading dim of a
            ::testing::Values(0, 15),                                      // increment to the leading dim of b
            ::testing::Values(0, 17)                                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<float>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sgemmGeneric_sup_path,
        sgemmGeneric,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                                         // storage format
            ::testing::Values('n', 't'),                                                    // transa
            ::testing::Values('n', 't'),                                                    // transb
            ::testing::Values(1002, 1025, 1054, 1083, 1112, 1111, 1327, 1333, 1338, 1378),  // m
            ::testing::Values(453, 462, 471, 504, 513, 522, 531, 540, 549, 558, 567 ),      // n
            ::testing::Range(gtint_t(250), gtint_t(261), 1),                                // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.0, 1.7),                                         // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                                         // beta
            ::testing::Values(0, 13),                                                       // increment to the leading dim of a
            ::testing::Values(0, 15),                                                       // increment to the leading dim of b
            ::testing::Values(0, 17)                                                        // increment to the leading dim of c
        ),
        ::gemmGenericPrint<float>()
    );

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sgemmGeneric_native_path,
        sgemmGeneric,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                            // storage format
            ::testing::Values('n', 't'),                       // transa
            ::testing::Values('n', 't'),                       // transb
            ::testing::Values(5017, 5025, 5061, 5327),         // m
            ::testing::Values(1709, 1731, 5005, 5417 ),        // n
            ::testing::Values(515, 527, 604),                  // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.0, 1.7),            // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),            // beta
            ::testing::Values(0, 13),                          // increment to the leading dim of a
            ::testing::Values(0, 15),                          // increment to the leading dim of b
            ::testing::Values(0, 17)                           // increment to the leading dim of c
        ),
        ::gemmGenericPrint<float>()
    );