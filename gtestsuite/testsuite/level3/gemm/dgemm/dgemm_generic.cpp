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

class dgemmGeneric :
        public ::testing::TestWithParam<std::tuple<char,
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


//matrix storage format, transA, transB, m, n, k, alpha, beta, lda, ldb, ldc
TEST_P( dgemmGeneric, API )
{
    using T = double;
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
        //thresh = (15*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}

INSTANTIATE_TEST_SUITE_P(
        expect_dgemm_k1_path,
        dgemmGeneric,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                      // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n'),                                      // transa
            ::testing::Values('n'),                                      // transb
            ::testing::Values(3, 17, 103, 178),                          // m
            ::testing::Values(2, 26, 79),                                // n
            ::testing::Values(1),                                        // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.7),                           // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a
            ::testing::Values(0, 3),                                     // increment to the leading dim of b
            ::testing::Values(0, 3)                                      // increment to the leading dim of c
        ),
        ::gemmGenericPrint<double>()
    );

//----------------------------- bli_dgemm_tiny kernel ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_dgemm_tiny_path,
        dgemmGeneric,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                      // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(3, 81, 138),                               // m
            ::testing::Values(2, 35, 100),                               // n
            ::testing::Values(5, 12, 24),                                // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.7),                           // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a
            ::testing::Values(0, 3),                                     // increment to the leading dim of b
            ::testing::Values(0, 3)                                      // increment to the leading dim of c
        ),
        ::gemmGenericPrint<double>()
    );

//----------------------------- dgemm_small kernel ------------------------------------


// Tests both bli_dgemm_small and bli_dgemm_small_At
INSTANTIATE_TEST_SUITE_P(
        expect_dgemm_small_path,
        dgemmGeneric,
        ::testing::Combine(
            // Test both storage types
            ::testing::Values('c'),                                        // storage format
            // Covers all possible combinations of storage schemes
            ::testing::Values('n', 't'),                                   // transa
            ::testing::Values('n', 't'),                                   // transb
            ::testing::Values(5, 19, 32, 44),                              // m
            ::testing::Values(25, 27, 32),                                 // n
            // k-unroll factor = KR = 1
            ::testing::Values(5, 17, 24),                                   // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.7),                             // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                        // beta
            ::testing::Values(0, 3),                                       // increment to the leading dim of a
            ::testing::Values(0, 3),                                       // increment to the leading dim of b
            ::testing::Values(0, 3)                                        // increment to the leading dim of c
        ),
        ::gemmGenericPrint<double>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_dgemm_sup_path,
        dgemmGeneric,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                            // storage format
            ::testing::Values('n', 't'),                                       // transa
            ::testing::Values('n', 't'),                                       // transb
            ::testing::Values(1002, 1377),                                     // m
            ::testing::Values(453, 567),                                       // n
            ::testing::Values(105, 124),                                       // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.7),                                 // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                            // beta
            ::testing::Values(0, 3),                                           // increment to the leading dim of a
            ::testing::Values(0, 3),                                           // increment to the leading dim of b
            ::testing::Values(0, 3)                                            // increment to the leading dim of c
        ),
        ::gemmGenericPrint<double>()
    );

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_dgemm_native_path,
        dgemmGeneric,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                            // storage format
            // Covers vectorized section of 8xk and 6xk pack kernels for both storage formats
            ::testing::Values('n', 't'),                       // transa
            ::testing::Values('n', 't'),                       // transb
            ::testing::Values(5017, 5061),                     // m
            ::testing::Values(709, 5417),                      // n
            ::testing::Values(515, 604),                       // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.7),                 // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),            // beta
            ::testing::Values(0, 3),                           // increment to the leading dim of a
            ::testing::Values(0, 3),                           // increment to the leading dim of b
            ::testing::Values(0, 3)                            // increment to the leading dim of c
        ),
        ::gemmGenericPrint<double>()
    );
