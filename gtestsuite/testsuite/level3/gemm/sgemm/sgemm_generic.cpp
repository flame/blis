/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class sgemmGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   float,      // alpha
                                                   float,      // beta
                                                   gtint_t     // inc to the lda, ldb and ldc
                                                   >> {};

class sgemmGeneric3 :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   float,      // alpha
                                                   float,      // beta
                                                   gtint_t,    // inc to the lda
                                                   gtint_t,    // inc to the ldb
                                                   gtint_t     // inc to the ldc
                                                   >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmGeneric3);

using T = float;
void sgemmGeneric( char storage, char transa, char transb, gtint_t m, gtint_t n, gtint_t k,
                   T alpha, T beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc )
{
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
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 3.9;
#else
        double adj = 3.8;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
#endif
}

TEST_P( sgemmGeneric1, API )
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

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    sgemmGeneric( storage, transa, transb, m, n, k,
                  alpha, beta, lda_inc, ldb_inc, ldc_inc );
}

TEST_P( sgemmGeneric3, API )
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

    sgemmGeneric( storage, transa, transb, m, n, k,
                  alpha, beta, lda_inc, ldb_inc, ldc_inc );
}

// ----------------------------- alpha = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_alpha0_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0, 1, 2, 79),                              // k
            ::testing::Values(0.0),                                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- k = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k0_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0),                                        // k
            ::testing::Values(     1.0, -1.0, 2.3),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

//----------------------------- m = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_m1_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1),                                        // m
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

//----------------------------- n = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_n1_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 79),                                 // m
            ::testing::Values(1),                                        // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

//----------------------------- k = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k1_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1),                                        // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

//----------------------------- bli_sgemm_tiny kernel ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_tiny_path,
        sgemmGeneric3,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(3, 81, 138),                               // m
            ::testing::Values(2, 35, 100),                               // n
            ::testing::Values(5, 12, 24),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 7),                                     // increment to the leading dim of a
            ::testing::Values(0, 4),                                     // increment to the leading dim of b
            ::testing::Values(0, 11)                                     // increment to the leading dim of c
        ),
        ::gemmGeneric3Print<float>()
    );

//----------------------------- sgemm_small kernel -----------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_small_path,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            // Covers all possible combinations of storage schemes
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(5, 20, 32, 44),                            // m
            ::testing::Values(25, 37, 42),                               // n
            ::testing::Values(2, 13, 24),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sup_path,
        sgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(603, 700),                                 // m
            ::testing::Values(453, 567),                                 // n
            ::testing::Values(155, 250),                                 // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_native_path,
        sgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(318),                                      // m
            ::testing::Values(643),                                      // n
            ::testing::Values(511),                                      // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.0, 1.7),                      // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

INSTANTIATE_TEST_SUITE_P(
        expect_native_path_Large,
        sgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(318),                                      // m
            ::testing::Values(7417),                                     // n
            ::testing::Values(511),                                      // k
            // No condition based on alpha
            ::testing::Values(0.0, -1.0, 1.0, 1.7),                      // alpha
            // No condition based on beta
            ::testing::Values(0.0, -1.0, 1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- Extreme M value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_M,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(9689, 33444),                              // m
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(1, 34),                                    // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- Extreme N value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_N,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(1, 6),                                     // m
            ::testing::Values(9689, 33444),                              // n
            ::testing::Values(1, 34),                                    // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );

// ----------------------------- Extreme K value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_K,
        sgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 't'),                                 // transa
            ::testing::Values('n', 't'),                                 // transb
            ::testing::Values(1, 34),                                    // m
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(9689, 33444),                              // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<float>()
    );
