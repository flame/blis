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
#include "level3/gemmt/test_gemmt.h"

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class zgemmtGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   dcomplex,     // alpha
                                                   dcomplex,     // beta
                                                   gtint_t,      // inc to the lda, ldb and ldc
                                                   bool>> {};    // is memory test

class zgemmtGeneric3 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   dcomplex,     // alpha
                                                   dcomplex,     // beta
                                                   gtint_t,      // lda_inc
                                                   gtint_t,      // ldb_inc
                                                   gtint_t,      // ldc_inc
                                                   bool>> {};    // is memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zgemmtGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zgemmtGeneric3);

using T = dcomplex;
void zgemmtGeneric( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
                    T alpha, T beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
                    bool is_mem_test )
{
    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

TEST_P( zgemmtGeneric1, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies if the upper or lower triangular part of C is used
    char uplo = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<3>(GetParam());
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
    bool is_mem_test = std::get<9>(GetParam());

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    zgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

TEST_P( zgemmtGeneric3, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies if the upper or lower triangular part of C is used
    char uplo = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<3>(GetParam());
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
    bool is_mem_test = std::get<11>(GetParam());

    zgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

// Disable tests for BLIS_TYPED case due to compiler errors.
#ifndef TEST_BLIS_TYPED

// ----------------------------- alpha = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_alpha0_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0, 1, 2, 79),                              // k
            ::testing::Values(dcomplex{0.0, 0.0}),                       // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

// ----------------------------- k = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k0_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0),                                        // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

//----------------------------- n = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_n1_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1),                                        // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

//----------------------------- k = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k1_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1),                                        // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

//-------------------- No GEMMT tiny code path yet but use this name ---------------------
INSTANTIATE_TEST_SUITE_P(
        expect_tiny_path,
        zgemmtGeneric3,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(2, 43),                                    // n
            ::testing::Values(2, 19),                                    // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 7),                                     // increment to the leading dim of a
            ::testing::Values(0, 4),                                     // increment to the leading dim of b
            ::testing::Values(0, 11),                                    // increment to the leading dim of c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric3Print<dcomplex>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sup_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(3, 32, 89),                                // n
            ::testing::Values(5, 17, 244),                               // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_native_path,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(263, 577),                                 // n
            ::testing::Values(3, 47),                                    // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        expect_native_path_Large,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(807, 2701),                                // n
            ::testing::Values(905),                                      // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

// ----------------------------- Extreme N value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_N_Large,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(9689),                                     // n
            ::testing::Values(1, 34),                                    // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

// ----------------------------- Extreme K value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_K,
        zgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(9689, 33444),                              // k
            ::testing::Values(                     dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},  dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.7},
                              dcomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<dcomplex>()
    );

#endif // ifndef TEST_BLIS_TYPED
