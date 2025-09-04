/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

using T = dcomplex;

static float AOCL_NAN = std::numeric_limits<float>::quiet_NaN();
static float AOCL_INF = std::numeric_limits<float>::infinity();

class zgemmEVT :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   gtint_t,    // MatrixA row index
                                                   gtint_t,    // MatrixA col index
                                                   T,          // MatrixA Exception value
                                                   gtint_t,    // MatrixB row index
                                                   gtint_t,    // MatrixB col index
                                                   T,          // MatrixB Exception value
                                                   gtint_t,    // MatrixC row index
                                                   gtint_t,    // MatrixC col index
                                                   T,          // MatrixC Exception value
                                                   T,          //alpha
                                                   T,          //beta
                                                   gtint_t,    // inc to the lda
                                                   gtint_t,    // inc to the ldb
                                                   gtint_t     // inc to the ldc
                                                   >> {};

TEST_P( zgemmEVT, API )
{
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

    gtint_t ai = std::get<6>(GetParam());
    gtint_t aj = std::get<7>(GetParam());
    T aex = std::get<8>(GetParam());

    gtint_t bi = std::get<9>(GetParam());
    gtint_t bj = std::get<10>(GetParam());
    T bex = std::get<11>(GetParam());

    gtint_t ci = std::get<12>(GetParam());
    gtint_t cj = std::get<13>(GetParam());
    T cex = std::get<14>(GetParam());

    // specifies alpha value
    T alpha = std::get<15>(GetParam());
    // specifies beta value
    T beta = std::get<16>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<17>(GetParam());
    gtint_t ldb_inc = std::get<18>(GetParam());
    gtint_t ldc_inc = std::get<19>(GetParam());

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
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc,
                  alpha, beta, ai, aj, aex, bi, bj, bex, ci, cj, cex, thresh );
}

// Exception value testing(on matrices)

/*
    It contains both the exception value testing(EVT) and the
    positive accuracy testing of the bli_ZGEMM_4x4_avx2_k1_nn( ... ) computational
    kernel. This kernel is invoked from the BLAS layer, and inputs are given
    in a manner so as to avoid the other code-paths and test only the required
    kernel.

*/
/*
    For the bli_ZGEMM_4x4_avx2_k1_nn kernel, the main and fringe dimensions are as follows:
    For m : Main = { 4 }, fringe = { 2, 1 }
    For n : Main = { 4 }, fringe = { 2, 1 }

    Without any changes to the BLAS layer in BLIS, the fringe case of 1 cannot be touched
    separately, since if m/n is 1, the inputs are redirected to ZGEMV.

*/

// Testing for the main loop case for m and n
// The kernel uses 2 loads and 4 broadcasts. The exception values
// are induced at one index individually for each of the loads.
// They are also induced in the broadcast direction at two places.
INSTANTIATE_TEST_SUITE_P(
        K1_transA_N_transB_N_main,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(4)),                                  // m
            ::testing::Values(gtint_t(4)),                                  // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(gtint_t(3)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(2)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(1)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-2.2, 3.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{3.4, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.2, -2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{3.1, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

// Testing the fringe cases
// Fringe case minimum size is 2 along both m and n.
// Invloves only one load(AVX2 or (AVX2+SSE)). Thus,
// the exception values are induced at the first and second indices of the
// column vector A and row vector B.
INSTANTIATE_TEST_SUITE_P(
        K1_transA_N_transB_N_fringe,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(2), gtint_t(3)),                      // m
            ::testing::Values(gtint_t(2), gtint_t(3)),                      // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(gtint_t(0)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(1)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(1)),                                  // ci
            ::testing::Values(gtint_t(0)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-2.2, 3.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{2.3, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.2, -2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{5.6, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

// Exception value testing(on alpha and beta)
// Alpha and beta are set to exception values
INSTANTIATE_TEST_SUITE_P(
        K1_transA_N_transB_N_alphabeta,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(2), gtint_t(4)),                      // m
            ::testing::Values(gtint_t(2), gtint_t(4)),                      // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(gtint_t(0)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0)),                                  // bj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(0)),                                  // cj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // alpha
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

/********************************************************/
/* Testing for tiny code paths                          */
/* m,n,k is choosen such that tiny code path is called  */
/* Matrix A, B, C are filled with Infs and Nans         */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        Disabled_Tiny_Matrix,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(30)),                                 // m
            ::testing::Values(gtint_t(20)),                                 // n
            ::testing::Values(gtint_t(10)),                                 // k
            ::testing::Values(gtint_t(11)),                                 // ai
            ::testing::Values(gtint_t(1)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(8)),                                  // bi
            ::testing::Values(gtint_t(5)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/ //Failures
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(1)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-2.2, 3.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{6.0, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.2, -2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{5.6, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

/********************************************************/
/* Testing for small code paths                         */
/* m,n,k is choosen such that small code path is called */
/* Matrix A, B, C are filled with Infs and Nans         */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        DISABLED_Small_Matrix,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(201)),                                // m
            ::testing::Values(gtint_t(4)),                                  // n
            ::testing::Values(gtint_t(10)),                                 // k
            ::testing::Values(gtint_t(3)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(2)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/ //Failures
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(1)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-2.2, 3.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{6.0, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.2, -2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{5.6, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

/******************************************************/
/* Testing for SUP code paths                         */
/* m,n,k is choosen such that SUP code path is called */
/* Matrix A, B, C are filled with Infs and Nans         */
/******************************************************/
INSTANTIATE_TEST_SUITE_P(
        DISABLED_Skinny_Matrix,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(gtint_t(90)),                                 // m
            ::testing::Values(gtint_t(80)),                                 // n
            ::testing::Values(gtint_t(1080)),                               // k
            ::testing::Values(gtint_t(3)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/ //Failure
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(2)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(1)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{3.6, -1.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{34.0, 0.0}, T{0.0, 1.0}),                   // alpha
            ::testing::Values(T{-5.7, 1.2}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{3.0, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

/*********************************************************/
/* Testing for Native code paths                         */
/* m,n,k is choosen such that Native code path is called */
/* Matrix A, B, C are filled with Infs and Nans         */
/*********************************************************/
INSTANTIATE_TEST_SUITE_P(
        DISABLED_Large_Matrix,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(200)),                                // m
            ::testing::Values(gtint_t(200)),                                // n
            ::testing::Values(gtint_t(130)),                                // k
            ::testing::Values(gtint_t(1)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/   //Failures
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, 2.3}, /*T{AOCL_INF, 0.0},*/
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(3)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 0.0},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-2.2, 3.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{4.1, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.2, -2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{4.3, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );

/********************************************************/
/* Testing for all code paths                           */
/* m,n,k is choosen such that all code path are covered */
/* Matrix A, B, C are filled valid integers or floats   */
/* Matrix A, B, C are filled with Infs and Nans         */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        alpha_beta,
        zgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(14), gtint_t(200)),                   // m
            ::testing::Values(gtint_t(10), gtint_t(300)),                   // n
            ::testing::Values(gtint_t(20), gtint_t(1005)),                  // k
            ::testing::Values(gtint_t(0)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0)),                                  // bj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(0)),                                  // cj
            ::testing::Values(T{0.0, 0.0}),
            ::testing::Values(T{AOCL_NAN, 2.3}, /* T{AOCL_INF, 0.0}, */
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // alpha
            ::testing::Values(T{AOCL_NAN, 2.3}, /* T{AOCL_INF, 0.0}, */
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<dcomplex>()
    );
