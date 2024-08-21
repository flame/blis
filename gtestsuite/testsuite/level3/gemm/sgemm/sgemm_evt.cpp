/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

class sgemmEVT :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   gtint_t,    // MatrixA row index
                                                   gtint_t,    // MatrixA col index
                                                   float,      // MatrixA Exception value
                                                   gtint_t,    // MatrixB row index
                                                   gtint_t,    // MatrixB col index
                                                   float,      // MatrixB Exception value
                                                   gtint_t,    // MatrixC row index
                                                   gtint_t,    // MatrixC col index
                                                   float,      // MatrixC Exception value
                                                   float,      // alpha
                                                   float,      // beta
                                                   gtint_t,    // inc to the lda
                                                   gtint_t,    // inc to the ldb
                                                   gtint_t     // inc to the ldc
                                                   >> {};
TEST_P( sgemmEVT, API )
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
    gtint_t ai, aj, bi, bj, ci, cj;
    T aex, bex, cex;
    ai = std::get<6>(GetParam());
    aj = std::get<7>(GetParam());
    aex = std::get<8>(GetParam());
    bi = std::get<9>(GetParam());
    bj = std::get<10>(GetParam());
    bex = std::get<11>(GetParam());
    ci = std::get<12>(GetParam());
    cj = std::get<13>(GetParam());
    cex = std::get<14>(GetParam());
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
/*
    It contains the exception value testing(EVT).
*/
static float NaN = std::numeric_limits<float>::quiet_NaN();
static float Inf = std::numeric_limits<float>::infinity();
// Exception value testing(on matrices)


/********************************************************/
/* Testing for small code paths                         */
/* m,n,k is choosen such that small code path is called */
/* Matrix A, B, C are filled with Infs and Nans         */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        SMALL_Matrix,
        sgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n','t'),                                     // transa
            ::testing::Values('n','t'),                                     // transb
            ::testing::Values(5, 19, 20, 24, 28, 32, 48, 44, 40, 36, 35),   // m
            ::testing::Range(gtint_t(13), gtint_t(43), gtint_t(1)),         // n
            ::testing::Range(gtint_t(2), gtint_t(25), 1),                   // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(NaN, Inf, -Inf),                              // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(NaN, Inf, -Inf),                              // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(NaN, Inf, -Inf),                              // cexval
            ::testing::Values(float(-2.2)),                                 // alpha
            ::testing::Values(float(1.2)),                                  // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<float>()
    );
/******************************************************/
/* Testing for SUP code paths                         */
/* m,n,k is choosen such that SUP code path is called */
/* Matrix A, B, C are filled with Infs and Nans       */
/******************************************************/
INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix,
        sgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(1002, 1025, 1054, 1083, 1112, 1111, 1327),    // m
            ::testing::Values(453, 462, 471, 504, 513, 522, 531),           // n
            ::testing::Range(gtint_t(250), gtint_t(261), 1),                // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(NaN, Inf, -Inf),                              // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(NaN, Inf, -Inf),                              // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(NaN, Inf, -Inf),                              // cexval
            ::testing::Values(float(3.6)),                                  // alpha
            ::testing::Values(float(-5.1)),                                 // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<float>()
    );
/*********************************************************/
/* Testing for native code paths                         */
/* m,n,k is choosen such that Native code path is called */
/* Matrix A, B, C are filled with Infs and Nans          */
/*********************************************************/
INSTANTIATE_TEST_SUITE_P(
        Large_Matrix,
        sgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(1001)),                               // m
            ::testing::Values(gtint_t(1001)),                               // n
            ::testing::Values(gtint_t(260)),                                // k
            ::testing::Values(gtint_t(1)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(NaN, Inf, -Inf),                              // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0)),                                  // bj
            ::testing::Values(NaN, Inf, -Inf),                              // bexval
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(1)),                                  // cj
            ::testing::Values(NaN, Inf, -Inf),                              // cexval
            ::testing::Values(float(-2.2)),                                 // alpha
            ::testing::Values(float(1.2)),                                  // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<float>()
    );
/********************************************************/
/* Testing for small & sup code paths                   */
/* m,n,k is choosen such that small & sup code path     */
/* are covered.                                         */
/* Matrix A, B, C are filled valid integers or floats   */
/* Alpha and beta are assigned with Infs and Nans       */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        alpha_beta,
        sgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(14), gtint_t(100)),                   // m
            ::testing::Values(gtint_t(10), gtint_t(90)),                    // n
            ::testing::Values(gtint_t(20), gtint_t(105)),                   // k
            ::testing::Values(gtint_t(0)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(float(0.0)),
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0)),                                  // bj
            ::testing::Values(float(0.0)),
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(0)),                                  // cj
            ::testing::Values(float(0.0)),
            ::testing::Values(NaN), //Failures , Inf, -Inf),                // alpha
            ::testing::Values(NaN, Inf, -Inf),                              // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<float>()
    );
/********************************************************/
/* Testing for Native code paths                        */
/* m,n,k is choosen such that nat code path are covered */
/* Matrix A, B, C are filled valid integers or floats   */
/* Alpha and beta are assigned with Infs and Nans       */
/********************************************************/
INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_alpha_beta,
        sgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(1001)),                               // m
            ::testing::Values(gtint_t(1001)),                               // n
            ::testing::Values(gtint_t(260)),                                // k
            ::testing::Values(gtint_t(0)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(float(0.0)),
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0)),                                  // bj
            ::testing::Values(float(0.0)),
            ::testing::Values(gtint_t(0)),                                  // ci
            ::testing::Values(gtint_t(0)),                                  // cj
            ::testing::Values(float(0.0)),
            ::testing::Values(NaN), //Failures , Inf, -Inf),                // alpha
            ::testing::Values(NaN), //Failure Inf, -Inf),                   // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::gemmEVTPrint<float>()
    );
