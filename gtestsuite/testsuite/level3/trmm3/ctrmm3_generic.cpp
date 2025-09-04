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
#include "test_trmm3.h"

class ctrmm3Generic :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   scomplex,
                                                   scomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ctrmm3Generic);

TEST_P( ctrmm3Generic, API )
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies matrix A appears left or right in
    // the matrix multiplication
    char side = std::get<1>(GetParam());
    // specifies upper or lower triangular part of A is used
    char uploa = std::get<2>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<3>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<4>(GetParam());
    // denotes whether matrix a in unit or non-unit diagonal
    char diaga = std::get<5>(GetParam());
    // matrix size m
    gtint_t m  = std::get<6>(GetParam());
    // matrix size n
    gtint_t n  = std::get<7>(GetParam());
    // specifies alpha value
    T alpha = std::get<8>(GetParam());
    // specifies alpha value
    T beta = std::get<9>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<10>(GetParam());
    gtint_t ldb_inc = std::get<11>(GetParam());
    gtint_t ldc_inc = std::get<12>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trmm3.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() &&
            (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        if ( side == 'l' || side == 'L' )
            thresh = (3*m+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = (3*n+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha, lda_inc, ldb_inc, beta, ldc_inc, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha, lda_inc, ldb_inc, beta, ldc_inc, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha, lda_inc, ldb_inc, beta, ldc_inc, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha, lda_inc, ldb_inc, beta, ldc_inc, thresh );
#endif
}

#ifdef TEST_BLIS_TYPED
// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        ctrmm3Generic,
        ::testing::Combine(
            ::testing::Values('c','r'),                                      // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t','c'),                                  // transa
            ::testing::Values('n'),                                          // transb /*transb works only for 'n' case*/
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // m
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // n
            ::testing::Values(scomplex{2.0,-1.0}),                           // alpha
            ::testing::Values(scomplex{-1.0,1.0}),                           // beta
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                    // increment to the leading dim of c
        ),
        ::trmm3GenericPrint<scomplex>()
    );
#endif
