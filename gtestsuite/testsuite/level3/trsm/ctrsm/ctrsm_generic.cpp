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
#include "level3/trsm/test_trsm.h"

class ctrsmGeneric :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // side
                                                   char,          // uplo
                                                   char,          // transa
                                                   char,          // diaga
                                                   gtint_t,       // m
                                                   gtint_t,       // n
                                                   scomplex,      // alpha
                                                   gtint_t,       // lda_inc
                                                   gtint_t>> {};  // ldb_inc

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(strsmGeneric);

TEST_P( ctrsmGeneric, API )
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
    // denotes whether matrix a in unit or non-unit diagonal
    char diaga = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // specifies alpha value
    T alpha = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trsm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        if ( side == 'l' || side == 'L' )
            thresh = 3*m*testinghelpers::getEpsilon<T>();
        else
            thresh = 3*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
#endif
}

/**
 * @brief Test CTRSM native path, which starts from size 1001 for BLAS api
 *        and starts from size 0 for BLIS api.
 */
INSTANTIATE_TEST_SUITE_P(
        Native,
        ctrsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','c','t'),                                  // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 112, 1200),                                 // m
            ::testing::Values(1, 154, 1317),                                 // n
            ::testing::Values(scomplex{2.0,-1.0}),                           // alpha
            ::testing::Values(gtint_t(31)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(45))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<scomplex>()
    );

/**
 * @brief Test CTRSM small avx2 path all fringe cases
 *        Kernel size for avx2 small path is 8x3, testing in range of
 *        1 to 8 ensures all finge cases are being tested.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2_fringe,
        ctrsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(13), 3),                    // m
            ::testing::Range(gtint_t(1), gtint_t(9), 2),                     // n
            ::testing::Values(scomplex{2.0,-3.4}),                           // alpha
            ::testing::Values(gtint_t(58)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(32))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<scomplex>()
    );

/**
 * @brief Test CTRSM small avx2 path, this code path is used in range 0 to 1000
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2,
        ctrsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 1000),                                     // m
            ::testing::Values(48, 1000),                                     // n
            ::testing::Values(scomplex{2.0,-3.4}),                           // alpha
            ::testing::Values(gtint_t(85)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(33))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<scomplex>()
    );

/**
 * @brief Test CTRSM with differnt values of alpha
 *      code paths covered:
 *          TRSV              -> 1
 *          TRSM_AVX2_small   -> 3
 *          TRSM_NATIVE       -> 1001
 */
INSTANTIATE_TEST_SUITE_P(
        Alpha,
        ctrsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 3, 1001),                                   // n
            ::testing::Values(1, 3, 1001),                                   // m
            ::testing::Values(scomplex{2.0, 0.0}, scomplex{0.0, -10.0},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}),      // alpha
            ::testing::Values(gtint_t(0), gtint_t(45)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(93))                       // increment to the leading dim of b
        ),
        ::trsmGenericPrint<scomplex>()
    );
