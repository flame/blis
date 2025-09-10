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
#include "test_hemv.h"

class zhemvGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   dcomplex,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P( zhemvGeneric, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is u,l
    char uploa = std::get<1>(GetParam());
    // denotes whether matrix a is n,c
    char conja = std::get<2>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // specifies beta value
    T beta = std::get<6>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<7>(GetParam());
    // stride size for y
    gtint_t incy = std::get<8>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite hemv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // With adjustment applied for complex data.
     double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() && (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 2.4;
#endif
        thresh = adj*(3*n+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_hemv<T>( storage, uploa, conja, conjx, n, alpha, lda_inc, incx, beta, incy, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_hemv<T>( storage, uploa, conja, conjx, n, alpha, lda_inc, incx, beta, incy, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_hemv<T>( storage, uploa, conja, conjx, n, alpha, lda_inc, incx, beta, incy, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_hemv<T>( storage, uploa, conja, conjx, n, alpha, lda_inc, incx, beta, incy, thresh );
#endif
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        BlackboxSmall,
        zhemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n'),                                          // conja
            ::testing::Values('n'),                                          // conjx
            ::testing::Range(gtint_t(1),gtint_t(21),1),                      // n
            ::testing::Values(dcomplex{0.0, 0.0},dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0},dcomplex{1.0, -2.0}),      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0},dcomplex{1.0, -2.0}),      // beta
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for x
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(5))                        // increment to the leading dim of a
        ),
        ::hemvGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        BlackboxMedium,
        zhemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n'),                                          // conja
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(25),
                              gtint_t(33),
                              gtint_t(98),
                              gtint_t(173),
                              gtint_t(211)
                            ),                                               // n
            ::testing::Values(dcomplex{0.0, 0.0},dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0},dcomplex{1.0, -2.0}),      // alpha
            ::testing::Values(dcomplex{0.0, 0.0},dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0},dcomplex{1.0, -2.0}),      // beta
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for x
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(5))                        // increment to the leading dim of a
        ),
        ::hemvGenericPrint<dcomplex>()
    );
