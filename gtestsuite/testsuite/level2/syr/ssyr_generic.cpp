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
#include "test_syr.h"

class ssyrGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   float,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P( ssyrGeneric, API )
{
    using T = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is u,l
    char uploa = std::get<1>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<2>(GetParam());
    // matrix size n
    gtint_t n  = std::get<3>(GetParam());
    // specifies alpha value
    float alpha = std::get<4>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<5>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<6>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite syr.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_syr<T>( storage, uploa, conjx, n, alpha, incx, lda_inc, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_syr<T>( storage, uploa, conjx, n, alpha, incx, lda_inc, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_syr<T>( storage, uploa, conjx, n, alpha, incx, lda_inc, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_syr<T>( storage, uploa, conjx, n, alpha, incx, lda_inc, thresh );
#endif
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        BlackboxSmall,
        ssyrGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n'),                                          // conjx
            ::testing::Range(gtint_t(1),gtint_t(21),1),                      // n
            ::testing::Values( 0.0, 1.0, -1.0, 2.7 ),                        // alpha
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for x
            ::testing::Values(gtint_t(0), gtint_t(5))                        // increment to the leading dim of a
        ),
        ::syrGenericPrint<float>()
    );

INSTANTIATE_TEST_SUITE_P(
        BlackboxMedium,
        ssyrGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(25),
                              gtint_t(33),
                              gtint_t(98),
                              gtint_t(173),
                              gtint_t(211)
                            ),                                               // n
            ::testing::Values( 0.0, 1.0, -1.0, 2.7 ),                        // alpha
            ::testing::Values(gtint_t(1),gtint_t(-1),gtint_t(2)),            // stride size for x
            ::testing::Values(gtint_t(0), gtint_t(5))                        // increment to the leading dim of a
        ),
        ::syrGenericPrint<float>()
    );
