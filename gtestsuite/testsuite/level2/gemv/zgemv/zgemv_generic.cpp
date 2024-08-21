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
#include "level2/gemv/test_gemv.h"

using T = dcomplex;

class zgemvGeneric :
        public ::testing::TestWithParam<std::tuple<char,        // storage format
                                                   char,        // transa
                                                   char,        // conjx
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   T,           // alpha
                                                   T,           // beta
                                                   gtint_t,     // incx
                                                   gtint_t,     // incy
                                                   gtint_t,     // lda_inc
                                                   bool>> {};   // is_memory_test

TEST_P( zgemvGeneric, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // specifies beta value
    T beta = std::get<6>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<7>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<8>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<10>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gemv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() && (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        if(( transa == 'n' ) || ( transa == 'N' ))
            thresh = (3*n+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = (3*m+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemv<T>( storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test );
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                          // storage format
            ::testing::Values('n','c','t'),                             // transa
            ::testing::Values('n'),                                     // conjx
            ::testing::Range(gtint_t(10), gtint_t(31), 10),             // m
            ::testing::Range(gtint_t(10), gtint_t(31), 10),             // n
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.0, -2.0}),  // alpha
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{-1.0, 1.0}),  // beta
            ::testing::Values(gtint_t(1)),                              // stride size for x
            ::testing::Values(gtint_t(1)),                              // stride size for y
            ::testing::Values(gtint_t(0)),                              // increment to the leading dim of a
            ::testing::Values(false, true)                              // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Tiny_Matrixsizes,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n','c'),                                      // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                     // m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                     // n
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.0, -2.0}),       // alpha
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.0, -2.0}),       // beta
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(gtint_t(7), gtint_t(3)),                       // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Average_Matrixsizes,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                                          // storage format
            ::testing::Values('t','c'),                                                 // transa
            ::testing::Values('n'),                                                     // conjx
            ::testing::Range(gtint_t(128), gtint_t(512), 31),                           // m
            ::testing::Range(gtint_t(512), gtint_t(128), -31),                          // n
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{-1.0, 2.0}, T{-2.0, 1.0}),    // alpha
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{-1.0, -3.1}),                 // beta
            ::testing::Values(gtint_t(1)),                                              // stride size for x
            ::testing::Values(gtint_t(1)),                                              // stride size for y
            ::testing::Values(gtint_t(1)),                                              // increment to the leading dim of a
            ::testing::Values(false, true)                                              // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Large_Matrixsizes,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Range(gtint_t(1024), gtint_t(32767), 1023),           // m
            ::testing::Range(gtint_t(1024), gtint_t(32767), 1023),           // n
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.1, 2.1}),        // alpha
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.1, 2.1}),        // beta
            ::testing::Values(gtint_t(11), gtint_t(119), gtint_t(211)),      // stride size for x
            ::testing::Values(gtint_t(211), gtint_t(119), gtint_t(11)),      // stride size for y
            ::testing::Values(gtint_t(1), gtint_t(252)),                     // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),

      ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Unit_MN,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n','c','t'),                                  // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(1)),                                   // m
            ::testing::Values(gtint_t(1)),                                   // n
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{1.0, -0.1}),       // alpha
            ::testing::Values(T{1.0, 1.0}, T{0.0, 0.0}, T{0.1, 1.0},
                              T{-2.0, 1.0}, T{-3.0, 2.0}, T{-1.0, -2.0}),    // beta
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );
