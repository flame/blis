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
#include "test_copyv.h"

class ccopyvGeneric :
        public ::testing::TestWithParam<std::tuple<char,                     // n: use x, c: use conj(x)
                                                   gtint_t,                  // m size of vector
                                                   gtint_t,                  // stride size for x
                                                   gtint_t>> {};             // stride size for y

// Tests using random values as vector elements.
TEST_P( ccopyvGeneric, API )
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether vec x is n,c
    char conjx = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_copyv<T>( conjx, n, incx, incy );
}

// Black box testing for generic and main use of ccopy.
INSTANTIATE_TEST_SUITE_P(
        smallSize,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),                                                               // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),                                                               // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(2), gtint_t(11)),                      // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(33))                       // stride size for y
        ),
        ::copyvGenericPrint()
    );

#ifndef TEST_BLIS_TYPED
// Test for negative increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NegativeIncrements,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(-5), gtint_t(7)),                      // stride size for x
            ::testing::Values(gtint_t(13), gtint_t(-9))                      // stride size for y
        ),
        ::copyvGenericPrint()
    );
#endif
// To cover small, medium and large sizes of M with unit increment.
INSTANTIATE_TEST_SUITE_P(
        differentSizesOfM,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(1760),
                              gtint_t(255),
                              gtint_t(1280),
                              gtint_t(64),
                              gtint_t(32),
                              gtint_t(16),
                              gtint_t(8),
                              gtint_t(1920),
                              gtint_t(2240),
                              gtint_t(5400),
                              gtint_t(2483),
                              gtint_t(184),
                              gtint_t(160),
                              gtint_t(1916),
                              gtint_t(908),
                              gtint_t(732)),                                // m size of vector
            ::testing::Values(gtint_t(1)),                                  // stride size for x
            ::testing::Values(gtint_t(1))                                   // stride size for y
        ),
        ::copyvGenericPrint()
    );
//To cover large sizes with non unit increments.
INSTANTIATE_TEST_SUITE_P(
        largeSize,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(3000)),                                // m size of vector
            ::testing::Values(gtint_t(5)),                                   // stride size for x
            ::testing::Values(gtint_t(2))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );
//incx and incy is greater than size of a vector m.
INSTANTIATE_TEST_SUITE_P(
        strideGreaterThanSize,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(3)),                                   // m size of vector
            ::testing::Values(gtint_t(55)),                                  // stride size for x
            ::testing::Values(gtint_t(66))                                   // stride size for y
        ),
        ::copyvGenericPrint()
    );
