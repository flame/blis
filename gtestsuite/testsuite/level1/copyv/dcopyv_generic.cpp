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

class dcopyvGeneric :
        public ::testing::TestWithParam<std::tuple<char,                     // n: use x, c: use conj(x)
                                                   gtint_t,                  // m size of vector
                                                   gtint_t,                  // stride size for x
                                                   gtint_t>> {};             // stride size for y

// Tests using random values as vector elements.
TEST_P( dcopyvGeneric, API )
{
    using T = double;
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

// Black box testing for generic and main use of scopy.
INSTANTIATE_TEST_SUITE_P(
        smallSize,
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );

#ifdef TEST_BLIS_TYPED // BLIS-api specific
// Test when conjugate of x is used as an argument.
// Only test very few cases as sanity check since conj(x) = x for real types.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        ConjX,
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // c: use conj(x)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );
#endif

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitPositiveIncrements,
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // use x, not conj(x) (since it is real)
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
        dcopyvGeneric,
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
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(1270),
                              gtint_t(64),
                              gtint_t(32),
                              gtint_t(16),
                              gtint_t(8),
                              gtint_t(4),
                              gtint_t(960),
                              gtint_t(3120),
                              gtint_t(1900),
                              gtint_t(124),
                              gtint_t(880),
                              gtint_t(80),
                              gtint_t(256),
                              gtint_t(480),
                              gtint_t(788),
                              gtint_t(36),
                              gtint_t(24)),                                  // m size of vector
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );
//To cover large sizes with non unit increments.
INSTANTIATE_TEST_SUITE_P(
        largeSize,
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(1000)),                                // m size of vector
            ::testing::Values(gtint_t(2)),                                   // stride size for x
            ::testing::Values(gtint_t(3))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );
//incx and incy is greater than size of a vector m.
INSTANTIATE_TEST_SUITE_P(
        StrideGreaterThanSize,
        dcopyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(4)),                                   // m size of vector
            ::testing::Values(gtint_t(6)),                                   // stride size for x
            ::testing::Values(gtint_t(8))                                    // stride size for y
        ),
        ::copyvGenericPrint()
    );
