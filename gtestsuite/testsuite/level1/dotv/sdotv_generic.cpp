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
#include "test_dotv.h"

class sdotvGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

// Tests using random integers as vector elements.
TEST_P( sdotvGeneric, API )
{
    using T = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether vec x is n,c
    char conjx = std::get<0>(GetParam());
    // denotes whether vec y is n,c
    char conjy = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<4>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite dotv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = 2*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_dotv<T>( conjx, conjy, n, incx, incy, thresh );
}

// Black box testing for generic and main use of sdotv.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        sdotvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Values('n'),                                          // n: use y, not conj(y) (since it is real)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::dotvGenericPrint()
    );

#ifdef TEST_BLIS_TYPED // BLIS-api specific
// Test when conjugate of x is used as an argument.
// Only test very few cases as sanity check since conj(x) = x for real types.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        ConjX,
        sdotvGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // c: use conj(x)
            ::testing::Values('c'),                                          // c: use conj(y)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::dotvGenericPrint()
    );
#endif

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitPositiveIncrements,
        sdotvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Values('n'),                                          // n: use y, not conj(y) (since it is real)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(2), gtint_t(11)),                      // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(33))                       // stride size for y
        ),
        ::dotvGenericPrint()
    );

#ifndef TEST_BLIS_TYPED
// Test for negative increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NegativeIncrements,
        sdotvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values('n'),                                          // n: use y, c: use conj(y)
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(-2)),                                  // stride size for x
            ::testing::Values(gtint_t(-3))                                   // stride size for y
        ),
        ::dotvGenericPrint()
    );
#endif
