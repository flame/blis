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
#include "test_xpbyv.h"

class zxpbyvGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zxpbyvGeneric);

// Tests using random integers as vector elements.
TEST_P( zxpbyvGeneric, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // beta
    T beta = std::get<4>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite xpbyv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (beta == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (beta == testinghelpers::ONE<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = 2*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_xpbyv<T>( conj_x, n, incx, incy, beta, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_xpbyv<T>( conj_x, n, incx, incy, beta, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_xpbyv<T>( conj_x, n, incx, incy, beta, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_xpbyv<T>( conj_x, n, incx, incy, beta, thresh );
#endif
}

#ifdef TEST_BLIS_TYPED
// Black box testing for generic and main use of zaxpby.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values('n', 'c'),                                     // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   /*(gtint_t(-5), gtint_t(-17))*/  // stride size for x
            ::testing::Values(gtint_t(1)),                                   /*(gtint_t(-12), gtint_t(-4))*/  // stride size for y
            ::testing::Values(dcomplex{2.0, -1.0}, dcomplex{-2.0, 3.0})      // beta
        ),
        ::xpbyvGenericPrint<dcomplex>()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        zxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values('n', 'c'),                                     // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(2), gtint_t(11)),                      /*(gtint_t(-5), gtint_t(-17))*/  // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(33)),                      /*(gtint_t(-12), gtint_t(-4))*/  // stride size for y
            ::testing::Values(dcomplex{4.0, 3.1})                            // beta
        ),
        ::xpbyvGenericPrint<dcomplex>()
    );
#endif
