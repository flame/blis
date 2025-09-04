/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level1/scalv/test_scalv.h"

class csscalvGeneric :
        public ::testing::TestWithParam<std::tuple<char,        // conj_alpha
                                                   gtint_t,     // n
                                                   gtint_t,     // incx
                                                   float>> {}; // alpha

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(csscalvGeneric);

// Tests using random integers as vector elements.
TEST_P( csscalvGeneric, API )
{
    using T = scomplex;
    using U = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether alpha or conj(alpha) will be used:
    char conj_alpha = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // alpha
    U alpha = std::get<3>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scalv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<U>() || alpha == testinghelpers::ONE<U>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_scalv<T, U>( conj_alpha, n, incx, alpha, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_scalv<T, U>( conj_alpha, n, incx, alpha, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_scalv<T, U>( conj_alpha, n, incx, alpha, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_scalv<T, U>( conj_alpha, n, incx, alpha, thresh );
#endif
}

// bli_csscal not present in BLIS
#ifndef TEST_BLIS_TYPED

// Black box testing for generic use of dscal.
INSTANTIATE_TEST_SUITE_P(
        unitPositiveIncrementSmall,
        csscalvGeneric,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Range(gtint_t(1), gtint_t(101), 1),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 7.0),
                                float(-3.0)
            )
        ),
        (::scalvGenericPrint<float, float>())
    );

// Black box testing for generic use of dscal.
INSTANTIATE_TEST_SUITE_P(
        unitPositiveIncrementLarge,
        csscalvGeneric,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(gtint_t(111), gtint_t(193), gtint_t(403)),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 7.0),
                                float(-3.0)
            )
        ),
        (::scalvGenericPrint<float, float>())
    );

INSTANTIATE_TEST_SUITE_P(
        nonUnitPositiveIncrementSmall,
        csscalvGeneric,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Range(gtint_t(1), gtint_t(9), 1),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(2),
                                gtint_t(41)
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 7.0),
                                float(-3.0)
            )
        ),
        (::scalvGenericPrint<float, float>())
    );

INSTANTIATE_TEST_SUITE_P(
        nonUnitPositiveIncrementLarge,
        csscalvGeneric,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(gtint_t(111), gtint_t(193), gtint_t(403)),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(2),
                                gtint_t(41)
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 7.0),
                                float(-3.0)
            )
        ),
        (::scalvGenericPrint<float, float>())
    );

// alpha=0 testing only for BLAS and CBLAS as
// BLIS uses setv and won't propagate Inf and NaNs
INSTANTIATE_TEST_SUITE_P(
        alphaZero,
        csscalvGeneric,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Range(gtint_t(1), gtint_t(101), 1),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1),
                                gtint_t(2),
                                gtint_t(41)
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                double( 0.0)
            )
        ),
        (::scalvGenericPrint<float, float>())
    );

// Test for negative increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NegativeIncrements,
        csscalvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(-2), gtint_t(-1)),                     // stride size for x
            ::testing::Values(3)                                             // alpha
        ),
        (::scalvGenericPrint<float, float>())
    );

#endif // not TEST_BLIS_TYPED






