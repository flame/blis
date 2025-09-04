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
#include "level1/axpyv/test_axpyv.h"

class zaxpyvGeneric :
        public ::testing::TestWithParam<std::tuple<char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   dcomplex>> {};   // alpha
// Tests using random integers as vector elements.
TEST_P( zaxpyvGeneric, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y
    char conj_x = std::get<0>(GetParam());
    // vector length
    gtint_t n = std::get<1>(GetParam());
    // stride size for x
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y
    gtint_t incy = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpyv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (alpha == testinghelpers::ONE<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.02;
#else
        double adj = 1.0;
#endif
        thresh = adj*2*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
#endif
}

// Black box testing for generic and main use of zaxpy.
INSTANTIATE_TEST_SUITE_P(
        unitStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),                                                               // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(dcomplex{-3.7, 1.2}, dcomplex{1.5, 2.6},
                              dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                           // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        nonUnitPositiveStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),                                                               // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(2)),                                   // stride size for x
            ::testing::Values(gtint_t(3)),                                   // stride size for y
            ::testing::Values(dcomplex{-3.7, 1.2}, dcomplex{1.5, 2.6},
                              dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                           // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );

#ifndef TEST_BLIS_TYPED
// Test for negative increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        negativeStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(-4)),                                  // stride size for x
            ::testing::Values(gtint_t(-3)),                                  // stride size for y
            ::testing::Values(dcomplex{-3.7, 1.2}, dcomplex{1.5, 2.6},
                              dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                           // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );
#endif
// To cover small, medium and large sizes of M with unit increment.
INSTANTIATE_TEST_SUITE_P(
        DifferentSizesOfM,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(36),                                   //m size of vector
                              gtint_t(1000),
                              gtint_t(2999),
                              gtint_t(3666),
                              gtint_t(777)),
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(dcomplex{2.0, 1.1},
                              dcomplex{0.0, 0.0},
                              dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                            // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );
//incx and incy are zero.
INSTANTIATE_TEST_SUITE_P(
        ZeroIncrements,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(10)),                                  // m size of vector
            ::testing::Values(gtint_t(0),gtint_t(2)),                        // stride size for x
            ::testing::Values(gtint_t(3),gtint_t(0)),                        // stride size for y
            ::testing::Values(dcomplex{4.0, 3.1},
                              dcomplex{0.0, 0.0},
                              dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                            // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );
//To cover large sizes with non unit increments.
INSTANTIATE_TEST_SUITE_P(
        largeSize,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(1000)),                                // m size of vector
            ::testing::Values(gtint_t(2)),                                   // stride size for x
            ::testing::Values(gtint_t(3)),                                   // stride size for y
            ::testing::Values(dcomplex{4.0, 3.1},
                              dcomplex{0.0, 0.0},
                              dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                          // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );
//incx and incy is greater than size of a vector m.
INSTANTIATE_TEST_SUITE_P(
        strideGreaterThanSize,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, c: use conj(x)
            ::testing::Values(gtint_t(6)),                                   // m size of vector
            ::testing::Values(gtint_t(10)),                                  // stride size for x
            ::testing::Values(gtint_t(14)),                                  // stride size for y
            ::testing::Values(dcomplex{4.0, 3.1},
                              dcomplex{0.0, 0.0},
                              dcomplex{1.0, 0.0},
                              dcomplex{-1.0, 0.0})                            // alpha
        ),
        ::axpyvGenericPrint<dcomplex>()
    );
