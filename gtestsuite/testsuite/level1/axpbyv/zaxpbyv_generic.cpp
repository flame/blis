/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_axpbyv.h"

class zaxpbyvAccTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   dcomplex>> {};
// Tests using random integers as vector elements.
TEST_P(zaxpbyvAccTest, RandomData)
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
    // alpha
    T alpha = std::get<4>(GetParam());
    // beta
    T beta = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpbyv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
    {
        // Like SCALV
        if (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>())
            thresh = 0.0;
        else
            thresh = testinghelpers::getEpsilon<T>();
    }
    else if (beta == testinghelpers::ZERO<T>())
    {
        // Like SCAL2V
        if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
            thresh = 0.0;
        else
            thresh = testinghelpers::getEpsilon<T>();
    }
    else if (beta == testinghelpers::ONE<T>())
    {
        // Like AXPYV
        if (alpha == testinghelpers::ZERO<T>())
            thresh = 0.0;
        else
            thresh = 2*testinghelpers::getEpsilon<T>();
    }
    else if (alpha == testinghelpers::ONE<T>())
        thresh = 2*testinghelpers::getEpsilon<T>();
    else
        thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv<T>(conj_x, n, incx, incy, alpha, beta, thresh);
}

/*
    The code structure for bli_zaxpbyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 8 --> L8
        Fringe loops :  In blocks of 6 --> L6
                        In blocks of 4 --> L4
                        In blocks of 2 --> L2

    For non-unit strides : A single loop, to process element wise.
    NOTE : Any size, requiring the fringe case of 1 with unit stride falls to
           the non-unit stride loop and executes it once for just the last element.
*/

// Accuracy testing of the main loop, single and multiple runs
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_acc_unitStrides_main,
    zaxpbyvAccTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(8), gtint_t(40)), // m
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{1.0, 2.0}) // beta
        ),
    ::axpbyvGenericPrint<dcomplex>());

// Accuracy testing of different combinations of fringe loops(L6, L4, L2, 1)
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_acc_unitStrides_fringe,
    zaxpbyvAccTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Range(gtint_t(1), gtint_t(7), 1), // m
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{1.0, 2.0}) // beta
        ),
    ::axpbyvGenericPrint<dcomplex>());

// Accuracy testing of 3*L8 + L6 + L4 + L2 + 1, a case of main + all fringe cases taken
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_acc_unitStrides_combine,
    zaxpbyvAccTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(30), gtint_t(34), gtint_t(36), gtint_t(37)), // m
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{1.0, 2.0}) // beta
        ),
    ::axpbyvGenericPrint<dcomplex>());

// Accuracy testing with non-unit strides
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_acc_nonUnitStrides,
    zaxpbyvAccTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(10), gtint_t(17)), // m
        ::testing::Values(
#ifndef TEST_BLIS_TYPED
                          gtint_t(-3),
#endif
                          gtint_t(4)), // stride size for x
        ::testing::Values(
#ifndef TEST_BLIS_TYPED
                          gtint_t(-2),
#endif
                          gtint_t(6)), // stride size for y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{1.0, 2.0}) // beta
        ),
    ::axpbyvGenericPrint<dcomplex>());
