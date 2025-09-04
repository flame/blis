/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

class zscalvEVT :
        public ::testing::TestWithParam<std::tuple<char,            // conj_alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // xi
                                                   dcomplex,        // x_exval
                                                   dcomplex>> {};   // alpha


// Tests using random integers as vector elements.
TEST_P( zscalvEVT, API )
{
    using T = dcomplex;
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
    // index of extreme value for x:
    gtint_t xi = std::get<3>(GetParam());
    // extreme value for x:
    T x_exval = std::get<4>(GetParam());
    // alpha:
    T alpha = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scalv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scalv<T>( conj_alpha, n, incx, xi, x_exval, alpha, thresh );
}

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Tests for Zen3 Architecture.
/**
 * Tests for bli_zscalv_zen_int (AVX2) kernel.
 * Loops:
 * L8      - Main loop, handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
    vec_unitStride_zen3,
    zscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c'  // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(71) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(0), gtint_t(64), gtint_t(67),
                           gtint_t(69), gtint_t(70)
        ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{ NaN,  0.0},
                           dcomplex{ Inf,  0.0},
                           dcomplex{-Inf,  0.0},
                           dcomplex{ 0.0,  Inf},
                           dcomplex{-2.1,  NaN},
                           dcomplex{ 1.2, -Inf},
                           dcomplex{ NaN,  Inf},
                           dcomplex{ Inf,  NaN},
                           dcomplex{ NaN,  NaN},
                           dcomplex{ Inf, -Inf}
        ),
        // alpha: value of scalar.
        ::testing::Values( dcomplex{-5.1, -7.3},
                           dcomplex{-1.0, -1.0},
                           dcomplex{ 0.0,  0.0},
                           dcomplex{ 1.0,  1.0},
                           dcomplex{ 7.3,  5.1}
        )
    ),
    (::scalvEVTPrint<dcomplex, dcomplex>())
);

// EVT with non-unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
    vec_nonUnitStride,
    zscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c'  // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(55) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(3) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(1), gtint_t(27), gtint_t(51) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{ NaN,  NaN},
                           dcomplex{ NaN,  Inf},
                           dcomplex{ NaN, -Inf},
                           dcomplex{ Inf,  NaN},
                           dcomplex{ Inf,  Inf},
                           dcomplex{ Inf, -Inf},
                           dcomplex{-Inf,  NaN},
                           dcomplex{-Inf,  Inf},
                           dcomplex{-Inf, -Inf}
        ),
        // alpha: value of scalar.
        ::testing::Values( dcomplex{-5.1, -7.3},
                           dcomplex{-1.0, -1.0},
                           dcomplex{ 0.0,  0.0},
                           dcomplex{ 1.0,  1.0},
                           dcomplex{ 7.3,  5.1}
        )
    ),
    (::scalvEVTPrint<dcomplex, dcomplex>())
);

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
    alpha_unitStride_zen3,
    zscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c' // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(71) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(0) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{0.0, 0.0} ),
        // alpha: value of scalar.
        ::testing::Values( dcomplex{ NaN,  NaN},
                           dcomplex{ NaN,  Inf},
                           dcomplex{ NaN, -Inf},
                           dcomplex{ Inf,  NaN},
                           dcomplex{ Inf,  Inf},
                           dcomplex{ Inf, -Inf},
                           dcomplex{-Inf,  NaN},
                           dcomplex{-Inf,  Inf},
                           dcomplex{-Inf, -Inf}
        )
    ),
    (::scalvEVTPrint<dcomplex, dcomplex>())
);

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
    alpha_nonUnitStride,
    zscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c' // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(55) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(3) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(0) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{0.0, 0.0} ),
        // alpha: value of scalar.
        ::testing::Values( dcomplex{ NaN,  NaN},
                           dcomplex{ NaN,  Inf},
                           dcomplex{ NaN, -Inf},
                           dcomplex{ Inf,  NaN},
                           dcomplex{ Inf,  Inf},
                           dcomplex{ Inf, -Inf},
                           dcomplex{-Inf,  NaN},
                           dcomplex{-Inf,  Inf},
                           dcomplex{-Inf, -Inf}
        )
    ),
    (::scalvEVTPrint<dcomplex, dcomplex>())
);
