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

class zdscalvEVT :
        public ::testing::TestWithParam<std::tuple<char,            // conj_alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // xi
                                                   dcomplex,        // x_exval
                                                   double>> {};     // alpha

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zdscalvEVT);

// Tests using random integers as vector elements.
TEST_P( zdscalvEVT, API )
{
    using T = dcomplex;
    using RT = double;
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
    RT alpha = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scalv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<RT>() || alpha == testinghelpers::ONE<RT>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scalv<T, RT>( conj_alpha, n, incx, xi, x_exval, alpha, thresh );
}

// bli_zdscal not present in BLIS
#ifndef TEST_BLIS_TYPED

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Tests for Zen3 Architecture.
/**
 * Tests for bli_zdscalv_zen_int10 (AVX2) kernel.
 * Loops:
 * L30     - Main loop, handles 30 elements
 * L24     - handles 24 elements
 * L16     - handles 16 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
 *
 * n = 105  : L30*3 + L8 + L4 + L2 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 69  - L30
 *      97  - L8
 *     101  - L4
 *     103  - L2
 *     104  - LScalar
 *
 * n = 79  : L30*2 + L16 + L2 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 58  - L30
 *      69  - L16
 *      77  - L2
 *      78  - LScalar
 *
 * n = 59  : L30 + L24 + L4 + LScalar
 * Indices  - Loop into which extreme value is induced
 *       0  - L30
 *      51  - L24
 *      55  - L4
 *      58  - LScalar
*/
// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
    vec_unitStride_zen3,
    zdscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c' // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(105),
                           gtint_t( 79),
                           gtint_t( 59)
            ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( // n = 105
                           gtint_t(0),      // L30
                           gtint_t(97),     // L8
                           gtint_t(101),    // L4
                           gtint_t(103),    // L2
                           gtint_t(104),    // LScalar

                           // n = 79
                           gtint_t(69),     // L16
                           gtint_t(77),     // L2
                           gtint_t(78),     // LScalar

                           // n = 59
                           gtint_t(51),     // L24
                           gtint_t(55),     // L4
                           gtint_t(58)      // LScalar
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
        ::testing::Values( double(-5.1),
                           double(-1.0),
                           double( 0.0),
                           double( 1.0),
                           double( 7.3)
        )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

// Tests for Zen4 Architecture.
/**
 * Tests for bli_zdscalv_zen_int_avx512 (AVX512) kernel.
 * Loops:
 * L16     - Main loop, handles 16 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
 *
 * n = 63   : L16*3 + L8 + L4 + L2 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 31  - L16
 *      48  - L8
 *      56  - L4
 *      60  - L2
 *      62  - LScalar
*/
// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
    vec_unitStride_zen4,
    zdscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c' // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(63) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( // n = 63
                           gtint_t(0),      // L16
                           gtint_t(31),     // l16
                           gtint_t(48),     // L8
                           gtint_t(56),     // L4
                           gtint_t(60),     // L2
                           gtint_t(62)      // LScalar
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
        ::testing::Values( double(-5.1),
                           double(-1.0),
                           double( 0.0),
                           double( 1.0),
                           double( 7.3)
        )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

// EVT with non-unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
    vec_nonUnitStride,
    zdscalvEVT,
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
        ::testing::Values( double(-5.1),
                           double(-1.0),
                           double( 0.0),
                           double( 1.0),
                           double( 7.3)
        )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
    alpha_unitStride_zen3,
    zdscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c'  // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(105),
                           gtint_t( 79),
                           gtint_t( 59) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(0) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{0.0, 0.0} ),
        // alpha: value of scalar.
        ::testing::Values( NaN, Inf, -Inf )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
    alpha_unitStride_zen4,
    zdscalvEVT,
    ::testing::Combine(
        // conj(alpha): uses n (no_conjugate) since it is real.
        ::testing::Values( 'n'
#ifdef TEST_BLIS_TYPED
                         , 'c'  // conjugate option is BLIS-api specific.
#endif
        ),
        // m: size of vector.
        ::testing::Values( gtint_t(63) ),
        // incx: stride of x vector.
        ::testing::Values( gtint_t(1) ),
        // xi: index of extreme value for x.
        ::testing::Values( gtint_t(0) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{0.0, 0.0} ),
        // alpha: value of scalar.
        ::testing::Values( NaN, Inf, -Inf )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
    alpha_nonUnitStride,
    zdscalvEVT,
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
        ::testing::Values( gtint_t(0) ),
        // x_exval: extreme value for x.
        ::testing::Values( dcomplex{0.0, 0.0} ),
        // alpha: value of scalar.
        ::testing::Values( NaN, Inf, -Inf )
    ),
    (::scalvEVTPrint<dcomplex, double>())
);

#endif // not TEST_BLIS_TYPED
