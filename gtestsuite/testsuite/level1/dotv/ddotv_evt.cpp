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
#include "test_dotv.h"

class ddotvEVT :
        public ::testing::TestWithParam<std::tuple<char,            // conjx
                                                   char,            // conjy
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // xi
                                                   double,          // xexval
                                                   gtint_t,         // incy
                                                   gtint_t,         // yi
                                                   double>> {};     // yexval

// Tests using random integers as vector elements.
TEST_P( ddotvEVT, API )
{
    using T = double;
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
    // index of extreme value for x:
    gtint_t xi = std::get<4>(GetParam());
    // extreme value for x:
    double x_exval = std::get<5>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<6>(GetParam());
    // index of extreme value for y:
    gtint_t yi = std::get<7>(GetParam());
    // extreme value for y:
    double y_exval = std::get<8>(GetParam());

    // Set the threshold for the errors:
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = 2*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_dotv<T>( conjx, conjy, n, incx, xi, x_exval, incy, yi, y_exval, thresh );
}


static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Tests for Zen4 Architecture.
/**
 * bli_ddotv_zen_int_avx512( ... )
 * Loops:
 * L40     - Main loop, handles 40 elements
 * L16     - handles 16 elements
 * L8      - handles 8 elements
 * LScalar - leftover loop
 *
 * n = 109  : L40*2 + L16 + L8 + LScalar
 * Indices  - Loop into which extreme value is induced
 *    0, 79 - L40
 *       93 - L16
 *      101 - L8
 *      108 - LScalar
 */
// EVT with unit stride X vector containing Infs/NaNs.
// Unit stride Y vector contains random elements.
INSTANTIATE_TEST_SUITE_P(
        vecX_unitStride_zen4,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(109)
            ),
            // incx: stride of x vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(79), gtint_t(93),
                                gtint_t(101), gtint_t(108)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(0) ),        // set as 0 since testing only for x
            // y_exval: extreme value for y.
            ::testing::Values( double(0.0) )        // dummy value since testing only for x
        ),
        ::dotvEVTPrint<double>()
    );


// EVT with unit stride Y vector containing Infs/NaNs.
// Unit stride X vector contains random elements.
INSTANTIATE_TEST_SUITE_P(
        vecY_unitStride_zen4,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values( gtint_t(109) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),        // unit stride
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(0) ),        // set as 0 since testing only for y
            // x_exval: extreme value for x.
            ::testing::Values( double(0.0) ),       // dummy value since testing only for y
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),        // unit stride
            // yi: index of extreme value for y.
            ::testing::Values(
                                gtint_t(0), gtint_t(79), gtint_t(93),
                                gtint_t(101), gtint_t(108)
             ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );

// EVT with unit stride vectors X and Y contatining Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vecXY_unitStride_zen4,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(109)
            ),
            // incx: stride of x vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(79), gtint_t(93),
                                gtint_t(101), gtint_t(108)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // yi: index of extreme value for y.
            ::testing::Values(
                                gtint_t(0), gtint_t(79), gtint_t(93),
                                gtint_t(101), gtint_t(108)
            ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );

// Tests for Zen3 Architecture.
/**
 * bli_ddotv_zen_int10( ... )
 * Loops:
 * L40     - Main loop, handles 40 elements
 * L20     - handles 20 elements
 * L16     - handles 16 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * LScalar - leftover loop
 *
 * n = 119  : L40*2 + L20 + L16 + LScalar
 * Indices  - Loop into which extreme value is induced
 *    0, 78 - L40
 *       94 - L20
 * 101, 110 - L16
 *      112 - L16
 *      118 - LScalar
 *
 * n = 113  : L40*2 + L20 + L8 + L4 + LScalar
 * Indices - Loop into which extreme value is induced
 *   0, 78 - L40
 *      94 - L20
 *     101 - L8
 *     110 - L4
 *     112 - LScalar
 */
// EVT with unit stride X vector containing Infs/NaNs.
// Unit stride Y vector contains random elements.
INSTANTIATE_TEST_SUITE_P(
        vecX_unitStride_zen3,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(119),
                                gtint_t(113)
            ),
            // incx: stride of x vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(78), gtint_t(94),
                                gtint_t(101), gtint_t(110), gtint_t(112),
                                gtint_t(118)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(0) ),        // set as 0 since testing only for x
            // y_exval: extreme value for y.
            ::testing::Values( double(0.0) )        // dummy value since testing only for x
        ),
        ::dotvEVTPrint<double>()
    );

// EVT with unit stride Y vector containing Infs/NaNs.
// Unit stride X vector contains random elements.
INSTANTIATE_TEST_SUITE_P(
        vecY_unitStride_zen3,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(119),
                                gtint_t(113)
            ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),        // unit stride
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(0) ),        // set as 0 since testing only for y
            // x_exval: extreme value for x.
            ::testing::Values( double(0.0) ),       // dummy value since testing only for y
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),        // unit stride
            // yi: index of extreme value for y.
            ::testing::Values(
                                gtint_t(0), gtint_t(78), gtint_t(94),
                                gtint_t(110), gtint_t(118)
            ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );

// EVT with unit stride vectors X and Y contatining Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vecXY_unitStride_zen3,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(119),
                                gtint_t(115)
            ),
            // incx: stride of x vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(79), gtint_t(93),
                                gtint_t(101), gtint_t(108)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values(gtint_t(1)),          // unit stride
            // yi: index of extreme value for y.
            ::testing::Values(
                                gtint_t(0), gtint_t(78), gtint_t(94),
                                gtint_t(110), gtint_t(118)
            ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );

// EVT with non-unit stride vectors X and Y containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vecXY_nonUnitStride,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values( gtint_t(55) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(3) ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(1), gtint_t(27), gtint_t(51) ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(7) ),
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(3), gtint_t(29), gtint_t(47) ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );

// EVT with negative stride vectors X and Y containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vecXY_negativeStride,
        ddotvEVT,
        ::testing::Combine(
            // conj(x): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // conj(y): user n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values( gtint_t(55) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(-3) ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(1), gtint_t(27), gtint_t(51) ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(-7) ),
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(3), gtint_t(29), gtint_t(47) ),
            // y_exval: extreme value for y.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dotvEVTPrint<double>()
    );
