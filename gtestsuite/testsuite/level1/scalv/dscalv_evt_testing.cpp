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
#include "test_scalv.h"

class dscalv_EVT :
        public ::testing::TestWithParam<std::tuple<char,        // conj_alpha
                                                   gtint_t,     // n
                                                   gtint_t,     // incx
                                                   gtint_t,     // xi
                                                   double,      // x_exval
                                                   double>> {}; // alpha


// Tests using random integers as vector elements.
TEST_P( dscalv_EVT, ExceptionData )
{
    using T = double;
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
    double x_exval = std::get<4>(GetParam());
    // alpha:
    T alpha = std::get<5>(GetParam());

    // Set the threshold for the errors:
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

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class dscalv_EVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, gtint_t, gtint_t, gtint_t, double, double>> str) const {
        char    conjx   = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t incx    = std::get<2>(str.param);
        gtint_t xi      = std::get<3>(str.param);
        double  x_exval = std::get<4>(str.param);
        double  alpha   = std::get<5>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dscal_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dscal";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_dscalv";
#endif
        str_name += "_n" + std::to_string(n);
        str_name += (conjx == 'n') ? "_noconjx" : "_conjx";
        std::string incx_str = ( incx >= 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        str_name = str_name + "_X_" + std::to_string(xi);
        str_name = str_name + "_" + testinghelpers::get_value_string(x_exval);
        std::string alpha_str = testinghelpers::get_value_string(alpha);// ( alpha >= 0) ? std::to_string(int(alpha)) : "m" + std::to_string(int(std::abs(alpha)));
        str_name = str_name + "_a" + alpha_str;

        return str_name;
    }
};

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Tests for Zen4 Architecture.
/**
 * bli_dscalv_zen_int_avx512( ... )
 * Loops:
 * L64     - Main loop, handles 64 elements
 * L32     - handles 32 elements
 * L16     - handles 16 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
 *
 * n = 383  : L64*5 + L20 + L16 + L8 + L4 + L2 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 319 - L64
 *      351 - L32
 *      367 - L16
 *      375 - L8
 *      379 - L4
 *      380 - L2
 *      382 - LScalar
 */
// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vec_unitStride_zen4,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(383)
                ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)
            ),
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(319), gtint_t(351),
                                gtint_t(367), gtint_t(375), gtint_t(379),
                                gtint_t(380), gtint_t(382)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // alpha: value of scalar.
            ::testing::Values(
                                double(-3.3),
                                double(-1.0),
                                double( 0.0),
                                double( 1.0),
                                double( 7.3)
            )
        ),
        ::dscalv_EVTPrint()
    );

// Tests for Zen3 Architecture.
/**
 * bli_dscalv_zen_int10( ... )
 * Loops:
 * L64     - Main loop, handles 64 elements
 * L48     - handles 48 elements
 * L32     - handles 32 elements
 * L12     - handles 12 elements
 * L4      - handles 4 elements
 * LScalar - leftover loop
 *
 * n = 565  : L64*8 + L48 + L4 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 511 - L64
 * 520, 525 - L48
 * 528, 555 - L48
 *      561 - L4
 *      564 - LScalar
 *
 * n = 556  : L64*8 + L32 + L12
 * Indices  - Loop into which extreme value is induced
 *   0, 511 - L64
 * 520, 525 - L32
 *      555 - L12
 *
 * n = 529  : L64*8 + L12 + L4 + LScalar
 * Indices  - Loop into which extreme value is induced
 *   0, 511 - L64
 *      520 - L12
 *      525 - L4
 *      528 - LScalar
 */
// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vec_unitStride_zen3,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(565),
                                gtint_t(556),
                                gtint_t(529)
                ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)
            ),
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(0), gtint_t(511), gtint_t(520),
                                gtint_t(525), gtint_t(528), gtint_t(555),
                                gtint_t(561), gtint_t(564)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // alpha: value of scalar.
            ::testing::Values(
                                double(-3.3),
                                double(-1.0),
                                double( 0.0),
                                double( 1.0),
                                double( 7.3)
            )
        ),
        ::dscalv_EVTPrint()
    );

// EVT with non-unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vec_nonUnitStride,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(55)
                ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3)
            ),
            // xi: index of extreme value for x.
            ::testing::Values(
                                gtint_t(1), gtint_t(27), gtint_t(51)
            ),
            // x_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // alpha: value of scalar.
            ::testing::Values(
                                double(-3.3),
                                double(-1.0),
                                double( 0.0),
                                double( 1.0),
                                double( 7.3)
            )
        ),
        ::dscalv_EVTPrint()
    );

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
        alpha_unitStride_zen3,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(565),
                                gtint_t(556),
                                gtint_t(529)
                ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(1) ),
            // x_exval: extreme value for x.
            ::testing::Values( double(0.0) ),
            // alpha: value of scalar.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dscalv_EVTPrint()
    );

// EVT with alpha containing Infs/NaNs on a unit stride vector.
INSTANTIATE_TEST_SUITE_P(
        alpha_unitStride_zen4,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,
                              'c' // conjugate option is BLIS-api specific.
#endif
            ),
            // m: size of vector.
            ::testing::Values( gtint_t(383) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(1) ),
            // x_exval: extreme value for x.
            ::testing::Values( double(0.0) ),
            // alpha: value of scalar.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dscalv_EVTPrint()
    );

// EVT with alpha containing Infs/NaNs on a non-unit stride vector.
INSTANTIATE_TEST_SUITE_P(
        alpha_nonUnitStride,
        dscalv_EVT,
        ::testing::Combine(
            // conj(alpha): uses n (no_conjugate) since it is real.
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
            ::testing::Values( gtint_t(1) ),
            // x_exval: extreme value for x.
            ::testing::Values( double(0.0) ),
            // alpha: value of scalar.
            ::testing::Values( NaN, Inf, -Inf )
        ),
        ::dscalv_EVTPrint()
    );
