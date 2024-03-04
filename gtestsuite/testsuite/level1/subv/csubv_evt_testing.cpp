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
#include "test_subv.h"

class csubvEVT :
        public ::testing::TestWithParam<std::tuple<char,           // x or conj(x)
                                                   gtint_t,        // vector length
                                                   gtint_t,        // stride size of x
                                                   gtint_t,        // stride size of y
                                                   gtint_t,        // xi, index for exval in x
                                                   scomplex,       // xexval
                                                   gtint_t,        // yi, index for exval in y
                                                   scomplex>> {};  // yexval

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(csubvEVT);

TEST_P( csubvEVT, NaNInfCheck )
{
    using T = scomplex;
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
    // index for exval in x
    gtint_t xi = std::get<4>(GetParam());
    // exval for x
    T xexval = std::get<5>(GetParam());
    // index for exval in y
    gtint_t yj = std::get<6>(GetParam());
    // exval for y
    T yexval = std::get<7>(GetParam());

    // Set the threshold for the errors:
    double thresh = 20 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_subv<T>( conj_x, n, incx, incy, xi, xexval,
                   yj, yexval, thresh );
}

// Test-case logger : Used to print the test-case details when vectors have exception value.
// The string format is as follows :
// n(vec_size)_(conjx/noconjx)_incx(m)(abs_incx)_incy(m)(abs_incy)_X_(xi)_(xexval)_(yi)_(yexval)
class csubvEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, gtint_t, gtint_t, gtint_t, gtint_t, scomplex, gtint_t, scomplex>> str) const {
        char conjx      = std::get<0>(str.param);
        gtint_t n      = std::get<1>(str.param);
        gtint_t incx   = std::get<2>(str.param);
        gtint_t incy   = std::get<3>(str.param);
        gtint_t xi = std::get<4>(str.param);
        scomplex xexval = std::get<5>(str.param);
        gtint_t yj = std::get<6>(str.param);
        scomplex yexval = std::get<7>(str.param);
        std::string str_name = "bli_";
        str_name += "n_" + std::to_string(n);
        str_name += ( conjx == 'n' )? "_noconjx" : "_conjx";
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx_" + incx_str;
        std::string incy_str = ( incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_incy_" + incy_str;
        std::string xexval_str = testinghelpers::get_value_string(xexval);
        std::string yexval_str = testinghelpers::get_value_string(yexval);
        str_name = str_name + "_X_" + std::to_string(xi);
        str_name = str_name + "_" + xexval_str;
        str_name = str_name + "_Y_" + std::to_string(yj);
        str_name = str_name + "_" + yexval_str;
        return str_name;
    }
};

static float NaN = std::numeric_limits<double>::quiet_NaN();
static float Inf = std::numeric_limits<double>::infinity();

#ifdef TEST_BLIS_TYPED
// Exception value testing(on X vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
        vecX_unitStrides,
        csubvEVT,
        ::testing::Combine(
            // n: use x, c: use conj(x)
            ::testing::Values('n','c'),
            // n: size of vector.
            // as we don't have BLIS vectorized kernels for subv,
            // having fewer sizes or maybe a Range would be sufficient
            // to ensure code coverage of the reference kernel.
            ::testing::Values(
                gtint_t(100)),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1)),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1)),
            // indices to set exception values on x
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on x
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{NaN, -Inf}),
            // index on y
            ::testing::Values(gtint_t(0)),
            // value on y
            ::testing::Values(scomplex{0.0, 0.0})
        ),
        ::csubvEVTPrint()
    );

// Exception value testing(on Y vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
        vecY_unitStrides,
        csubvEVT,
        ::testing::Combine(
            // n: use x, c: use conj(x)
            ::testing::Values('n','c'),
            // n: size of vector.
            // as we don't have BLIS vectorized kernels for subv,
            // having fewer sizes or maybe a Range would be sufficient
            // to ensure code coverage of the reference kernel.
            ::testing::Values(
                gtint_t(100)),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1)),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1)),
            // index on x
            ::testing::Values(gtint_t(0)),
            // value on x
            ::testing::Values(scomplex{0.0, 0.0}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on y
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{NaN, -Inf})
        ),
        ::csubvEVTPrint()
    );

// Exception value testing(on X and Y vectors) with unit strides
INSTANTIATE_TEST_SUITE_P(
        vecXY_unitStrides,
        csubvEVT,
        ::testing::Combine(
            // n: use x, c: use conj(x)
            ::testing::Values('n','c'),
            // n: size of vector.
            // as we don't have BLIS vectorized kernels for subv,
            // having fewer sizes or maybe a Range would be sufficient
            // to ensure code coverage of the reference kernel.
            ::testing::Values(
                gtint_t(100)),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1)),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1)),
            // indices to set exception values on x
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on x
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{NaN, -Inf}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on y
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{NaN, -Inf})
        ),
        ::csubvEVTPrint()
    );

// Exception value testing(on X & Y vectors) with non-unit strides.
// The indices are such that we cover _vecX_, _vecY_ and _vecXY_ cases together.
INSTANTIATE_TEST_SUITE_P(
        vecXY_nonUnitStrides,
        csubvEVT,
        ::testing::Combine(
            // n: use x, c: use conj(x)
            ::testing::Values('n','c'),
            // n: size of vector.
            // as we don't have BLIS vectorized kernels for subv,
            // having fewer sizes or maybe a Range would be sufficient
            // to ensure code coverage of the reference kernel.
            ::testing::Values(
                gtint_t(50)),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(3)),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(5)),
            // indices to set exception values on x
            ::testing::Values(gtint_t(1), gtint_t(27), gtint_t(49)),
            // exception values to set on x
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{0.0, 0.0}, scomplex{NaN, -Inf}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(26), gtint_t(49)),
            // exception values to set on y
            ::testing::Values(scomplex{NaN, 0.0}, scomplex{-Inf, 0.0},
                          scomplex{0.0, Inf}, scomplex{-2.3, NaN},
                          scomplex{4.5, -Inf}, scomplex{NaN, Inf},
                          scomplex{0.0, 0.0}, scomplex{NaN, -Inf})
        ),
        ::csubvEVTPrint()
    );
#endif
