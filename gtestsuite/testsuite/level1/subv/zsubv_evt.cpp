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

class zsubvEVT :
        public ::testing::TestWithParam<std::tuple<char,           // x or conj(x)
                                                   gtint_t,        // vector length
                                                   gtint_t,        // stride size of x
                                                   gtint_t,        // stride size of y
                                                   gtint_t,        // xi, index for exval in x
                                                   dcomplex,       // xexval
                                                   gtint_t,        // yi, index for exval in y
                                                   dcomplex>> {};  // yexval

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zsubvEVT);

TEST_P( zsubvEVT, API )
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
    // index for exval in x
    gtint_t xi = std::get<4>(GetParam());
    // exval for x
    T xexval = std::get<5>(GetParam());
    // index for exval in y
    gtint_t yj = std::get<6>(GetParam());
    // exval for y
    T yexval = std::get<7>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite subv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_subv<T>( conj_x, n, incx, incy, xi, xexval,
                   yj, yexval, thresh );
}

#ifdef TEST_BLIS_TYPED

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Exception value testing(on X vector alone) with unit strides on zen3
INSTANTIATE_TEST_SUITE_P(
        vecX_unitStrides,
        zsubvEVT,
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
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{NaN, -Inf}),
            // index on y
            ::testing::Values(gtint_t(0)),
            // value on y
            ::testing::Values(dcomplex{0.0, 0.0})
        ),
        ::subvEVTPrint<dcomplex>()
    );

// Exception value testing(on Y vector alone) with unit strides on zen3
INSTANTIATE_TEST_SUITE_P(
        vecY_unitStrides,
        zsubvEVT,
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
            ::testing::Values(dcomplex{0.0, 0.0}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on y
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{NaN, -Inf})
        ),
        ::subvEVTPrint<dcomplex>()
    );

// Exception value testing(on X and Y vectors) with unit strides on zen3
INSTANTIATE_TEST_SUITE_P(
        vecXY_unitStrides,
        zsubvEVT,
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
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{NaN, -Inf}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(7),
                          gtint_t(19), gtint_t(27), gtint_t(38),
                          gtint_t(69), gtint_t(99)),
            // exception values to set on y
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{NaN, -Inf})
        ),
        ::subvEVTPrint<dcomplex>()
    );

// Exception value testing(on X & Y vectors) with non-unit strides.
// The indices are such that we cover _vecX_, _vecY_ and _vecXY_ cases together.
INSTANTIATE_TEST_SUITE_P(
        vecXY_nonUnitStrides,
        zsubvEVT,
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
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{0.0, 0.0}, dcomplex{NaN, -Inf}),
            // indices to set exception values on y
            ::testing::Values(gtint_t(0), gtint_t(26), gtint_t(49)),
            // exception values to set on y
            ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{0.0, 0.0}, dcomplex{NaN, -Inf})
        ),
        ::subvEVTPrint<dcomplex>()
    );
#endif
