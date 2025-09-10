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
#include "test_asumv.h"

class dasumvEVT :
        public ::testing::TestWithParam<std::tuple<gtint_t,     // n
                                                   gtint_t,     // incx
                                                   gtint_t,     // xi
                                                   double,      // ix_exval
                                                   gtint_t,     // xj
                                                   double>> {}; // jx_exval

TEST_P( dasumvEVT, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());
    // index of extreme value for x:
    gtint_t xi = std::get<2>(GetParam());
    // extreme value for x:
    double ix_exval = std::get<3>(GetParam());
    // index of extreme value for x:
    gtint_t xj = std::get<4>(GetParam());
    // extreme value for x:
    double jx_exval = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite asumv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0 || incx <= 0)
        thresh = 0.0;
    else
        thresh = n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_asumv<T>( n, incx, xi, ix_exval, xj, jx_exval, thresh );
}

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// EVT with unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vec_unitStride,
        dasumvEVT,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values( gtint_t(55) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // xi: first index to set extreme value in x.
            ::testing::Values( gtint_t(1), gtint_t(27), gtint_t(51) ),
            // ix_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // xj: second index to set extreme value in x.
            ::testing::Values( gtint_t(13) ),
            // jx_exval: extreme value for x.
            // jx_exval = 1.0 tests for the vector with only one extreme value.
            ::testing::Values( 1.0, NaN, Inf, -Inf )
        ),
        ::asumvEVTPrint<double>()
    );

// EVT with non-unit stride vector containing Infs/NaNs.
INSTANTIATE_TEST_SUITE_P(
        vec_nonUnitStride,
        dasumvEVT,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values( gtint_t(55) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(3) ),
            // xi: first index to set extreme value in x.
            ::testing::Values( gtint_t(1), gtint_t(27), gtint_t(51) ),
            // ix_exval: extreme value for x.
            ::testing::Values( NaN, Inf, -Inf ),
            // xj: second index to set extreme value in x.
            ::testing::Values( gtint_t(13) ),
            // jx_exval: extreme value for x.
            // jx_exval = 1.0 tests for the vector with only one extreme value.
            ::testing::Values( 1.0, NaN, Inf, -Inf )
        ),
        ::asumvEVTPrint<double>()
    );
