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
#include "test_ger.h"

using  T = scomplex;
using RT = testinghelpers::type_info<T>::real_type;
static RT NaN = std::numeric_limits<RT>::quiet_NaN();
static RT Inf = std::numeric_limits<RT>::infinity();

class cgerEVT :
        public ::testing::TestWithParam<std::tuple<char,        // storage
                                                   char,        // conjx
                                                   char,        // conjy
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   T,           // alpha
                                                   gtint_t,     // incx
                                                   gtint_t,     // incy
                                                   gtint_t,     // lda_inc
                                                   gtint_t,     // ai
                                                   gtint_t,     // aj
                                                   T,           // a_exval
                                                   gtint_t,     // xi
                                                   T,           // x_exval
                                                   gtint_t,     // yi
                                                   T>> {};      // y_exval

TEST_P( cgerEVT, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<1>(GetParam());
    // denotes whether vector y is n,c
    char conjy = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<6>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<7>(GetParam());
    // lda increment:
    // If increment is zero, then the array size matches the matrix size.
    // If increment is non-negative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    // ai:
    gtint_t ai = std::get<9>(GetParam());
    // aj:
    gtint_t aj = std::get<10>(GetParam());
    // a_exval:
    T a_exval = std::get<11>(GetParam());
    // xi:
    gtint_t xi = std::get<12>(GetParam());
    // x_exval:
    T x_exval = std::get<13>(GetParam());
    // yi:
    gtint_t yi = std::get<14>(GetParam());
    // y_exval:
    T y_exval = std::get<15>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite ger.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = 7*testinghelpers::getEpsilon<T>();


    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc,
                 ai, aj, a_exval, xi, x_exval, yi, y_exval, thresh );
}

INSTANTIATE_TEST_SUITE_P(
        unitStride,
        cgerEVT,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: uses n (no_conjugate) since it is real.
            ::testing::Values( 'n' ),
            // conjy: uses n (no_conjugate) since it is real.
            ::testing::Values( 'n' ),
            // m
            ::testing::Values( gtint_t(55) ),
            // n
            ::testing::Values( gtint_t(33) ),
            // alpha: value of scalar
            ::testing::Values( T{1.0, 1.0}, T{2.3, -1.2}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),
            // inc_lda: increment to the leading dim of a.
            ::testing::Values( gtint_t(0) ),
            // ai: index of extreme value for a.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // aj: index of extreme value for a.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // a_exval: extreme value for a.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // x_exval: extreme value for x.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // y_exval: extreme value for y.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} )
        ),
        ::gerEVTPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        nonUnitStrides,
        cgerEVT,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: uses n (no_conjugate) since it is real.
            ::testing::Values( 'n' ),
            // conjy: uses n (no_conjugate) since it is real.
            ::testing::Values( 'n' ),
            // m
            ::testing::Values( gtint_t(55) ),
            // n
            ::testing::Values( gtint_t(33) ),
            // alpha: value of scalar
            ::testing::Values( T{1.0, 1.0}, T{2.3, -1.2}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(3) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(5) ),
            // inc_lda: increment to the leading dim of a.
            ::testing::Values( gtint_t(7) ),
            // ai: index of extreme value for a.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // aj: index of extreme value for a.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // a_exval: extreme value for a.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // xi: index of extreme value for x.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // x_exval: extreme value for x.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} ),
            // yi: index of extreme value for y.
            ::testing::Values( gtint_t(0), gtint_t(7) ),
            // y_exval: extreme value for y.
            ::testing::Values( T{0.0, 0.0}, T{NaN, NaN}, T{NaN, Inf}, T{Inf, -Inf} )
        ),
        ::gerEVTPrint<scomplex>()
    );
