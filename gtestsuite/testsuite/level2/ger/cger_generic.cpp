/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

class cgerGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   scomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P( cgerGeneric, API )
{
    using T = scomplex;
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
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment is non-negative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite ger.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // With adjustment for complex data.
    double thresh;
    double adj = 3.0;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = adj*3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc, thresh );
}

INSTANTIATE_TEST_SUITE_P(
        unitPositiveIncrement,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // m
            ::testing::Range( gtint_t(10), gtint_t(101), 10 ),
            // n
            ::testing::Range( gtint_t(10), gtint_t(101), 10 ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-1.0, 4.0}, scomplex{1.0, 1.0}, scomplex{3.0, -2.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0) )
        ),
        ::gerGenericPrint<scomplex>()
    );

#ifdef TEST_BLIS_TYPED
// Test when conjugate of x is used as an argument. This option is BLIS-api specific.
// Only test very few cases as sanity check since conj(x) = x for real types.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        conjXY,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // m
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // n
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-1.0, 4.0}, scomplex{1.0, 1.0}, scomplex{3.0, -2.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(1) )
        ),
        ::gerGenericPrint<scomplex>()
    );
#endif

INSTANTIATE_TEST_SUITE_P(
        nonUnitPositiveIncrements,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // m
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // n
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-1.0, 4.0}, scomplex{1.0, 1.0}, scomplex{3.0, -2.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(2) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(3) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(5) )
        ),
        ::gerGenericPrint<scomplex>()
    );

// @note negativeIncrement tests are resulting in Segmentation Faults when
//  BLIS_TYPED interface is being tested.
#ifndef TEST_BLIS_TYPED
INSTANTIATE_TEST_SUITE_P(
        negativeIncrements,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'n', 'c' ),
            // m
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // n
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-1.0, 4.0}, scomplex{1.0, 1.0}, scomplex{3.0, -2.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(-2) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(-3) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0) )
        ),
        ::gerGenericPrint<scomplex>()
    );
#endif

INSTANTIATE_TEST_SUITE_P(
        scalarCombinations,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // m
            ::testing::Values( gtint_t(35) ),
            // n
            ::testing::Values( gtint_t(40) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-100.0, 200.0}, scomplex{200.0, 100.0}, scomplex{-175.0, -143.0},scomplex{187.0, -275.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(2) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(3) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(2) )
        ),
        ::gerGenericPrint<scomplex>()
    );
//large values of m and n
INSTANTIATE_TEST_SUITE_P(
        largeSize,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // m
            ::testing::Values( gtint_t(3500) ),
            // n
            ::testing::Values( gtint_t(4000) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-10.0, 8.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(2), gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(3), gtint_t(1) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(2) )
        ),
        ::gerGenericPrint<scomplex>()
    );
//Stride greater than m and n
INSTANTIATE_TEST_SUITE_P(
        strideGreaterThanSize,
        cgerGeneric,
        ::testing::Combine(
            // storage scheme: row/col-stored matrix
            ::testing::Values( 'c'
            // row-stored tests are disabled for BLAS since BLAS only supports col-storage scheme.
#ifndef TEST_BLAS_LIKE
                             , 'r'
#endif
            ),
            // conjx: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // conjy: use n for no_conjugate and c for conjugate.
            ::testing::Values( 'c' ),
            // m
            ::testing::Values( gtint_t(3) ),
            // n
            ::testing::Values( gtint_t(4) ),
            // alpha: value of scalar
            ::testing::Values( scomplex{-10.0, 8.0} ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(15) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(18) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(20) )
        ),
        ::gerGenericPrint<scomplex>()
    );

