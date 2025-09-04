/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level2/ger/test_ger.h"

class sgerGeneric :
        public ::testing::TestWithParam<std::tuple<char,            // storage
                                                   char,            // conjx
                                                   char,            // conjy
                                                   gtint_t,         // m
                                                   gtint_t,         // n
                                                   float,           // alpha
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   gtint_t>> {};	// lda_inc

TEST_P( sgerGeneric, API )
{
    using T = float;
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
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_ger<T>( storage, conjx, conjy, m, n, alpha, incx, incy, lda_inc, thresh );
#endif
}

INSTANTIATE_TEST_SUITE_P(
        unitPositiveIncrement,
        sgerGeneric,
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
            ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(13), gtint_t(39), gtint_t(100) ),
            // n
            ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(49), gtint_t(76), gtint_t(100) ),
            // alpha: value of scalar
            ::testing::Values( float(-4.1), float(1.0), float(2.3) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(1) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0), gtint_t(3) )
        ),
        ::gerGenericPrint<float>()
    );

INSTANTIATE_TEST_SUITE_P(
        nonUnitPositiveIncrements,
        sgerGeneric,
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
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // n
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // alpha: value of scalar
            ::testing::Values( float(-4.1), float(1.0), float(2.3) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(2) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(3) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0), gtint_t(3) )
        ),
        ::gerGenericPrint<float>()
    );

// @note negativeIncrement tests are resulting in Segmentation Faults when
//  BLIS_TYPED interface is being tested.
#ifndef TEST_BLIS_TYPED
INSTANTIATE_TEST_SUITE_P(
        negativeIncrements,
        sgerGeneric,
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
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // n
            ::testing::Values( gtint_t(3), gtint_t(30), gtint_t(112) ),
            // alpha: value of scalar
            ::testing::Values( float(-4.1), float(1.0), float(2.3) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(-2) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(-3) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0), gtint_t(3) )
        ),
        ::gerGenericPrint<float>()
    );
#endif

INSTANTIATE_TEST_SUITE_P(
        LargeSize,
        sgerGeneric,
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
            ::testing::Values( gtint_t(5000) ),
            // n
            ::testing::Values( gtint_t(4000) ),
            // alpha: value of scalar
            ::testing::Values( float(3.4) ),
            // incx: stride of x vector.
            ::testing::Values( gtint_t(2), gtint_t(1) ),
            // incy: stride of y vector.
            ::testing::Values( gtint_t(3), gtint_t(1) ),
            // inc_lda: increment to the leading dim of a
            ::testing::Values( gtint_t(0), gtint_t(3) )
        ),
        ::gerGenericPrint<float>()
    );
