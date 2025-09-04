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
#include "level2/gemv/test_gemv.h"

using  T = dcomplex;
using RT = testinghelpers::type_info<T>::real_type;
static RT AOCL_NaN = std::numeric_limits<RT>::quiet_NaN();
static RT AOCL_Inf = std::numeric_limits<RT>::infinity();

class zgemvEVT :
        public ::testing::TestWithParam<std::tuple<char,            // storage format
                                                   char,            // transa
                                                   char,            // conjx
                                                   gtint_t,         // m
                                                   gtint_t,         // n
                                                   T,               // alpha
                                                   T,               // beta
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   T,               // a_exval
                                                   T,               // x_exval
                                                   T,               // y_exval
                                                   gtint_t>> {};    // lda_inc

TEST_P( zgemvEVT, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // specifies beta value
    T beta = std::get<6>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<7>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<8>(GetParam());
    // exception value for a:
    T a_exval = std::get<9>(GetParam());
    // exception value for x:
    T x_exval = std::get<10>(GetParam());
    // exception value for y:
    T y_exval = std::get<11>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<12>(GetParam());

    bool is_memory_test = false;
    bool is_evt_test = true;

    // Set the threshold for the errors:
    // Check gtestsuite gemv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() && (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        if(( transa == 'n' ) || ( transa == 'N' ))
            thresh = (3*n+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = (3*m+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemv<T>( storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test, is_evt_test, a_exval, x_exval, y_exval );
}

INSTANTIATE_TEST_SUITE_P(
        matrix_vector_unitStride,
        zgemvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                      // storage format
            ::testing::Values('n','t'),             // transa
            ::testing::Values('n'),                 // conjx
            ::testing::Values(gtint_t(15)),         // m
            ::testing::Values(gtint_t(32)),         // n
            ::testing::Values(T{ 0.0,  0.0},
                              T{ 1.0,  1.0},
                              T{ 2.1, -1.2},
                              T{-1.0,  0.0},
                              T{ 1.0,  0.0}),       // alpha
            ::testing::Values(T{ 0.0,  0.0},
                              T{ 1.0,  1.0},
                              T{ 2.1, -1.2},
                              T{-1.0,  0.0},
                              T{ 1.0,  0.0}),       // beta
            ::testing::Values(gtint_t(1)),          // stride size for x
            ::testing::Values(gtint_t(1)),          // stride size for y
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // a_exval
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // x_exval
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // y_exval
            ::testing::Values(gtint_t(0))           // increment to the leading dim of a
        ),
        ::gemvEVTPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        matrix_vector_nonUnitStride,
        zgemvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                      // storage format
            ::testing::Values('n','t'),             // transa
            ::testing::Values('n'),                 // conjx
            ::testing::Values(gtint_t(55)),         // m
            ::testing::Values(gtint_t(55)),         // n
            ::testing::Values(T{ 0.0,  0.0},
                              T{ 1.0,  1.0},
                              T{ 2.1, -1.2},
                              T{-1.0,  0.0},
                              T{ 1.0,  0.0}),       // alpha
            ::testing::Values(T{ 0.0,  0.0},
                              T{ 1.0,  1.0},
                              T{ 2.1, -1.2},
                              T{-1.0,  0.0},
                              T{ 1.0,  0.0}),       // beta
            ::testing::Values(gtint_t(3)),          // stride size for x
            ::testing::Values(gtint_t(5)),          // stride size for y
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // a_exval
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // x_exval
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // y_exval
            ::testing::Values(gtint_t(7))           // increment to the leading dim of a
        ),
        ::gemvEVTPrint<T>()
    );


INSTANTIATE_TEST_SUITE_P(
        alpha_beta_unitStride,
        zgemvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                      // storage format
            ::testing::Values('n','t'),             // transa
            ::testing::Values('n'),                 // conjx
            ::testing::Values(gtint_t(32),
                              gtint_t(1)),         // m
            ::testing::Values(gtint_t(32),
                              gtint_t(1)),         // n
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // alpha
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // beta
            ::testing::Values(gtint_t(1)),          // stride size for x
            ::testing::Values(gtint_t(1)),          // stride size for y
            ::testing::Values(T{0.0, 0.0}),         // a_exval
            ::testing::Values(T{0.0, 0.0}),         // x_exval
            ::testing::Values(T{0.0, 0.0}),         // y_exval
            ::testing::Values(gtint_t(0))           // increment to the leading dim of a
        ),
        ::gemvEVTPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        alpha_beta_nonUnitStride,
        zgemvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                      // storage format
            ::testing::Values('n','t'),             // transa
            ::testing::Values('n'),                 // conjx
            ::testing::Values(gtint_t(55)),         // m
            ::testing::Values(gtint_t(55)),         // n
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // alpha
            ::testing::Values(T{AOCL_NaN,  AOCL_NaN},
                              T{AOCL_Inf, -AOCL_Inf},
                              T{AOCL_NaN,  AOCL_Inf},
                              T{2.1,  AOCL_Inf},
                              T{AOCL_Inf, -1.2},
                              T{AOCL_Inf,  0.0},
                              T{0.0,  AOCL_Inf},
                              T{0.0,  0.0}),        // beta
            ::testing::Values(gtint_t(3)),          // stride size for x
            ::testing::Values(gtint_t(5)),          // stride size for y
            ::testing::Values(T{0.0, 0.0}),         // a_exval
            ::testing::Values(T{0.0, 0.0}),         // x_exval
            ::testing::Values(T{0.0, 0.0}),         // y_exval
            ::testing::Values(gtint_t(7))           // increment to the leading dim of a
        ),
        ::gemvEVTPrint<T>()
    );
