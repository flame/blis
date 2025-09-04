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
#include "level2/trsv/test_trsv.h"

class DISABLED_ztrsvEVT :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // uplo
                                                   char,          // trans
                                                   char,          // diag
                                                   gtint_t,       // n
                                                   dcomplex,      // alpha
                                                   gtint_t,       // incx
                                                   dcomplex,      // exception value for X
                                                   dcomplex,      // excepton value for Y
                                                   gtint_t>> {};  // ld_inc

TEST_P( DISABLED_ztrsvEVT, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is u,l
    char uploa = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix diag is u,n
    char diaga = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<6>(GetParam());
    // extreme value for x
    dcomplex xexval  = std::get<7>(GetParam());
    // extreme value for A
    dcomplex aexval  = std::get<8>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trsv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0 || alpha == T{0.0})
        thresh = 0.0;
    else
        if(alpha == T{1.0})
          thresh = 2*n*testinghelpers::getEpsilon<T>();
        else
          thresh = 3*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_trsv<T>( storage, uploa, transa, diaga, n, alpha, lda_inc, incx, thresh, false, true, xexval, aexval);
}

static double AOCL_NAN = std::numeric_limits<double>::quiet_NaN();
static double AOCL_INF = std::numeric_limits<double>::infinity();

INSTANTIATE_TEST_SUITE_P(
        Native,
        DISABLED_ztrsvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=NONUNIT_DIAG u=UNIT_DIAG
            ::testing::Values(gtint_t(32),
                              gtint_t(24),
                              gtint_t(8),
                              gtint_t(4),
                              gtint_t(2),
                              gtint_t(1),
                              gtint_t(15)
                            ),                                               // n
            ::testing::Values(dcomplex{1.0, 0.0}
#ifdef TEST_BLIS_TYPED
            ,dcomplex{6.1, -2.9}, dcomplex{-3.3, -1.4},
             dcomplex{-1.0, 0.0}, dcomplex{0.0, 0.0}
#endif
            ),                                                               // alpha
            ::testing::Values(gtint_t(-2), gtint_t(-1),
                              gtint_t( 1), gtint_t( 2)),                     // stride size for x
            ::testing::Values(
                              dcomplex{AOCL_NAN, 2.1},
                              dcomplex{2.1, AOCL_NAN},
                              dcomplex{AOCL_NAN, AOCL_INF},
                              // dcomplex{2.3, AOCL_INF},                    // fail
                              // dcomplex{AOCL_INF, 2.3},                    // fail
                              // dcomplex{0.0, AOCL_INF},                    // fail
                              // dcomplex{AOCL_INF, 0.0},                    // fail
                              // dcomplex{0.0, -AOCL_INF},                   // fail
                              // dcomplex{-AOCL_INF, 0.0},                   // fail
                              dcomplex{1, 0} ),                              // exception value for x
            ::testing::Values(
                              dcomplex{AOCL_NAN, 3.2},
                              dcomplex{2.1, AOCL_NAN},
                              dcomplex{AOCL_NAN, AOCL_INF},
                              // dcomplex{2.3, AOCL_INF},                    // fail
                              // dcomplex{AOCL_INF, 6.1},                    // fail
                              dcomplex{1, 0}),                               // exception value for A
            ::testing::Values(gtint_t(0), gtint_t(10))                       // increment to the leading dim of a
        ),
        ::trsvEVTPrint<dcomplex>()
    );
