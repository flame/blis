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
#include "test_trsv.h"

class dtrsvEVT :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // uplo
                                                   char,          // trans
                                                   char,          // diag
                                                   gtint_t,       // n
                                                   double,        // alpha
                                                   gtint_t,       // incx
                                                   double,        // exception value for X
                                                   double,        // excepton value for Y
                                                   gtint_t>> {};  // ld_inc

TEST_P( dtrsvEVT, NaNInfCheck )
{
    using T = double;
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
    double xexval  = std::get<7>(GetParam());
    // extreme value for A
    double aexval  = std::get<8>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trsv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = 2*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_trsv<T>( storage, uploa, transa, diaga, n, alpha, lda_inc, incx, thresh, false, true, xexval, aexval);
}

class dtrsvEVTPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,double,gtint_t,double,double,gtint_t>> str) const {
        char sfm       = std::get<0>(str.param);
        char uploa     = std::get<1>(str.param);
        char transa    = std::get<2>(str.param);
        char diaga     = std::get<3>(str.param);
        gtint_t n      = std::get<4>(str.param);
        double alpha   = std::get<5>(str.param);
        gtint_t incx   = std::get<6>(str.param);
        double xexval  = std::get<7>(str.param);
        double aexval  = std::get<8>(str.param);
        gtint_t ld_inc = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "blas_";
#elif TEST_CBLAS
        std::string str_name = "cblas_";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_";
#endif
        str_name    = str_name + "stor_" + sfm;
        str_name    = str_name + "_uplo_" + uploa;
        str_name    = str_name + "_transa_" + transa;
        str_name    = str_name + "_diaga_" + diaga;
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name    = str_name + "_ex_x_" + testinghelpers::get_value_string(xexval);
        str_name    = str_name + "_ex_a_" + testinghelpers::get_value_string(aexval);
        str_name    = str_name + "_lda_" + std::to_string(
                    testinghelpers::get_leading_dimension( sfm, transa, n, n, ld_inc )
                );
        return str_name;
    }
};

static double AOCL_NAN = std::numeric_limits<double>::quiet_NaN();
static double AOCL_INF = std::numeric_limits<double>::infinity();

INSTANTIATE_TEST_SUITE_P(
        Native,
        dtrsvEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                                 // storage format
            ::testing::Values('u','l'),                                        // uploa
            ::testing::Values('n','t'),                                        // transa
            ::testing::Values('n','u'),                                        // diaga , n=NONUNIT_DIAG u=UNIT_DIAG
            ::testing::Values(gtint_t(32),
                              gtint_t(24),
                              gtint_t(8),
                              gtint_t(4),
                              gtint_t(2),
                              gtint_t(1),
                              gtint_t(15)
                            ),                                                 // n (random values)
            ::testing::Values( 1.0
#ifdef TEST_BLIS_TYPED
            , -2.2, 5.4, -1.0, 0.0
#endif
            ),                                                                 // alpha
            ::testing::Values(gtint_t(-2), gtint_t(-1),
                              gtint_t( 1), gtint_t( 2)),                       // stride size for x
            ::testing::Values(AOCL_NAN, -AOCL_INF, AOCL_INF, 1 /*,0  <-fail*/),// exception value for x
            ::testing::Values(AOCL_NAN, -AOCL_INF, AOCL_INF, 0),               // exception value for A
            ::testing::Values(gtint_t(0), gtint_t(10))                         // increment to the leading dim of a
        ),
        ::dtrsvEVTPrint()
    );
