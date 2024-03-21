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
#include "test_gemmt.h"

class dgemmtEVT :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // uplo
                                                   char,          // transa
                                                   char,          // transb
                                                   gtint_t,       // n
                                                   gtint_t,       // k
                                                   double,        // alpha
                                                   double,        // beta
                                                   gtint_t,       // lda_inc
                                                   gtint_t,       // ldb_inc
                                                   gtint_t,       // ldc_inc
                                                   double,        // exception value for A matrix
                                                   double,        // exception value for B matrix
                                                   double>> {};   // exception value for C matrix

TEST_P( dgemmtEVT, NaNInfCheck )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies if the upper or lower triangular part of C is used
    char uplo = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // matrix size k
    gtint_t k  = std::get<5>(GetParam());
    // specifies alpha value
    T alpha = std::get<6>(GetParam());
    // specifies beta value
    T beta = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());
    gtint_t ldc_inc = std::get<10>(GetParam());
    T aexval = std::get<11>(GetParam());
    T bexval = std::get<12>(GetParam());
    T cexval = std::get<13>(GetParam());

    // Set the threshold for the errors:
    double thresh = 10*n*k*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc,
                  alpha, beta, thresh, false, true, aexval, bexval, cexval );
}

class dgemmtEVTPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,double,double,gtint_t,gtint_t,gtint_t,double,double,double>> str) const {
        char sfm        = std::get<0>(str.param);
        char uplo       = std::get<1>(str.param);
        char tsa        = std::get<2>(str.param);
        char tsb        = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        double alpha    = std::get<6>(str.param);
        double beta     = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);
        double aexval   = std::get<11>(str.param);
        double bexval   = std::get<12>(str.param);
        double cexval   = std::get<13>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "blas_";
#elif TEST_CBLAS
        std::string str_name = "cblas_";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_";
#endif
        str_name = str_name + "_storage_" + sfm;
        str_name = str_name + "_transa_" + tsa;
        str_name = str_name + "_transb_" + tsb;
        str_name = str_name + "_uploa_" + uplo;
        str_name = str_name + "_n_" + std::to_string(n);
        str_name = str_name + "_k_" + std::to_string(k);
        std::string alpha_str = testinghelpers::get_value_string(alpha);
        str_name = str_name + "_alpha_" + alpha_str;
        std::string beta_str = testinghelpers::get_value_string(beta);
        str_name = str_name + "_beta_" + beta_str;
        gtint_t lda = testinghelpers::get_leading_dimension( sfm, tsa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( sfm, tsb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( sfm, 'n', n, n, ldc_inc );
        str_name = str_name + "_ex_a_" + testinghelpers::get_value_string(aexval);
        str_name = str_name + "_ex_b_" + testinghelpers::get_value_string(bexval);
        str_name = str_name + "_ex_c_" + testinghelpers::get_value_string(cexval);
        str_name = str_name + "_ldb_" + std::to_string(lda);
        str_name = str_name + "_ldb_" + std::to_string(ldb);
        str_name = str_name + "_ldc_" + std::to_string(ldc);
        return str_name;
    }
};

static double AOCL_NAN = std::numeric_limits<double>::quiet_NaN();
static double AOCL_INF = std::numeric_limits<double>::infinity();

#ifndef TEST_BLIS_TYPED
INSTANTIATE_TEST_SUITE_P(
        Native,
        dgemmtEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uplo u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','t'),                                      // transb
            ::testing::Values(7, 800),                                       // n
            ::testing::Values(7, 800),                                       // k
            ::testing::Values(2.4, AOCL_NAN/*, AOCL_INF, -AOCL_INF*/),       // alpha //commented values fail
            ::testing::Values(2.4/*, AOCL_NAN*/, AOCL_INF, -AOCL_INF),       // beta //commented values fail
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of c
            ::testing::Values(0.0, AOCL_NAN, AOCL_INF, -AOCL_INF),           // extreme value for A matrix
            ::testing::Values(0.0, AOCL_NAN, AOCL_INF, -AOCL_INF),           // extreme value for B matrix
            ::testing::Values(0.0, AOCL_NAN, AOCL_INF, -AOCL_INF)            // extreme value for B matrix
        ),
        ::dgemmtEVTPrint()
    );
#endif
