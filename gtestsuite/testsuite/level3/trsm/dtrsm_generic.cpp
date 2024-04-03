/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-24, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_trsm.h"

class dtrsmTest :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // side
                                                   char,          // uplo
                                                   char,          // transa
                                                   char,          // diaga
                                                   gtint_t,       // m
                                                   gtint_t,       // n
                                                   double,        // alpha
                                                   gtint_t,       // lda_inc
                                                   gtint_t>> {};  // ldb_inc

TEST_P(dtrsmTest, Accuracy_test)
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies matrix A appears left or right in
    // the matrix multiplication
    char side = std::get<1>(GetParam());
    // specifies upper or lower triangular part of A is used
    char uploa = std::get<2>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<3>(GetParam());
    // denotes whether matrix a in unit or non-unit diagonal
    char diaga = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // specifies alpha value
    T alpha = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    double thresh = 1.5*(std::max)(m, n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
}

class dtrsmTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, char, char, gtint_t, gtint_t, double, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char side       = std::get<1>(str.param);
        char uploa      = std::get<2>(str.param);
        char transa     = std::get<3>(str.param);
        char diaga      = std::get<4>(str.param);
        gtint_t m       = std::get<5>(str.param);
        gtint_t n       = std::get<6>(str.param);
        double alpha    = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dtrsm_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dtrsm";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_dtrsm";
#endif
        str_name = str_name + "_" + sfm+sfm+sfm;
        str_name = str_name + "_" + side + uploa + transa;
        str_name = str_name + "_d" + diaga;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(n);
        std::string alpha_str = isnan( alpha ) ? "NaN" : isinf( alpha ) ? "Inf" : ( alpha > 0) ? std::to_string(int(alpha)) : "m" + std::to_string(int(std::abs(alpha)));
        str_name = str_name + "_a" + alpha_str;
        str_name = str_name + "_" + std::to_string(lda_inc);
        str_name = str_name + "_" + std::to_string(ldb_inc);
        return str_name;
    }
};

/**
 * @brief Test DTRSM native path, which starts from size 1500 for BLAS api
 *        and starts from size 0 for BLIS api.
 */
INSTANTIATE_TEST_SUITE_P(
        Native,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 2, 112, 1551),                              // m
            ::testing::Values(1, 2, 154, 1676),                              // n
            ::testing::Values(-2.4),                                         // alpha
            ::testing::Values(gtint_t(5)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(3))                                    // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );

/**
 * @brief Test DTRSM small avx2 path all fringe cases
 *        Kernel size for avx2 small path is 6x8, testing in range of
 *        1 to 8 ensures all finge cases are being tested.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2_fringe,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                     // m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                     // n
            ::testing::Values(-2.4),                                         // alpha
            ::testing::Values(gtint_t(5)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(3))                                    // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );

/**
 * @brief Test DTRSM small avx2 path which is used in
 *        range [0, 50] for genoa and [0, 1499] for milan
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 110, 51, 1499),                           // m
            ::testing::Values(17, 48 , 51, 1499),                           // n
            ::testing::Values(-2.4),                                         // alpha
            ::testing::Values(gtint_t(5)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(3))                                    // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );

/**
 * @brief Test DTRSM small avx512 path all fringe cases
 *        small avx512 is used in range [51, 1499]
 *        Kernel size for avx512 small path is 8x8, therefore
 *        testing in range of 51 to 58 covers all fringe cases.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX512_fringe,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(51), gtint_t(59), 1),                   // m
            ::testing::Range(gtint_t(51), gtint_t(59), 1),                   // n
            ::testing::Values(-2.4),                                         // alpha
            ::testing::Values(gtint_t(5)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(3))                                    // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );

/**
 * @brief Test DTRSM small avx512 path
 *        small avx512 is used in range [51, 1499]
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX512,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(51, 410, 1499),                                // n
            ::testing::Values(51, 531, 1499),                                // m
            ::testing::Values(-2.4),                                         // alpha
            ::testing::Values(gtint_t(5)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(3))                                    // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );

/**
 * @brief Test DTRSM with differnt values of alpha
 *      code paths covered:
 *          TRSV              -> 1
 *          TRSM_AVX2_small   -> 2
 *          TRSM_AVX512_small -> 300
 *          TRSM_NATIVE       -> 1500
 */
INSTANTIATE_TEST_SUITE_P(
        Alpha,
        dtrsmTest,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 2, 300, 1500),                              // n
            ::testing::Values(1, 2, 300, 1500),                              // m
            ::testing::Values(-2.4, 0.0, 1.0, 3.1, NAN, INFINITY),           // alpha
            ::testing::Values(gtint_t(0), gtint_t(5)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(3))                        // increment to the leading dim of b
        ),
        ::dtrsmTestPrint()
    );
