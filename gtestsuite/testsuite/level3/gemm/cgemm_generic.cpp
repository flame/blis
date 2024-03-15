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
#include "test_gemm.h"
class cgemmAPI :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   scomplex,   // alpha
                                                   scomplex,   // beta
                                                   gtint_t,    // inc to the lda 
                                                   gtint_t,    // inc to the ldb 
                                                   gtint_t     // inc to the ldc
                                                   >> {};
TEST_P(cgemmAPI, FunctionalTest)
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
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
    // Set the threshold for the errors:
    double thresh = 10*m*n*testinghelpers::getEpsilon<T>();
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}
class cgemmPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, scomplex, scomplex, gtint_t, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char tsa        = std::get<1>(str.param);
        char tsb        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        scomplex alpha  = std::get<6>(str.param);
        scomplex beta   = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "blas_";
#elif TEST_CBLAS
        std::string str_name = "cblas_";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_";
#endif
        str_name = str_name + "storageOfMatrix_" + sfm;
        str_name = str_name + "_transA_" + tsa + "_transB_" + tsb;
        str_name = str_name + "_m_" + std::to_string(m);
        str_name = str_name + "_n_" + std::to_string(n);
        str_name = str_name + "_k_" + std::to_string(k);
        std::string alpha_str = (alpha.real < 0) ? ("m" + std::to_string(int(std::abs(alpha.real)))) : std::to_string(int(alpha.real));
        alpha_str = alpha_str + ((alpha.imag < 0) ? ("m" + std::to_string(int(std::abs(alpha.imag)))) : "i" + std::to_string(int(alpha.imag)));
        std::string beta_str = (beta.real < 0) ? ("m" + std::to_string(int(std::abs(beta.real)))) : std::to_string(int(beta.real));
        beta_str = beta_str + ((beta.imag < 0) ?  ("m" + std::to_string(int(std::abs(beta.imag)))) : "i" + std::to_string(int(beta.imag)));
        str_name = str_name + "_alpha_" + alpha_str;
        str_name = str_name + "_beta_" + beta_str;
        gtint_t lda = testinghelpers::get_leading_dimension( sfm, tsa, m, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( sfm, tsb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( sfm, 'n', m, n, ldc_inc );
        str_name = str_name + "_lda_" + std::to_string(lda);
        str_name = str_name + "_ldb_" + std::to_string(ldb);
        str_name = str_name + "_ldc_" + std::to_string(ldc);
        return str_name;
    }
};

/********************************************************************/
/* Testing SUP and Native implementation of cgemm API               */
/********************************************************************/
/************************** SCALM************************************/
/* Scaling of C matrix for below conditions                         */
/* 1. When alpha is zero                                            */
/* 2. When Matrix A or Matrix B has zero dimension                  */
/* Scale Matrix C by Beta and return                                */
/********************************************************************/
/************************** SUP *************************************/
/* Current SUP implmentation does not support below parameters      */
/* 1. General Stride                                                */
/* 2. Conjugate                                                     */
/* 3. Input dimensions greater than below thresholds                */
/*    m > 380 ||  n > 256 || k > 220                                */
/* SUP implementations is suitable for Skinny Matrices              */
/* List of API's:                                                   */
/*  1. bli_cgemmsup_rv_zen_asm_3x8m: M preferred kernel             */
/*  2. bli_cgemmsup_rv_zen_asm_3x8n: N preferred kernel             */
/********************************************************************/
/************************** NATIVE***********************************/
/*  When SUP method does not support given input arguments,         */
/*  Native implmentation will be invoked, it is well suited for     */
/*  square, large sizes                                             */
/* API Name: bli_cgemm_haswell_asm_3x8                              */
/********************************************************************/

INSTANTIATE_TEST_SUITE_P(
        Alpha_zero,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(300), gtint_t(32), gtint_t(17)),      // m
            ::testing::Values(gtint_t(200), gtint_t(22), gtint_t(18)),      // n
            ::testing::Values(gtint_t(150), gtint_t(16), gtint_t(19)),      // k
            ::testing::Values(scomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{12.9, 12.3}, scomplex{0.0, 1.9},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.2, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(2344)),                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(9185)),                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(4367))                    // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Matrix_Dimension_zero,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(0), gtint_t(12)),                     // m
            ::testing::Values(gtint_t(0), gtint_t(12)),                     // n
            ::testing::Values(gtint_t(0), gtint_t(16)),                     // k
            ::testing::Values(scomplex{1.2, 0.8}),                          // alpha
            ::testing::Values(scomplex{12.9, 12.3}, scomplex{0.0, 1.9},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.2, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(2344)),                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(9185)),                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(4367))                    // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Range(gtint_t(300), gtint_t(320), gtint_t(1)),       // m
            ::testing::Range(gtint_t(200), gtint_t(220), gtint_t(1)),       // n
            ::testing::Range(gtint_t(150), gtint_t(160), gtint_t(1)),       // k
            ::testing::Values(scomplex{-1.0, -2.0}),                        // alpha
            ::testing::Values(scomplex{12.0, 2.3}),                         // beta
            ::testing::Values(gtint_t(0), gtint_t(2344)),                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(9185)),                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(4367))                    // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Alpha_Beta,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Range(gtint_t(300), gtint_t(304), gtint_t(1)),       // m
            ::testing::Range(gtint_t(200), gtint_t(209), gtint_t(1)),       // n
            ::testing::Values(gtint_t(150)),                                // k
            ::testing::Values(scomplex{10.0, 20.0}, scomplex{0.0, -30.0},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{12.0, 2.3}, scomplex{0.0, 1.3},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{5.0, 0.0}, scomplex{0.0, 0.0}),      // beta
            ::testing::Values(gtint_t(0), gtint_t(4567)),                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(7654)),                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(4321))                    // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Range(gtint_t(400), gtint_t(700), gtint_t(150)),     // m
            ::testing::Range(gtint_t(380), gtint_t(1000), gtint_t(200)),    // n
            ::testing::Values(gtint_t(270), gtint_t(280), gtint_t(1)),      // k
            ::testing::Values(scomplex{1.5, 3.5}),                          // alpha
            ::testing::Values(scomplex{2.0, 4.1}),                          // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Alpha_Beta,
        cgemmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Range(gtint_t(400), gtint_t(700), gtint_t(150)),     // m
            ::testing::Range(gtint_t(380), gtint_t(1000), gtint_t(200)),    // n
            ::testing::Values(gtint_t(270)),                                // k
            ::testing::Values(scomplex{11.5, -3.5}, scomplex{0.0, -10.0},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{2.0, 0.0}),                          // alpha
            ::testing::Values(scomplex{12.0, -4.1}, scomplex{0.0, 3.4},
                              scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{3.3, 0.0}, scomplex{0.0, 0.0}),      // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );
