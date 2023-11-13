/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

/*
    The following file contains both the exception value testing(EVT) and the
    positive accuracy testing of the bli_zgemm_4x4_avx2_k1_nn( ... ) computational
    kernel. This kernel is invoked from the BLAS layer, and inputs are given
    in a manner so as to avoid the other code-paths and test only the required
    kernel.

*/

#include <gtest/gtest.h>
#include "test_gemm.h"

class ZGemmEVTTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   dcomplex,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P(ZGemmEVTTest, Unit_Tester)
{
    using T = dcomplex;
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

    gtint_t ai, aj, bi, bj, ci, cj;
    T aex, bex, cex;
    ai = std::get<6>(GetParam());
    aj = std::get<7>(GetParam());
    aex = std::get<8>(GetParam());

    bi = std::get<9>(GetParam());
    bj = std::get<10>(GetParam());
    bex = std::get<11>(GetParam());

    ci = std::get<12>(GetParam());
    cj = std::get<13>(GetParam());
    cex = std::get<14>(GetParam());

    // specifies alpha value
    T alpha = std::get<15>(GetParam());
    // specifies beta value
    T beta = std::get<16>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<17>(GetParam());
    gtint_t ldb_inc = std::get<18>(GetParam());
    gtint_t ldc_inc = std::get<19>(GetParam());

    // Set the threshold for the errors:
    double thresh = 10*m*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc,
                  alpha, beta, ai, aj, aex, bi, bj, bex, ci, cj, cex, thresh );
}

// Helper classes for printing the test case parameters based on the instantiator
// These are mainly used to help with debugging, in case of failures

// Utility to print the test-case in case of exception value on matrices
class ZGemmEVMatPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t, dcomplex,
                                          gtint_t, gtint_t, dcomplex, gtint_t, gtint_t, dcomplex, dcomplex, dcomplex,
                                          gtint_t, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char tsa        = std::get<1>(str.param);
        char tsb        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        gtint_t ai, aj, bi, bj, ci, cj;
        dcomplex aex, bex, cex;
        ai = std::get<6>(str.param);
        aj = std::get<7>(str.param);
        aex = std::get<8>(str.param);

        bi = std::get<9>(str.param);
        bj = std::get<10>(str.param);
        bex = std::get<11>(str.param);

        ci = std::get<12>(str.param);
        cj = std::get<13>(str.param);
        cex = std::get<14>(str.param);

        dcomplex alpha  = std::get<15>(str.param);
        dcomplex beta   = std::get<16>(str.param);
        gtint_t lda_inc = std::get<17>(str.param);
        gtint_t ldb_inc = std::get<18>(str.param);
        gtint_t ldc_inc = std::get<19>(str.param);

#ifdef TEST_BLAS
        std::string str_name = "zgemm_";
#elif TEST_CBLAS
        std::string str_name = "cblas_zgemm";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_zgemm";
#endif
        str_name = str_name + "_" + sfm+sfm+sfm;
        str_name = str_name + "_" + tsa + tsb;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(n);
        str_name = str_name + "_" + std::to_string(k);
        str_name = str_name + "_A" + std::to_string(ai) + std::to_string(aj);
        str_name = str_name + "_" + testinghelpers::get_value_string(aex);
        str_name = str_name + "_B" + std::to_string(bi) + std::to_string(bj);
        str_name = str_name + "_" + testinghelpers::get_value_string(bex);
        str_name = str_name + "_C" + std::to_string(ci) + std::to_string(cj);
        str_name = str_name + "_" + testinghelpers::get_value_string(cex);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + std::to_string(lda_inc);
        str_name = str_name + "_" + std::to_string(ldb_inc);
        str_name = str_name + "_" + std::to_string(ldc_inc);
        return str_name;
    }
};

// Utility to print the test-case in case of exception value on matrices
class ZGemmEVAlphaBetaPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t, dcomplex,
                                          gtint_t, gtint_t, dcomplex, gtint_t, gtint_t, dcomplex, dcomplex, dcomplex,
                                          gtint_t, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char tsa        = std::get<1>(str.param);
        char tsb        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);

        dcomplex alpha  = std::get<15>(str.param);
        dcomplex beta   = std::get<16>(str.param);
        gtint_t lda_inc = std::get<17>(str.param);
        gtint_t ldb_inc = std::get<18>(str.param);
        gtint_t ldc_inc = std::get<19>(str.param);

#ifdef TEST_BLAS
        std::string str_name = "zgemm_";
#elif TEST_CBLAS
        std::string str_name = "cblas_zgemm";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_zgemm";
#endif
        str_name = str_name + "_" + sfm+sfm+sfm;
        str_name = str_name + "_" + tsa + tsb;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(n);
        str_name = str_name + "_" + std::to_string(k);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + std::to_string(lda_inc);
        str_name = str_name + "_" + std::to_string(ldb_inc);
        str_name = str_name + "_" + std::to_string(ldc_inc);
        return str_name;
    }
};

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

// Exception value testing(on matrices)

/*
    For the bli_zgemm_4x4_avx2_k1_nn kernel, the main and fringe dimensions are as follows:
    For m : Main = { 4 }, fringe = { 2, 1 }
    For n : Main = { 4 }, fringe = { 2, 1 }

    Without any changes to the BLAS layer in BLIS, the fringe case of 1 cannot be touched
    separately, since if m/n is 1, the inputs are redirected to ZGEMV.

*/

// Testing for the main loop case for m and n
// The kernel uses 2 loads and 4 broadcasts. The exception values
// are induced at one index individually for each of the loads.
// They are also induced in the broadcast direction at two places.
INSTANTIATE_TEST_SUITE_P(
        bli_zgemm_4x4_avx2_k1_nn_evt_mat_main,
        ZGemmEVTTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(4)),                                  // m
            ::testing::Values(gtint_t(4)),                                  // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // cexval
            ::testing::Values(dcomplex{-2.2, 3.3}),                         // alpha
            ::testing::Values(dcomplex{1.2, -2.3}),                         // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::ZGemmEVMatPrint()
    );

// Testing the fringe cases
// Fringe case minimum size is 2 along both m and n.
// Invloves only one load(AVX2 or (AVX2+SSE)). Thus,
// the exception values are induced at the first and second indices of the
// column vector A and row vector B.
INSTANTIATE_TEST_SUITE_P(
        bli_zgemm_4x4_avx2_k1_nn_evt_mat_fringe,
        ZGemmEVTTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(2), gtint_t(3)),                      // m
            ::testing::Values(gtint_t(2), gtint_t(3)),                      // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // bj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // bexval
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // ci
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // cj
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),     // cexval
            ::testing::Values(dcomplex{-2.2, 3.3}),                         // alpha
            ::testing::Values(dcomplex{1.2, -2.3}),                         // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::ZGemmEVMatPrint()
    );

// Exception value testing(on alpha and beta)
// Alpha and beta are set to exception values
INSTANTIATE_TEST_SUITE_P(
        bli_zgemm_4x4_avx2_k1_nn_evt_alphabeta,
        ZGemmEVTTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n'),                                          // transa
            ::testing::Values('n'),                                          // transb
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(4)),           // m
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(4)),           // n
            ::testing::Values(gtint_t(1)),                                   // k
            ::testing::Values(gtint_t(0)),                                   // ai
            ::testing::Values(gtint_t(0)),                                   // aj
            ::testing::Values(dcomplex{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                   // bi
            ::testing::Values(gtint_t(0)),                                   // bj
            ::testing::Values(dcomplex{0.0, 0.0}),
            ::testing::Values(gtint_t(0)),                                   // ci
            ::testing::Values(gtint_t(0)),                                   // cj
            ::testing::Values(dcomplex{0.0, 0.0}),
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),      // alpha
            ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0},
                              dcomplex{3.4, NaN}, dcomplex{NaN, -Inf}),      // beta
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                   // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                    // increment to the leading dim of c
        ),
        ::ZGemmEVAlphaBetaPrint()
    );
