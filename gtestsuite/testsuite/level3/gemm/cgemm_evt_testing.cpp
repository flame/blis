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
#include "test_gemm.h"

using T = scomplex;

static float AOCL_NAN = std::numeric_limits<float>::quiet_NaN();
static float AOCL_INF = std::numeric_limits<float>::infinity();

class cgemmEVT :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   gtint_t,    // MatrixA row index
                                                   gtint_t,    // MatrixA col index
                                                   T,          // MatrixA Exception value
                                                   gtint_t,    // MatrixB row index
                                                   gtint_t,    // MatrixB col index
                                                   T,          // MatrixB Exception value
                                                   gtint_t,    // MatrixC row index
                                                   gtint_t,    // MatrixC col index
                                                   T,          // MatrixC Exception value
                                                   T,          // alpha
                                                   T,          // beta
                                                   gtint_t,    // inc to the lda
                                                   gtint_t,    // inc to the ldb
                                                   gtint_t     // inc to the ldc
                                                   >> {};

TEST_P(cgemmEVT, NaNInfCheck)
{
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

    // ai, aj, bi, bj, ci, cj - Indices of all Matrices where
    // EV to be inserted
    gtint_t ai, aj, bi, bj, ci, cj;

    // aex, bex, cex - Exception value(EV) for each Matrix
    T aex, bex, cex;
    ai  = std::get<6>(GetParam());
    aj  = std::get<7>(GetParam());
    aex = std::get<8>(GetParam());

    bi  = std::get<9>(GetParam());
    bj  = std::get<10>(GetParam());
    bex = std::get<11>(GetParam());

    ci  = std::get<12>(GetParam());
    cj  = std::get<13>(GetParam());
    cex = std::get<14>(GetParam());

    // specifies alpha value
    T alpha = std::get<15>(GetParam());
    // specifies beta value
    T beta = std::get<16>(GetParam());

    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative,
    // the array size is bigger than the matrix size.
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

class cgemmPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, gtint_t,
                                          gtint_t, T, gtint_t, gtint_t, T, gtint_t, gtint_t, T,
                                          T, T, gtint_t, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char tsa        = std::get<1>(str.param);
        char tsb        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        gtint_t ai, aj, bi, bj, ci, cj;
        T aex, bex, cex;
        ai  = std::get<6>(str.param);
        aj  = std::get<7>(str.param);
        aex = std::get<8>(str.param);

        bi  = std::get<9>(str.param);
        bj  = std::get<10>(str.param);
        bex = std::get<11>(str.param);

        ci  = std::get<12>(str.param);
        cj  = std::get<13>(str.param);
        cex = std::get<14>(str.param);

        T alpha  = std::get<15>(str.param);
        T beta   = std::get<16>(str.param);
        gtint_t lda_inc = std::get<17>(str.param);
        gtint_t ldb_inc = std::get<18>(str.param);
        gtint_t ldc_inc = std::get<19>(str.param);
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
        str_name = str_name + "_A" + std::to_string(ai) + std::to_string(aj);
        str_name = str_name + "_" + testinghelpers::get_value_string(aex);
        str_name = str_name + "_B" + std::to_string(bi) + std::to_string(bj);
        str_name = str_name + "_" + testinghelpers::get_value_string(bex);
        str_name = str_name + "_C" + std::to_string(ci) + std::to_string(cj);
        str_name = str_name + "_" + testinghelpers::get_value_string(cex);
        str_name = str_name + "_alpha" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_beta" + testinghelpers::get_value_string(beta);
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
/* Testing ExceptionValue testing for SUP and Native implementation */
/* of cgemm API                                                     */
/********************************************************************/
/* Exception Values are AOCL_NAN, AOCL_INF, -AOCL_INF               */
/* 1. Matrix:                                                       */
/*    These values are inserted in user provided (i,j)th indices of */
/*    Matrix A, B, C                                                */
/* 2. Scaling Values:                                               */
/*    These values are inserted as alpha, beta values               */
/********************************************************************/

//Failures observed for EV: T{AOCL_INF, 0.0}
INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_No_Trans,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(300), gtint_t(310)),                  // m
            ::testing::Values(gtint_t(200), gtint_t(210)),                  // n
            ::testing::Values(gtint_t(150), gtint_t(155)),                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.2}, T{AOCL_INF, 5.2},
                              T{-3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, -2.3}, T{AOCL_INF, 8.9},
                              T{-3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{AOCL_NAN, 1.3}, T{AOCL_INF, 7.4},
                              T{3.3, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-1.0, -2.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                               T{91.0, 0.0}, T{0.0, 1.0}),                  // alpha
            ::testing::Values(T{12.0, 2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{12.0, 0.0}, T{0.0, 1.0}),                   // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Trans,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('t'),                                         // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(gtint_t(300), gtint_t(310)),                  // m
            ::testing::Values(gtint_t(200), gtint_t(210)),                  // n
            ::testing::Values(gtint_t(150), gtint_t(155)),                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 2.2}, T{AOCL_INF, -9.0},
                              T{-3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, -2.3}, T{AOCL_INF, -6.7},
                              T{-3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{AOCL_NAN, 1.3}, T{AOCL_INF, 5.6},
                              T{3.3, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // cexval
            ::testing::Values(T{-1.0, -2.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{12.0, 0.0}, T{0.0, 1.0}),                   // alpha
            ::testing::Values(T{12.0, 2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{12.0, 0.0}, T{0.0, 1.0}),                   // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_zeros_And_ExcpetionValues,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(200)),                                // m
            ::testing::Values(gtint_t(100)),                                // n
            ::testing::Values(gtint_t(150)),                                // k
            ::testing::Values(gtint_t(3)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(2)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(3)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // cexval
            ::testing::Values(T{-1.0, -2.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{2.3, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{12.0, 2.3}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{3.2, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Alpha_Beta,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(200), gtint_t(210)),                  // m
            ::testing::Values(gtint_t(100), gtint_t(110)),                  // n
            ::testing::Values(gtint_t(50), gtint_t(55)),                    // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{1.2, 2.3}),                                 // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{-2.3, -12}),                                // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{-0.7, 3.2}),                                // cexval
            ::testing::Values(T{AOCL_NAN, 1.4}, T{AOCL_INF, 7.4},
                              T{4.2, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF},
                              T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // alpha
            ::testing::Values(T{AOCL_NAN, 5.2}, T{AOCL_INF, 3.4},
                              T{1.6, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF},
                              T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_No_Trans,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(500), gtint_t(700)),                  // m
            ::testing::Values(gtint_t(680), gtint_t(1000)),                 // n
            ::testing::Values(gtint_t(370), gtint_t(375)),                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 9.3}, T{AOCL_INF, 3.9},
                              T{13.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, -5.6}, T{AOCL_INF, -3.1},
                              T{9.7, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{AOCL_NAN, 7.8}, T{AOCL_INF, -6.7},
                              T{-3.6, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // cexval
            ::testing::Values(T{-21.0, -12.0}),                             // alpha
            ::testing::Values(T{1.0, 2.13}),                                // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Trans,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('t'),                                         // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(gtint_t(595), gtint_t(900)),                  // m
            ::testing::Values(gtint_t(880), gtint_t(1200)),                 // n
            ::testing::Values(gtint_t(470), gtint_t(475)),                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 9.3}, T{AOCL_INF, -5.6},
                              T{13.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, -5.6}, T{AOCL_INF, 3.2},
                              T{9.7, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{AOCL_NAN, 7.8}, T{AOCL_INF, -6.7},
                              T{-3.6, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // cexval
            ::testing::Values(T{-21.0, -12.0}),                             // alpha
            ::testing::Values(T{1.0, 2.13}),                                // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Conj,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('c'),                                         // transa
            ::testing::Values('c'),                                         // transb
            ::testing::Values(gtint_t(700)),                                // m
            ::testing::Values(gtint_t(990)),                                // n
            ::testing::Values(gtint_t(475)),                                // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 9.3}, T{AOCL_INF, -3.2},
                              T{13.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{AOCL_NAN, -5.6}, T{AOCL_INF, 5.2},
                              T{9.7, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),    // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{AOCL_NAN, 7.8}, T{AOCL_INF, 7.6},
                              T{-3.6, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF}),   // cexval
            ::testing::Values(T{-21.0, -12.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{9.8, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.0, 2.13}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{4.3, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_zeros_And_ExcpetionValues,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(700), gtint_t(800)),                  // m
            ::testing::Values(gtint_t(990), gtint_t(1100)),                 // n
            ::testing::Values(gtint_t(475), gtint_t(575)),                  // k
            ::testing::Values(gtint_t(3)),                                  // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(2)),                                  // bj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // bexval
            ::testing::Values(gtint_t(2)),                                  // ci
            ::testing::Values(gtint_t(3)),                                  // cj
            ::testing::Values(T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // cexval
            ::testing::Values(T{-21.0, -12.0}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{2.4, 0.0}, T{0.0, 1.0}),                    // alpha
            ::testing::Values(T{1.0, 2.13}, T{0.0, 0.0},
                              T{1.0, 0.0}, T{-1.0, 0.0},
                              T{4.5, 0.0}, T{0.0, 1.0}),                    // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Alpha_Beta,
        cgemmEVT,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 't', 'c'),                               // transa
            ::testing::Values('n', 't', 'c'),                               // transb
            ::testing::Values(gtint_t(700), gtint_t(900)),                  // m
            ::testing::Values(gtint_t(1000), gtint_t(2000)),                // n
            ::testing::Values(gtint_t(470), gtint_t(475)),                  // k
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // ai
            ::testing::Values(gtint_t(0)),                                  // aj
            ::testing::Values(T{1.12, 12.3}),                               // aexval
            ::testing::Values(gtint_t(0)),                                  // bi
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // bj
            ::testing::Values(T{-12.3, -2}),                                // bexval
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // ci
            ::testing::Values(gtint_t(1), gtint_t(3)),                      // cj
            ::testing::Values(T{-1.7, -3.12}),                              // cexval
            ::testing::Values(T{AOCL_NAN, 2.3}, T{AOCL_INF, 8.9},
                              T{3.4, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF},
                              T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // alpha
            ::testing::Values(T{AOCL_NAN, 5.3}, T{AOCL_INF, 3.5},
                              T{2.9, AOCL_NAN}, T{AOCL_NAN, -AOCL_INF},
                              T{AOCL_NAN, 0}, T{AOCL_INF, 0.0},
                              T{0, AOCL_NAN}, T{0, -AOCL_INF}),             // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::cgemmPrint()
    );
