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

class ZGEMMAPI :
        public ::testing::TestWithParam<std::tuple<char,       // storage format
                                                   char,       // transa
                                                   char,       // transb
                                                   gtint_t,    // m
                                                   gtint_t,    // n
                                                   gtint_t,    // k
                                                   dcomplex,   //alpha
                                                   dcomplex,   //beta
                                                   gtint_t,    // inc to the lda 
                                                   gtint_t,    // inc to the ldb 
                                                   gtint_t     // inc to the ldc
                                                   >> {};

TEST_P(ZGEMMAPI, FunctionalTest)
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
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();
        //thresh = (15*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
}

class ZGEMMPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, dcomplex, dcomplex, gtint_t, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char tsa        = std::get<1>(str.param);
        char tsb        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        dcomplex alpha  = std::get<6>(str.param);
        dcomplex beta   = std::get<7>(str.param);
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
        str_name = str_name + "storageC_" + sfm;
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
/* Blas interface testing as per the code sequence                  */
/* Below API's will be invoked if input condition is satisified     */
/* List of API's    - Input conditions                              */
/* SCALM            : alpha = 0                                     */
/* GEMV             : m = 1 or n = 1                                */
/* K1               : k = 1 & tranaA = 'n' &  transB = 'n;          */
/* Small ST         : ((m0*k0) <= 16384) || ((n0*k0) <= 16384)))    */
/* SUP AVX2         : (m & n & k) <= 128                            */
/* SUP AVX512       : (m & k) <= 128  & n <= 110                    */
/* Native           : Default path,                                 */
/*                  : when none of the above API's are invoked      */
/********************************************************************/
INSTANTIATE_TEST_SUITE_P(
        SCALM,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n','c','t'),                                 // transa
            ::testing::Values('n','c','t'),                                 // transb
            ::testing::Values(gtint_t(10)),                                 // m
            ::testing::Values(gtint_t(10)),                                 // n
            ::testing::Values(gtint_t(10)),                                 // k
            ::testing::Values(dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{3.1, 15.9},
                              dcomplex{0.0, 0.0}),                          //beta
            ::testing::Values(gtint_t(0), gtint_t(130)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(120)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(150))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_M1_N1,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(1)),                                  // m
            ::testing::Values(gtint_t(1)),                                  // n
            ::testing::Range(gtint_t(100), gtint_t(200), gtint_t(100)),     // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(230)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(220)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(250))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_M1,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(1)),                                  // m
            ::testing::Range(gtint_t(2), gtint_t(200), gtint_t(40)),        // n
            ::testing::Range(gtint_t(100), gtint_t(200), gtint_t(100)),     // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(230)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(220)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(250))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_N1,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Range(gtint_t(1), gtint_t(100), gtint_t(20)),        // m
            ::testing::Values(gtint_t(1)),                                  // n
            ::testing::Range(gtint_t(100), gtint_t(200), gtint_t(100)),     // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{3.1, -1.5},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.3, -2.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(300)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(200)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(500))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

// Unit testing for bli_zgemm_4x4_avx2_k1_nn kernel
/* From the BLAS layer(post parameter checking), the inputs will be redirected to this kernel
   if m != 1, n !=1 and k == 1 */

INSTANTIATE_TEST_SUITE_P(
        K_1,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Range(gtint_t(2), gtint_t(8), 1),                    // m
            ::testing::Range(gtint_t(2), gtint_t(8), 1),                    // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(390)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(290)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(590))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        SMALL_Matrix_ST,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(7), gtint_t(8)), // m
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(7), gtint_t(8)), // n
            ::testing::Values(gtint_t(2), gtint_t(4), gtint_t(10)),            // k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{-1.0, -2.0}), // alpha
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{1.0, 2.0}),   // beta
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(3))                       // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Trans_N,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Range(gtint_t(100), gtint_t(105), gtint_t(1)),       // m
            ::testing::Range(gtint_t(80), gtint_t(85), gtint_t(1)),         // n
            ::testing::Range(gtint_t(1000), gtint_t(1010), gtint_t(1)),     // k
            ::testing::Values(dcomplex{-1.0, -2.0}, dcomplex{0.0, -30.0},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{5.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{12.0, 2.3}, dcomplex{0.0, 1.3},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{5.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(540)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(940)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(240))                                   // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        SKinny_Matrix_Trans_T,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('t'),                                         // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Range(gtint_t(105), gtint_t(110), gtint_t(1)),       // m
            ::testing::Range(gtint_t(190), gtint_t(195), gtint_t(1)),       // n
            ::testing::Range(gtint_t(500), gtint_t(510), gtint_t(1)),       // k
            ::testing::Values(dcomplex{-1.8, -21.0}, dcomplex{0.0, -33.0},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{5.3, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.8, 9.3}, dcomplex{0.0, 3.3},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{2.91, 0.0}, dcomplex{0.0, 0.0}),     // beta
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(0)),                                  // increment to the leading dim of b
            ::testing::Values(gtint_t(0))                                   // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Trans_N_C_T,
        ZGEMMAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(200)),                                // m
            ::testing::Values(gtint_t(180)),                                // n
            ::testing::Values(gtint_t(170)),                                // k
            ::testing::Values(dcomplex{1.5, 3.5}, dcomplex{0.0, -10.0},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{2.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{2.0, 4.1}, dcomplex{0.0, 3.4},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{3.3, 0.0}, dcomplex{0.0, 0.0}),      // beta
            ::testing::Values(gtint_t(0), gtint_t(300)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(200)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(500))                     // increment to the leading dim of c
        ),
        ::ZGEMMPrint()
    );
