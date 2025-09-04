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
#include "level3/gemm/test_gemm.h"

class zgemmGeneric :
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

TEST_P( zgemmGeneric, API )
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
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.2;
#else
        double adj = 2.5;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemm<T>( storage, transa, transb, m, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh );
#endif
}

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
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
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
            ::testing::Values(gtint_t(0), gtint_t(2)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(1)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(5))                     // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_M1_N1,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(1)),                                  // m
            ::testing::Values(gtint_t(1)),                                  // n
            ::testing::Values(gtint_t(100), gtint_t(200)),                  // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(2)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(3)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(5))                     // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_M1,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(1)),                                  // m
            ::testing::Values(gtint_t(2), gtint_t(89), gtint_t(197)),       // n
            ::testing::Values(gtint_t(100), gtint_t(200)),                  // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(2)),                    // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(3)),                    // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(5))                     // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        GEMV_N1,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(1), gtint_t(100), gtint_t(47)),       // m
            ::testing::Values(gtint_t(1)),                                  // n
            ::testing::Values(gtint_t(100), gtint_t(200)),                  // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{3.1, -1.5},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.3, -2.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(3)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(7))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

// Unit testing for bli_zgemm_4x4_avx2_k1_nn kernel
/* From the BLAS layer(post parameter checking), the inputs will be redirected to this kernel
   if m != 1, n !=1 and k == 1 */

INSTANTIATE_TEST_SUITE_P(
        K_1,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(2), gtint_t(9), gtint_t(16)),         // m
            ::testing::Values(gtint_t(2), gtint_t(7)),                      // n
            ::testing::Values(gtint_t(1)),                                  // k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{2.1, -1.9},
                              dcomplex{0.0, 0.0}),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(5)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(9)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(2))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

/* NOTE : The instantiator here defines sizes such that on zen4/zen5 machines,
          the tiny path is taken. */
INSTANTIATE_TEST_SUITE_P(
        Tiny_Matrix_ST,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(2), gtint_t(40), gtint_t(61)),  // m
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(7)),    // n
            ::testing::Values(gtint_t(10), gtint_t(16), gtint_t(21)), // k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{-1.0, -2.0}), // alpha
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{1.0, 2.0}),   // beta
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(3))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        SMALL_Matrix_ST,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values('n', 'c', 't'),                               // transb
            ::testing::Values(gtint_t(201), gtint_t(3), gtint_t(7), gtint_t(8)), // m
            ::testing::Values(gtint_t(2), gtint_t(3), gtint_t(7), gtint_t(8)),   // n
            ::testing::Values(gtint_t(2), gtint_t(4), gtint_t(10)),              // k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{-1.0, -2.0}), // alpha
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0}, dcomplex{0, 1.0}, dcomplex{1.0, 2.0}),   // beta
            ::testing::Values(gtint_t(0), gtint_t(1)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(2)),                      // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(3))                       // increment to the leading dim of c
        ),
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Skinny_Matrix_Trans_N,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(gtint_t(100), gtint_t(105)),                  // m
            ::testing::Values(gtint_t(80), gtint_t(85)),                    // n
            ::testing::Values(gtint_t(1000), gtint_t(1010)),                // k
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
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        SKinny_Matrix_Trans_T,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('t'),                                         // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(gtint_t(105)),                                // m
            ::testing::Values(gtint_t(190)),                                // n
            ::testing::Values(gtint_t(500)),                                // k
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
        ::gemmGenericPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        Large_Matrix_Trans_N_C_T,
        zgemmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
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
        ::gemmGenericPrint<dcomplex>()
    );
