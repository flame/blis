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
#include "test_herk.h"

class zherkGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   double,
                                                   double,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P( zherkGeneric, API )
{
    using T = dcomplex;
    using RT = typename testinghelpers::type_info<T>::real_type;
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
    // matrix size n
    gtint_t n  = std::get<3>(GetParam());
    // matrix size k
    gtint_t k  = std::get<4>(GetParam());
    // specifies alpha value
    RT alpha = std::get<5>(GetParam());
    // specifies beta value
    RT beta = std::get<6>(GetParam());
    // lda, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<7>(GetParam());
    gtint_t ldc_inc = std::get<8>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite herk.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == 0.0 || k == 0) && (beta == 0.0 || beta == 1.0))
        thresh = 0.0;
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 4.1;
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
        test_herk<T>( storage, uplo, transa, n, k, lda_inc, ldc_inc, alpha, beta, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_herk<T>( storage, uplo, transa, n, k, lda_inc, ldc_inc, alpha, beta, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_herk<T>( storage, uplo, transa, n, k, lda_inc, ldc_inc, alpha, beta, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_herk<T>( storage, uplo, transa, n, k, lda_inc, ldc_inc, alpha, beta, thresh );
#endif
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zherkGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // u:upper, l:lower
            ::testing::Values('n','c'),                                      // transa
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(30)),          // n
            ::testing::Values(gtint_t(1), gtint_t(2), gtint_t(26)),          // k
            ::testing::Values(2.0, -1.0),                                    // alpha
            ::testing::Values(-3.0, 2.0),                                    // beta
            ::testing::Values(gtint_t(0), gtint_t(4)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(2))                        // increment to the leading dim of c
        ),
        ::herkGenericPrint<dcomplex>()
    );
