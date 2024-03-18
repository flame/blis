/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_dotxf.h"

class ddotxfGeneric :
        public ::testing::TestWithParam<std::tuple<char,    // conj_x
                                                   char,    // conj_a
                                                   gtint_t, // m
                                                   gtint_t, // b
                                                   double,  // alpha
                                                   gtint_t, // inca
                                                   gtint_t, // lda
                                                   gtint_t, // incx
                                                   double,  // beta
                                                   gtint_t  // incy
                                                   >> {};
// Tests using random integers as vector elements.
TEST_P( ddotxfGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be used
    char conj_x = std::get<0>(GetParam());
    // denotes whether A or conj(A) will be used
    char conj_a = std::get<1>(GetParam());
    // matrix size m
    gtint_t m = std::get<2>(GetParam());
    // matrix size n
    gtint_t b = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());
    // lda increment for A
    gtint_t lda_inc = std::get<5>(GetParam());
    // stride size for A
    gtint_t inca = std::get<6>(GetParam());
    // stride size for x
    gtint_t incx = std::get<7>(GetParam());
    // beta
    T beta = std::get<8>(GetParam());
    // stride size for y
    gtint_t incy = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite dotxf.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    // Threshold adjustment
    if (m == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        if (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>())
            thresh = 0.0;
        else
            thresh = testinghelpers::getEpsilon<T>();
    else if (alpha == testinghelpers::ONE<T>())
        if (beta == testinghelpers::ZERO<T>())
            thresh = (m)*testinghelpers::getEpsilon<T>();
        else if (beta == testinghelpers::ONE<T>())
        {
#ifdef BLIS_INT_ELEMENT_TYPE
            double adj = 1.0;
#else
            double adj = 4.4;
#endif
            thresh = adj*(m+1)*testinghelpers::getEpsilon<T>();
        }
        else
            thresh = (m+2)*testinghelpers::getEpsilon<T>();
    else
        if (beta == testinghelpers::ZERO<T>())
            thresh = (2*m)*testinghelpers::getEpsilon<T>();
        else if (beta == testinghelpers::ONE<T>())
        {
#ifdef BLIS_INT_ELEMENT_TYPE
            double adj = 1.0;
#else
            double adj = 5.3;
#endif
            thresh = adj*(2*m+1)*testinghelpers::getEpsilon<T>();
        }
        else
        {
            thresh = (2*m+2)*testinghelpers::getEpsilon<T>();
        }

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_dotxf<T>( conj_x, conj_a, m, b, &alpha, inca, lda_inc, incx, &beta, incy, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_dotxf<T>( conj_x, conj_a, m, b, &alpha, inca, lda_inc, incx, &beta, incy, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_dotxf<T>( conj_x, conj_a, m, b, &alpha, inca, lda_inc, incx, &beta, incy, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_dotxf<T>( conj_x, conj_a, m, b, &alpha, inca, lda_inc, incx, &beta, incy, thresh );
#endif
}

// Black box testing for generic and main use of ddotxf.
INSTANTIATE_TEST_SUITE_P(
        FunctionalTest,
        ddotxfGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                         // n: use x, not conj(x) (since it is real)
            ::testing::Values('n'),                                         // n: use x, not conj(x) (since it is real)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                // m size of matrix
            ::testing::Range(gtint_t(6), gtint_t(10), 1),                   // b size of matrix
            ::testing::Values(double(0.0), double(1.0), double(2.3)),       // alpha
            ::testing::Values(gtint_t(0)),                                  // lda increment
            ::testing::Values(gtint_t(1)),                                  // stride size for a
            ::testing::Values(gtint_t(1)),                                  // stride size for x
            ::testing::Values(double(1.0)),                                 // beta
            ::testing::Values(gtint_t(1))                                   // stride size for y
        ),
        ::dotxfGenericPrint<double>()
    );

