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
#include "test_axpyf.h"

class daxpyfGeneric :
        public ::testing::TestWithParam<std::tuple<char,    // conjx
                                                   char,    // conja
                                                   gtint_t, // m
                                                   gtint_t, // b
                                                   double,  // alpha
                                                   gtint_t, // inca
                                                   gtint_t, // lda
                                                   gtint_t, // incx
                                                   gtint_t  // incy
                                                   >> {};
// Tests using random integers as vector elements.
TEST_P( daxpyfGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    char conj_a = std::get<1>(GetParam());
    gtint_t m = std::get<2>(GetParam());
    gtint_t b = std::get<3>(GetParam());
    T alpha = std::get<4>(GetParam());

    // stride size for x:
    gtint_t inca = std::get<5>(GetParam());
    // stride size for y:
    gtint_t lda = std::get<6>(GetParam());
    gtint_t incx = std::get<7>(GetParam());
    gtint_t incy = std::get<8>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpyf.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (alpha == testinghelpers::ONE<T>())
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 6.6;
#endif
        thresh = adj*(b+1)*testinghelpers::getEpsilon<T>();
    }
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 6.9;
#endif
        thresh = adj*(2*b+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_axpyf<T>( conj_x, conj_a, m, b, &alpha, inca, lda, incx, incy, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_axpyf<T>( conj_x, conj_a, m, b, &alpha, inca, lda, incx, incy, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_axpyf<T>( conj_x, conj_a, m, b, &alpha, inca, lda, incx, incy, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_axpyf<T>( conj_x, conj_a, m, b, &alpha, inca, lda, incx, incy, thresh );
#endif
}

// Black box testing for generic and main use of daxpy.
INSTANTIATE_TEST_SUITE_P(
        FunctionalTest,
        daxpyfGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of matrix
            ::testing::Range(gtint_t(6), gtint_t(10), 1),                    // b size of matrix
            ::testing::Values(double(0.0), double(1.0), double(2.3)),        // alpha
            ::testing::Values(gtint_t(0)),                                   // lda increment
            ::testing::Values(gtint_t(1)),                                   // stride size for a
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::axpyfGenericPrint<double>()
    );

