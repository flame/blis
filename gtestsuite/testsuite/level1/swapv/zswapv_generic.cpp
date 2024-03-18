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
#include "test_swapv.h"

class zswapvGeneric :
        // input params : vector length, stride size of x, stride size of y
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t>> {};

TEST_P( zswapvGeneric, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<2>(GetParam());

    using T = dcomplex;

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_swapv<T>( n, incx, incy );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_swapv<T>( n, incx, incy );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_swapv<T>( n, incx, incy );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_swapv<T>( n, incx, incy );
#endif
}

INSTANTIATE_TEST_SUITE_P(
        UnitIncrements,
        zswapvGeneric,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values(
                gtint_t(1),
                gtint_t(50),
                gtint_t(100)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1)
            )
        ),
        ::swapvGenericPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        zswapvGeneric,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values(
                gtint_t(1),
                gtint_t(9),
                gtint_t(55)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(500), gtint_t(-100)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(100), gtint_t(-200)
            )
        ),
        ::swapvGenericPrint()
    );
