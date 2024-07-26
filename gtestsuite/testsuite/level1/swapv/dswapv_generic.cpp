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
#include "test_swapv.h"

class dswapvGeneric :
        // input params : vector length, stride size of x, stride size of y
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t>> {};

TEST_P( dswapvGeneric, API )
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

    using T = double;

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_swapv<T>( n, incx, incy );
}

/*************************************************************************/
/* When n values are 32, 16, 8, 4 it is avx2 optimised                   */
/* Values to be tested to cover all loops                                */
/* 1, 2, 4, 8, 16, 32, 64, 128 : L1, L1*2, L4, L8, L16, L32, L64, 2*L64  */
/* 5, 9, 17, 33, 65, 129 : L1 + ( L4, L8, L16, L32, L64, 2*L64)          */
/* 6, 10, 18, 34, 68, 130 :  L1*2 + (L4, L8, L16, L32, L64, 2*L64)       */
/* 12, 24, 40, 72, 136 :  L8 + (L4, L16, L32, L64, 2*L64)                */
/* 20, 136 :  L16 + (L4, 2*L64)                                          */
/* 36, 96, 160 :  L32 +(L4, L8, L32, L64, 2*L64)                         */
/*************************************************************************/
INSTANTIATE_TEST_SUITE_P(
        UnitIncrements,
        dswapvGeneric,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values(
                gtint_t(1), gtint_t(2), gtint_t(4), gtint_t(8), gtint_t(16), gtint_t(32),
                gtint_t(64), gtint_t(128), gtint_t(5), gtint_t(9), gtint_t(17), gtint_t(33),
                gtint_t(65), gtint_t(129), gtint_t(6), gtint_t(10), gtint_t(18), gtint_t(34),
                gtint_t(68), gtint_t(130), gtint_t(12), gtint_t(24), gtint_t(40), gtint_t(72),
                gtint_t(136), gtint_t(20), gtint_t(36), gtint_t(96), gtint_t(160)
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
        dswapvGeneric,
        ::testing::Combine(
            // n: size of vector.
            ::testing::Values(
                gtint_t( 1),
                gtint_t( 9),
                gtint_t(55)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(500), gtint_t(-600)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(100), gtint_t(-500)
            )
        ),
        ::swapvGenericPrint()
    );
