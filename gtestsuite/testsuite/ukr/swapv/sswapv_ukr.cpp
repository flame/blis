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
#include "test_swapv_ukr.h"

class sswapvGeneric :
        public ::testing::TestWithParam<std::tuple<sswapv_ker_ft,   // Function pointer for dswapv kernels
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sswapvGeneric);

TEST_P( sswapvGeneric, UKR )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the kernel to be tested:
    sswapv_ker_ft ukr = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<4>(GetParam());

    using T = float;

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_swapv_ukr<T, sswapv_ker_ft>( ukr, n, incx, incy, is_memory_test );
}

// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

// Tests for bli_dswapv_zen_int8 (AVX2) kernel.
// For unit inc on x and y:
// When n values are 64, 32, 16, 8, 4 it is avx2 optimised

INSTANTIATE_TEST_SUITE_P(
        UnitIncrements,
        sswapvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sswapv_zen_int8),
            // n: size of vector.
            ::testing::Values(
                gtint_t(1), gtint_t(2), gtint_t(8), gtint_t(16), gtint_t(32),
                gtint_t(64), gtint_t(128), gtint_t(9), gtint_t(17), gtint_t(33),
                gtint_t(65), gtint_t(129), gtint_t(10), gtint_t(18), gtint_t(34),
                gtint_t(68), gtint_t(130), gtint_t(24), gtint_t(40), gtint_t(72),
                gtint_t(136), gtint_t(96), gtint_t(160)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1)
            ),
            // is_memory_test
            ::testing::Values(false, true)
        ),
        ::swapvUKRPrint<sswapv_ker_ft>()
    );

INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        sswapvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sswapv_zen_int8),
            // n: size of vector.
            ::testing::Values(
                gtint_t(1),
                gtint_t(9),
                gtint_t(55)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(500)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(500)
            ),
            // is_memory_test
            ::testing::Values(false, true)
        ),
        ::swapvUKRPrint<sswapv_ker_ft>()
    );
#endif
