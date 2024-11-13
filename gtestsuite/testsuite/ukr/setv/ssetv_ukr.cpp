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
#include "test_setv_ukr.h"
#include "common/blis_version_defs.h"

using T = float;
using FT = ssetv_ker_ft;

class ssetvGeneric :
        public ::testing::TestWithParam<std::tuple<FT,              // Function pointer type for ssetv kernels
                                                   char,            // conjalpha
                                                   T,               // alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ssetvGeneric);

// Tests using random integers as vector elements.
TEST_P( ssetvGeneric, UKR )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    FT ukr_fp = std::get<0>(GetParam());
    // denotes conjalpha
    char conjalpha = std::get<1>(GetParam());
    // denotes alpha
    T alpha = std::get<2>(GetParam());
    // vector length
    gtint_t n = std::get<3>(GetParam());
    // stride size for x
    gtint_t incx = std::get<4>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<5>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_setv_ukr<T, FT>( ukr_fp, conjalpha, alpha, n, incx, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_ssetv_zen_int kernel.
    The code structure for bli_ssetv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 128 -->  L128
        Fringe loops :  In blocks of 64  -->  L64
                        In blocks of 32  -->  L32
                        In blocks of 16  -->  L16
                        In blocks of 8   -->  L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
#ifdef K_bli_ssetv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_ssetv_zen_int_unitStrides,
        ssetvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ssetv_zen_int),
            ::testing::Values('n', 'c'),              // conjalpha
            ::testing::Values(float(1.2)),            // alpha
            ::testing::Values(// Testing the loops standalone
                              gtint_t(128),           // for size n, L128
                              gtint_t(64),            // L64
                              gtint_t(32),            // L32
                              gtint_t(16),            // L16
                              gtint_t(8),             // L8
                              gtint_t(7),             // LScalar
                              // Testing the loops with combinations
                              // 2*L128
                              gtint_t(256),
                              // 2*L128 + L64
                              gtint_t(320),
                              // 2*L128 + L64 + L32
                              gtint_t(352),
                              // 2*L128 + L64 + L32 + L16
                              gtint_t(368),
                              // 2*L128 + L64 + L32 + L16 + L8
                              gtint_t(376),
                              // 2*L128 + L64 + L32 + L16 + L8 + LScalar
                              gtint_t(383)),
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(false, true)            // is_memory_test
        ),
        (::setvUkrPrint<T, FT>())
    );
#endif

// Unit testing with non-unit strides, across all loops.
#ifdef K_bli_ssetv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_ssetv_zen_int_nonUnitStrides,
        ssetvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ssetv_zen_int),
            ::testing::Values('n', 'c'),                 // conjalpha
            ::testing::Values(float(1.2)),               // alpha
            ::testing::Values(gtint_t(25), gtint_t(37)), // size of the vector
            ::testing::Values(gtint_t(5)),               // stride size for x
            ::testing::Values(false, true)               // is_memory_test
        ),
        (::setvUkrPrint<T, FT>())
    );
#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_ssetv_zen_int_avx512 kernel.
    The code structure for bli_ssetv_zen_int_avx512( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 512 -->  L512
        Fringe loops :  In blocks of 256 -->  L256
                        In blocks of 128 -->  L128
                        In blocks of 64  -->  L64
                        In blocks of 32  -->  L32
                        In blocks of 16  -->  L16
                        Masked loop      --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
#ifdef K_bli_ssetv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_ssetv_zen_int_avx512_unitStrides,
        ssetvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ssetv_zen_int_avx512),
            ::testing::Values('n', 'c'),              // conjalpha
            ::testing::Values(float(1.2)),            // alpha
            ::testing::Values(// Testing the loops standalone
                              gtint_t(512),           // for size n, L512
                              gtint_t(256),           // L64
                              gtint_t(128),           // L128
                              gtint_t(64),            // L64
                              gtint_t(32),            // L32
                              gtint_t(16),            // L16
                              gtint_t(15),            // LScalar
                              // Testing the loops with combinations
                              // 2*L512
                              gtint_t(1024),
                              // 2*L512 + L256
                              gtint_t(1280),
                              // 2*L512 + L256 + L128
                              gtint_t(1408),
                              // 2*L512 + L256 + L128 + L64
                              gtint_t(1472),
                              // 2*L512 + L256 + L128 + L64 + L32
                              gtint_t(1504),
                              // 2*L512 + L256 + L128 + L64 + L32 + L16
                              gtint_t(1520),
                              // 2*L512 + L256 + L128 + L64 + L32 + L16 + LScalar
                              gtint_t(1535)),
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(false, true)            // is_memory_test
        ),
        (::setvUkrPrint<T, FT>())
    );
#endif

// Unit testing with non-unit strides, across all loops.
#ifdef K_bli_ssetv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_ssetv_zen_int_avx512_nonUnitStrides,
        ssetvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ssetv_zen_int_avx512),
            ::testing::Values('n', 'c'),                 // conjalpha
            ::testing::Values(float(1.2)),               // alpha
            ::testing::Values(gtint_t(25), gtint_t(37)), // size of the vector
            ::testing::Values(gtint_t(5)),               // stride size for x
            ::testing::Values(false, true)               // is_memory_test
        ),
        (::setvUkrPrint<T, FT>())
    );
#endif
#endif
