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
#include "test_copyv_ukr.h"

class zcopyvGeneric :
        public ::testing::TestWithParam<std::tuple<zcopyv_ker_ft,   // Function pointer type for zcopyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zcopyvGeneric);

// Tests using random integers as vector elements.
TEST_P( zcopyvGeneric, UKR )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    zcopyv_ker_ft ukr_fp = std::get<0>(GetParam());
    // denotes whether vec x is n,c
    char conjx = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<4>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<5>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_copyv_ukr<T, zcopyv_ker_ft>( ukr_fp, conjx, n, incx, incy, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_zcopyv_zen_int kernel.
    The code structure for bli_zcopyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 16 -->   L16
        Fringe loops :  In blocks of 8  -->   L8
                        In blocks of 4  -->   L4
                        In blocks of 2  -->   L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with Unit Strides(US), across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zcopyv_zen_int_unitStrides,
        zcopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zcopyv_zen_int),
            ::testing::Values('n'                                           // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(16),            // size n, for L16
                              gtint_t(8),             // L8
                              gtint_t(4),             // L4
                              gtint_t(2),             // L2
                              gtint_t(1),             // LScalar
                              // Testing the loops with combinations
                              gtint_t(80),           // 5*L16
                              gtint_t(88),           // 5*L16 + L8
                              gtint_t(92),           // 5*L16 + L8 + L4
                              gtint_t(94),           // 5*L16 + L8 + L4 + L2
                              gtint_t(95)),          // 5*L16 + L8 + L4 + L2 + 1(LScalar)
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(false, true)            // is_memory_test
        ),
        ::copyvUKRPrint<zcopyv_ker_ft>()
    );

// Unit testing with Non-Unit Strides(US), across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zcopyv_zen_int_nonUnitStrides,
        zcopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zcopyv_zen_int),
            ::testing::Values('n'                                           // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),
            ::testing::Values(gtint_t(25), gtint_t(37)), // size of the vector
            ::testing::Values(gtint_t(5)),               // stride size for x
            ::testing::Values(gtint_t(3)),               // stride size for y
            ::testing::Values(false, true)               // is_memory_test
        ),
        ::copyvUKRPrint<zcopyv_ker_ft>()
    );
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_zcopyv_zen4_asm_avx512 kernel.
    The code structure for bli_zcopyv_zen4_asm_avx512( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 128 -->   L128
        Fringe loops :  In blocks of 64  -->   L64
                        In blocks of 32  -->   L32
                        In blocks of 16  -->   L16
                        In blocks of 8   -->   L8
                        In blocks of 4   -->   L4
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with Unit Strides(US), across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zcopyv_zen4_asm_avx512_unitStrides,
        zcopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zcopyv_zen4_asm_avx512),
            ::testing::Values('n'                                           // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(128),           // size n, for L128
                              gtint_t(64),            // L64
                              gtint_t(32),            // L32
                              gtint_t(16),            // L16
                              gtint_t(8),             // L8
                              gtint_t(4),             // L4
                              gtint_t(3),             // LScalar
                              // Testing the loops with combinations
                              gtint_t(1280),           // 5*L256
                              gtint_t(1408),           // 5*L256 + L128
                              gtint_t(1472),           // 5*L256 + L128 + L32
                              gtint_t(1504),           // 5*L256 + L128 + L32 + L16
                              gtint_t(1520),           // 5*L258 + L128 + L32 + L16 + L8
                              gtint_t(1528),           // 5*L258 + L128 + L32 + L16 + L8 + L4
                              gtint_t(1531)),          // 5*L258 + L128 + L32 + L16 + L8 + L4 + 3(LScalar)
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(false, true)            // is_memory_test
        ),
        ::copyvUKRPrint<zcopyv_ker_ft>()
    );

// Unit testing with Non-Unit Strides(US), across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zcopyv_zen4_asm_avx512_nonUnitStrides,
        zcopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zcopyv_zen4_asm_avx512),
            ::testing::Values('n'                                           // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),
            ::testing::Values(gtint_t(25), gtint_t(37)), // size of the vector
            ::testing::Values(gtint_t(5)),               // stride size for x
            ::testing::Values(gtint_t(3)),               // stride size for y
            ::testing::Values(false, true)               // is_memory_test
        ),
        ::copyvUKRPrint<zcopyv_ker_ft>()
    );
#endif
