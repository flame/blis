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
#include "common/blis_version_defs.h"

class ccopyvGeneric :
        public ::testing::TestWithParam<std::tuple<ccopyv_ker_ft,   // Function pointer type for ccopyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ccopyvGeneric);

// Tests using random integers as vector elements.
TEST_P( ccopyvGeneric, UKR )
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    ccopyv_ker_ft ukr_fp = std::get<0>(GetParam());
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
    test_copyv_ukr<T, ccopyv_ker_ft>( ukr_fp, conjx, n, incx, incy, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_ccopyv_zen_int kernel.
    The code structure for bli_ccopyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 32 -->   L32
        Fringe loops :  In blocks of 16 -->   L16
                        In blocks of 8  -->   L8
                        In blocks of 4  -->   L4
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with Unit Strides(US), across all loops.
#ifdef K_bli_ccopyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_ccopyv_zen_int_unitStrides,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ccopyv_zen_int),
            ::testing::Values('n'                                           // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
            , 'c'                                                            // this option is BLIS-api specific.
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(32),            // size n, for L32
                              gtint_t(16),            // L16
                              gtint_t(8),             // L8
                              gtint_t(4),             // L4
                              gtint_t(3),             // LScalar
                              // Testing the loops with combinations
                              gtint_t(160),           // 5*L32
                              gtint_t(192),           // 5*L32 + L16
                              gtint_t(200),           // 5*L32 + L16 + L8
                              gtint_t(204),           // 5*L32 + L16 + L8 + L4
                              gtint_t(207)),          // 5*L32 + L16 + L8 + L4 + 1(LScalar)
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(false, true)            // is_memory_test
        ),
        ::copyvUKRPrint<ccopyv_ker_ft>()
    );
#endif

// Unit testing with Non-Unit Strides(US), across all loops.
#ifdef K_bli_ccopyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_ccopyv_zen_int_nonUnitStrides,
        ccopyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ccopyv_zen_int),
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
        ::copyvUKRPrint<ccopyv_ker_ft>()
    );
#endif
#endif
