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
#include "test_amaxv_ukr.h"
#include "common/blis_version_defs.h"

class damaxvGeneric :
        public ::testing::TestWithParam<std::tuple<damaxv_ker_ft,   // Function pointer type for damaxv kernels
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(damaxvGeneric);

// Tests using random integers as vector elements.
TEST_P( damaxvGeneric, UKR )
{
    using T = double;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    damaxv_ker_ft ukr_fp = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<3>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_amaxv_ukr<T, damaxv_ker_ft>( ukr_fp, n, incx, thresh, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_damaxv_zen_int kernel.
    The code structure for bli_damaxv_zen_int( ... ) is as follows :

    bli_damaxv_zen_int() --> bli_vec_absmax_double() --> bli_vec_search_double()
    bli_vec_absmax_double() structure:
    For unit strides :
        Main loop    :  In blocks of 48 --> L48
        Fringe loops :  In blocks of 32 --> L32
                        In blocks of 16 --> L16
                        In blocks of 8  --> L8
                        In blocks of 4  --> L4
                        In blocks of 2  --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.

    bli_vec_search_double() structure:
    For unit strides :
        Main loop    :  In blocks of 4 --> L4
                        In blocks of 2 --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
#ifdef K_bli_damaxv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_damaxv_zen_int_unitStrides,
        damaxvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_damaxv_zen_int),   // kernel address
            ::testing::Values(gtint_t(48),           // for size n, L48
                              gtint_t(32),           // L32
                              gtint_t(16),           // L16
                              gtint_t(8),            // L8
                              gtint_t(4),            // L4
                              gtint_t(2),            // L2
                              gtint_t(1),            // LScalar
                              gtint_t(144),          // 3*L48
                              gtint_t(176),          // 3*L48 + L32
                              gtint_t(175)),         // 3*L48 + L16 + L8 + L4 + L2 + LScalar
            ::testing::Values(gtint_t(1)),           // incx
            ::testing::Values(false, true)           // is_memory_test
        ),
        ::amaxvUKRPrint<damaxv_ker_ft>()
    );
#endif

// Unit testing with non-unit strides.
#ifdef K_bli_damaxv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_damaxv_zen_int_nonUnitStrides,
        damaxvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_damaxv_zen_int),   // kernel address
            ::testing::Values(gtint_t(10),           // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),           // incx
            ::testing::Values(false, true)           // is_memory_test
        ),
        ::amaxvUKRPrint<damaxv_ker_ft>()
    );
#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_damaxv_zen_int_avx512 kernel.
    The code structure for bli_damaxv_zen_int_avx512( ... ) is as follows :

    For unit strides :
        Main loop    :  In blocks of 64 --> L64
        Fringe loops :  In blocks of 32 --> L32
                        In blocks of 16 --> L16
                        In blocks of 8  --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
#ifdef K_bli_damaxv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_damaxv_zen_int_avx512_unitStrides,
        damaxvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_damaxv_zen_int_avx512),   // kernel address
            ::testing::Values(gtint_t(64),                  // for size n, L64
                              gtint_t(32),                  // L32
                              gtint_t(16),                  // L16
                              gtint_t(8),                   // L8
                              gtint_t(7),                   // 7(LScalar)
                              gtint_t(192),                 // 3*L64
                              gtint_t(255)),                // 3*L64 + L32 + L16 + L8 + 7(LScalar)
            ::testing::Values(gtint_t(1)),                  // incx
            ::testing::Values(false, true)                  // is_memory_test
        ),
        ::amaxvUKRPrint<damaxv_ker_ft>()
    );
#endif

#ifdef K_bli_damaxv_zen_int_avx512
// Unit testing with non-unit strides.
INSTANTIATE_TEST_SUITE_P(
        bli_damaxv_zen_int_avx512_nonUnitStrides,
        damaxvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_damaxv_zen_int_avx512),   // kernel address
            ::testing::Values(gtint_t(10),                  // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                  // incx
            ::testing::Values(false, true)                  // is_memory_test
        ),
        ::amaxvUKRPrint<damaxv_ker_ft>()
    );
#endif
#endif
