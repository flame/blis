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
#include "test_scalv_ukr.h"

class sscalvGeneric :
        public ::testing::TestWithParam<std::tuple<sscalv_ker_ft,   // Function pointer for sscalv kernels
                                                   char,            // conj_alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   float,           // alpha
                                                   bool>> {};       // is_memory_test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sscalvGeneric);

// Tests using random integers as vector elements.
TEST_P( sscalvGeneric, UKR )
{
    using T = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the kernel to be tested:
    sscalv_ker_ft ukr = std::get<0>(GetParam());
    // denotes whether alpha or conj(alpha) will be used:
    char conj_alpha = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // alpha:
    T alpha = std::get<4>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scalv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    float thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scalv_ukr<T, T, sscalv_ker_ft>( ukr, conj_alpha, n, incx, alpha, thresh, is_memory_test );
}

// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Tests for bli_sscalv_zen_int (AVX2) kernel.
/**
 * Loops:
 * L32     - Main loop, handles 32 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
INSTANTIATE_TEST_SUITE_P(
        bli_sscalv_zen_int_unitPositiveStride,
        sscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(32),       // L32
                                gtint_t(15),       // LScalar
                                gtint_t(96),       // 3*L32
                                gtint_t(111)       // 3*L32 + 15(LScalar)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)         // unit stride
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                // @note: disabling alpha = 0 test for bli_sscalv_zen_int.
                                //  Segmentation Fault is being observed for alpha = 0 since the
                                //  kernel isn't handling the condition where cntx = NULL.
                                // float( 0.0),
                                float( 7.0),
                                float(-3.0)
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<float,sscalv_ker_ft>())
    );

INSTANTIATE_TEST_SUITE_P(
        bli_sscalv_zen_int_nonUnitPositiveStrides,
        sscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                // @note: disabling alpha = 0 test for bli_sscalv_zen_int.
                                //  Segmentation Fault is being observed for alpha = 0 since the
                                //  kernel isn't handling the condition where cntx = NULL.
                                // float( 0.0),
                                float( 7.0),
                                float(-3.0)
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<float,sscalv_ker_ft>())
    );

// Tests for bli_sscalv_zen_int10 (AVX2) kernel.
/**
 * Cases and Loops:
 * C0 L128    - Main loop, handles 128 elements
 * C0 L96     - handles 96 elements
 * C1 L48     - handles 48 elements
 * C2 L24     - handles 24 elements
 * C2 L8      - handles 8 elements
 * C2 LScalar - leftover loop
 *
 * The switch cases are cascading, and the order
 * is C0 --> C1 --> C2
 *
 * LNUnit - loop for non-unit increments
*/
INSTANTIATE_TEST_SUITE_P(
        bli_sscalv_zen_int10_unitPositiveStride,
        sscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sscalv_zen_int10),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                // testing case 0 (n >= 500)
                                gtint_t(512),       // C0 4*L128
                                gtint_t(608),       // C0 4*L128 + C1 L96
                                gtint_t(599),       // C0 4*L128 + C2 (L48 + L24 + L8 + 7(LSscalar))
                                gtint_t(623),       // C0 4*L128 + C1 L96 + C2 (L8 + 7(LScalar))

                                // testing case 1 (300 <= n < 500)
                                gtint_t(384),       // C1 4*L96
                                gtint_t(432),       // C1 4*L96 + C2 L48
                                gtint_t(456),       // C1 4*L96 + C2 (L48 + L24)
                                gtint_t(464),       // C1 4*L96 + C2 (L48 + L24 + L8)
                                gtint_t(471),       // C1 4*L96 + C2 (L48 + L24 + L8 + 7(LScalar))

                                // testing case 2 (n < 300)
                                gtint_t(192),       // C2 4*L48
                                gtint_t(216),       // C2 (4*L48 + L24)
                                gtint_t(224),       // C2 (4*L48 + L24 + L8)
                                gtint_t(231)        // C2 (4*L48 + L24 + L8 + 7(LScalar))
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)      // unit stride
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 0.0),
                                float( 7.0),
                                float(-3.0)
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<float,sscalv_ker_ft>())
    );

INSTANTIATE_TEST_SUITE_P(
        bli_sscalv_zen_int10_nonUnitPositiveStrides,
        sscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_sscalv_zen_int10),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                float( 0.0),
                                float( 7.0),
                                float(-3.0)
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<float,sscalv_ker_ft>())
    );
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------
