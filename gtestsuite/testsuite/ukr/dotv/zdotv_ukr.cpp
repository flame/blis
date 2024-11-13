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
#include "test_dotv_ukr.h"
#include "common/blis_version_defs.h"

using T = dcomplex;
class zdotvGeneric :
        public ::testing::TestWithParam<std::tuple<zdotv_ker_ft,    // Function pointer for ddotv kernels
                                                   char,            // conjx
                                                   char,            // conjy
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zdotvGeneric);

// Tests using random integers as vector elements.
TEST_P( zdotvGeneric, UKR )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the kernel to be tested:
    zdotv_ker_ft ukr = std::get<0>(GetParam());
    // denotes whether vec x is n,c
    char conjx = std::get<1>(GetParam());
    // denotes whether vec y is n,c
    char conjy = std::get<2>(GetParam());
    // vector length:
    gtint_t n = std::get<3>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<4>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<5>(GetParam());
    // enable/disable memory test:
    bool is_memory_test = std::get<6>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite level1/dotv/dotv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = 2*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_dotv_ukr<T>( ukr, conjx, conjy, n, incx, incy, thresh, is_memory_test );
}

// ----------------------------------------------
// -----  Begin ZEN4 (AVX512) Kernel Tests  -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
// Tests for bli_zdotv_zen_int_avx512 (AVX512) kernel.
/**
 * Loops & If conditions:
 * L32     - Main loop, handles 32 elements
 * L16     - handles 16 elements
 * L12     - handles 12 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * LFringe - handles upto 4 leftover elements
 *
 * LNUnit  - loop for non-unit increments
*/
#ifdef K_bli_zdotv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_zdotv_zen_int_avx512_unitStride,
        zdotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zdotv_zen_int_avx512),
            // conj(x): use n (no_conjugate) or c (conjugate).
            ::testing::Values('n', 'c'),
            // conj(y): use n (no_conjugate) or c (conjugate).
            ::testing::Values('n', 'c'),
            // m: size of vector.
            ::testing::Values(
                               // Individual Loop Tests
                               // testing each loop and if individually.
                               gtint_t(64),     // L32, executed twice
                               gtint_t(32),     // L32
                               gtint_t(16),     // L16
                               gtint_t(12),     // L12
                               gtint_t( 8),     // L8
                               gtint_t( 4),     // LFringe
                               gtint_t( 3),     // LFringe
                               gtint_t( 2),     // LFringe
                               gtint_t( 1),     // LFringe

                               // Waterfall Tests
                               // testing the entire set of loops and ifs.
                               gtint_t(92),     // L32 * 2 + L16 + L12
                               gtint_t(91),     // L32 * 2 + L16 + L8 + L4 + LFringe * 3
                               gtint_t(79)      // L32 * 2 + L12 + LFringe
            ),
            // incx: stride of x vector.
            ::testing::Values(
                               gtint_t(1)       // unit stride
            ),
            // incy: stride of y vector.
            ::testing::Values(
                               gtint_t(1)       // unit stride
            ),
            // is_memory_test: enable/disable memory tests
            ::testing::Values( false, true )
        ),
        ::dotvUKRPrint<zdotv_ker_ft>()
    );
#endif

#ifdef K_bli_zdotv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_zdotv_zen_int_avx512_nonUnitPositiveStrides,
        zdotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zdotv_zen_int_avx512),
            // conj(x): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // conj(y): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // incy: stride of y vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // is_memory_test: enable/disable memory tests
            ::testing::Values( false, true )
        ),
        ::dotvUKRPrint<zdotv_ker_ft>()
    );
#endif

// Tests for bli_zdotv_zen_int_avx512 (AVX512) kernel.
/**
 * Loops & If conditions:
 * L32     - Main loop, handles 32 elements
 * L16     - handles 16 elements
 * L12     - handles 12 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * LFringe - handles upto 4 leftover elements
 *
 * LNUnit  - loop for non-unit increments
*/
#ifdef K_bli_zdotv_zen4_asm_avx512
INSTANTIATE_TEST_SUITE_P(
        DISABLED_bli_zdotv_zen4_asm_avx512_unitStride,
        zdotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zdotv_zen4_asm_avx512),
            // conj(x): use n (no_conjugate) or c (conjugate).
            ::testing::Values('n', 'c'),
            // conj(y): use n (no_conjugate) or c (conjugate).
            ::testing::Values('n', 'c'),
            // m: size of vector.
            ::testing::Values(
                               // Individual Loop Tests
                               // testing each loop and if individually.
                               gtint_t(64),     // L40, executed twice
                               gtint_t(32),     // L40
                               gtint_t(16),     // L16
                               gtint_t(12),     // L12
                               gtint_t( 8),     // L8
                               gtint_t( 4),     // LFringe
                               gtint_t( 3),     // LFringe
                               gtint_t( 2),     // LFringe
                               gtint_t( 1),     // LFringe

                               // Waterfall Tests
                               // testing the entire set of loops and ifs.
                               gtint_t(92),     // L32 * 2 + L16 + L12
                               gtint_t(91),     // L32 * 2 + L16 + L8 + L4 + LFringe * 3
                               gtint_t(79)      // L32 * 2 + L12 + LFringe
            ),
            // incx: stride of x vector.
            ::testing::Values(
                               gtint_t(1)       // unit stride
            ),
            // incy: stride of y vector.
            ::testing::Values(
                               gtint_t(1)       // unit stride
            ),
            // is_memory_test: enable/disable memory tests
            ::testing::Values( false, true )
        ),
        ::dotvUKRPrint<zdotv_ker_ft>()
    );
#endif

#ifdef K_bli_zdotv_zen4_asm_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_zdotv_zen4_asm_avx512_nonUnitPositiveStrides,
        zdotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zdotv_zen4_asm_avx512),
            // conj(x): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // conj(y): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // incy: stride of y vector.
            ::testing::Values(
                               gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // is_memory_test: enable/disable memory tests
            ::testing::Values( false, true )
        ),
        ::dotvUKRPrint<zdotv_ker_ft>()
    );
#endif
#endif
// ----------------------------------------------
// -----   End ZEN4 (AVX512) Kernel Tests   -----
// ----------------------------------------------
