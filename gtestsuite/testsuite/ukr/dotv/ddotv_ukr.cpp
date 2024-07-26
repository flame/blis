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

class ddotvGeneric :
        public ::testing::TestWithParam<std::tuple<ddotv_ker_ft,    // Function pointer for ddotv kernels
                                                   char,            // conjx
                                                   char,            // conjy
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ddotvGeneric);

// Tests using random integers as vector elements.
TEST_P( ddotvGeneric, UKR )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the kernel to be tested:
    ddotv_ker_ft ukr = std::get<0>(GetParam());
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
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Tests for bli_ddotv_zen_int (AVX2) kernel.
/**
 * Loops:
 * L16     - handles 16 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int_unitStride,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int),
            // conj(x): use n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // conj(y): use n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                               // testing each loop individually.
                               gtint_t(32),     // L16, executed twice
                               gtint_t(16),     // L16
                               gtint_t( 8),     // LScalar, executed 8 times
                               gtint_t( 1),     // LScalar

                               // testing entire set of loops.
                               gtint_t(33),     // L16 (executed twice) + LScalar
                               gtint_t(17),     // L16 and LScalar
                               gtint_t(18)      // L16 and LScalar (executed twice)
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int_nonUnitPositiveStrides,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int),
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );

// Tests for bli_ddotv_zen_int10 (AVX2) kernel.
/**
 * Loops:
 * L40     - Main loop, handles 40 elements
 * L20     - handles 20 elements
 * L16     - handles 16 elements
 * L8      - handles 8 elements
 * L4      - handles 4 elements
 * LScalar - leftover loop
 *
 * LNUnit  - loop for non-unit increments
*/
INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int10_unitStride,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int10),
            // conj(x): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // conj(y): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                               // testing each loop individually.
                               gtint_t(80),     // L40, executed twice
                               gtint_t(40),     // L40
                               gtint_t(20),     // L20
                               gtint_t(16),     // L16
                               gtint_t( 8),     // L8
                               gtint_t( 4),     // L4
                               gtint_t( 2),     // LScalar
                               gtint_t( 1),     // LScalar

                               // testing entire set of loops starting from loop m to n.
                               gtint_t(73),     // L40 through LScalar, excludes L16
                               gtint_t(33),     // L20 through LScalar, excludes L16
                               gtint_t(13),     // L8 through LScalar
                               gtint_t( 5),     // L4 through LScalar

                               // testing few combinations including L16.
                               gtint_t(77),     // L40 + L20 + L16 + LScalar
                               gtint_t(76),     // L40 + L20 + L16
                               gtint_t(57),     // L40 + L16 + LScalar
                               gtint_t(37)      // L20 + L16 + LScalar
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int10_nonUnitPositiveStrides,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int10),
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------


// ----------------------------------------------
// -----  Begin ZEN4 (AVX512) Kernel Tests  -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
// Tests for bli_ddotv_zen_int_avx512 (AVX512) kernel.
/**
 * Loops & If conditions:
 * L40     - Main loop, handles 40 elements
 * L16     - handles 16 elements
 * I8      - handles 8 elements
 * IScalar - handles upto 8 leftover elements
 *
 * LNUnit  - loop for non-unit increments
*/
INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int_avx512_unitStride,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int_avx512),
            // conj(x): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // conj(y): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                               // Individual Loop Tests
                               // testing each loop and if individually.
                               gtint_t(80),     // L40, executed twice
                               gtint_t(40),     // L40
                               gtint_t(16),     // L16
                               gtint_t( 8),     // I8
                               gtint_t( 7),     // IScalar
                               gtint_t( 6),     // IScalar
                               gtint_t( 5),     // IScalar
                               gtint_t( 4),     // IScalar
                               gtint_t( 3),     // IScalar
                               gtint_t( 2),     // IScalar
                               gtint_t( 1),     // IScalar

                               // Waterfall Tests
                               // testing the entire set of loops and ifs.
                               gtint_t(65)
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_ddotv_zen_int_avx512_nonUnitPositiveStrides,
        ddotvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_ddotv_zen_int_avx512),
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
        ::dotvUKRPrint<ddotv_ker_ft>()
    );
#endif
// ----------------------------------------------
// -----   End ZEN4 (AVX512) Kernel Tests   -----
// ----------------------------------------------
