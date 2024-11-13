/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
   Portions of this file consist of AI-generated content.

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
#include "test_axpyf_ukr.h"
#include "common/blis_version_defs.h"

using T = double;
using FT = daxpyf_ker_ft;

class daxpyfGeneric :
        public ::testing::TestWithParam<std::tuple<FT,   // Function pointer type for daxpyf kernels
                                                   char,            // conjA
                                                   char,            // conjx
                                                   gtint_t,         // m
                                                   gtint_t,         // b_fuse
                                                   T,               // alpha
                                                   gtint_t,         // inca
                                                   gtint_t,         // lda_inc
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(daxpyfGeneric);

// Tests using random integers as vector elements.
TEST_P( daxpyfGeneric, UKR )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    FT ukr_fp = std::get<0>(GetParam());
    // denotes conjugate for A
    char conjA = std::get<1>(GetParam());
    // denotes conjugate for x
    char conjx = std::get<2>(GetParam());
    // rows of matrix
    gtint_t m = std::get<3>(GetParam());
    // fuse factor
    gtint_t b_fuse = std::get<4>(GetParam());
    // alpha
    T alpha = std::get<5>(GetParam());
    // stride size for A
    gtint_t inca = std::get<6>(GetParam());
    // lda_inc for A
    gtint_t lda_inc = std::get<7>(GetParam());
    // stride size for x
    gtint_t incx = std::get<8>(GetParam());
    // stride size for y
    gtint_t incy = std::get<9>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<10>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpyf.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (alpha == testinghelpers::ONE<T>())
    {
       // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 4.0;
#endif

        thresh = adj*(2*b_fuse)*testinghelpers::getEpsilon<T>();
    }
    else
    {
       // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 2.0;
#else
        double adj = 4.7;
#endif
        thresh = adj*(3*b_fuse)*testinghelpers::getEpsilon<T>();
    }

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpyf_ukr<T, FT>( ukr_fp, conjA, conjx, m, b_fuse, alpha, inca, lda_inc, incx, incy, thresh, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_daxpyf_zen_int_avx512 kernel.
*/
// Unit testing with unit strides, across all fuse-factors.
#ifdef K_bli_daxpyf_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_daxpyf_zen_int_avx512_unitStrides,
        daxpyfGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpyf_zen_int_avx512),       // kernel address
            ::testing::Values('n'),     // use x, not conj(x) (since it is real)
            ::testing::Values('n'),     // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(1),
                              gtint_t(3),
                              gtint_t(5),
                              gtint_t(8),
                              gtint_t(16),
                              gtint_t(32),
                              gtint_t(55)),
            ::testing::Values(// b_fuse
                              gtint_t(2),   // bli_daxpyf_zen_int2_avx512
                              gtint_t(4),   // bli_daxpyf_zen_int4_avx512
                              gtint_t(6),   // bli_daxpyf_zen_int6_avx512
                              gtint_t(8),   // bli_daxpyf_zen_int8_avx512
                              gtint_t(12),  // bli_daxpyf_zen_int12_avx512
                              gtint_t(16),  // bli_daxpyf_zen_int16_avx512
                              gtint_t(32),  // bli_daxpyf_zen_int32_avx512
                              gtint_t(30),  // Combination of fuse factors 16, 8, 6
                              gtint_t(28),  // Combination of fuse factors 16, 8, 4
                              gtint_t(26)   // Combination of fuse factors 16, 8, 2
                             ),
            ::testing::Values( -2.1, -1.0, 0.0, 1.0, 2.1 ), // alpha
            ::testing::Values(gtint_t(1)),                  // inca
            ::testing::Values(gtint_t(0), gtint_t(1)),      // lda_inc
            ::testing::Values(gtint_t(1)),                  // stride size for x
            ::testing::Values(gtint_t(1)),                  // stride size for y
            ::testing::Values(false, true)                  // is_memory_test
        ),
        (::axpyfUkrPrint<T, FT>())
    );
#endif

// Unit testing with non-unit strides, across all fuse-factors.
#ifdef K_bli_daxpyf_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_daxpyf_zen_int_avx512_nonUnitStrides,
        daxpyfGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpyf_zen_int_avx512),       // kernel address
            ::testing::Values('n'),     // use x, not conj(x) (since it is real)
            ::testing::Values('n'),     // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(15), gtint_t(27)),    // for size n
            ::testing::Values(// b_fuse
                              gtint_t(2),   // bli_daxpyf_zen_int2_avx512
                              gtint_t(4),   // bli_daxpyf_zen_int4_avx512
                              gtint_t(6),   // bli_daxpyf_zen_int6_avx512
                              gtint_t(8),   // bli_daxpyf_zen_int8_avx512
                              gtint_t(16),  // bli_daxpyf_zen_int16_avx512
                              gtint_t(32)   // bli_daxpyf_zen_int32_avx512
                             ),
            ::testing::Values( -2.1, 0.0, 1.0, 2.1 ),   // alpha
            ::testing::Values(gtint_t(2)),              // inca
            ::testing::Values(gtint_t(3)),              // lda_inc
            ::testing::Values(gtint_t(2)),              // stride size for x
            ::testing::Values(gtint_t(3)),              // stride size for y
            ::testing::Values(false, true)              // is_memory_test
        ),
        (::axpyfUkrPrint<T, FT>())
    );
#endif
#endif
