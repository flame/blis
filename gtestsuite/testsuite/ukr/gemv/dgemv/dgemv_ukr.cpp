/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "common/blis_version_defs.h"
#include <gtest/gtest.h>
#include "ukr/gemv/test_gemv_ukr.h"
#include "level2/gemv/test_gemv.h"

using T = double;

class dgemvGeneric :
        public ::testing::TestWithParam<std::tuple<dgemv_ker_ft,
                                                   char,        // storage format
                                                   char,        // transa
                                                   char,        // conjx
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   T,           // alpha
                                                   T,           // beta
                                                   gtint_t,     // incx
                                                   gtint_t,     // incy
                                                   gtint_t,     // lda_inc
                                                   bool>> {};   // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemvGeneric);

TEST_P( dgemvGeneric, UKR )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    dgemv_ker_ft ukr_fp = std::get<0>(GetParam());
    // matrix storage format(row major, column major)
    char storage = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<3>(GetParam());
    // matrix size m
    gtint_t m  = std::get<4>(GetParam());
    // matrix size n
    gtint_t n  = std::get<5>(GetParam());
    // specifies alpha value
    T alpha = std::get<6>(GetParam());
    // specifies beta value
    T beta = std::get<7>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<8>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<9>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<10>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<11>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gemv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() && (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        if(( transa == 'n' ) || ( transa == 'N' ))
            thresh = (3*n+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = (3*m+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemv_ukr<T, dgemv_ker_ft>( ukr_fp, storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Unit-tests
#ifdef K_bli_dgemv_t_zen_int_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_primary_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Range( gtint_t(1), gtint_t(16), gtint_t(1)),         // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x (non-unit incx is handled by frame)
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx7_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx7_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx7_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(7)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx6_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx6_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx6_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(6)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx5_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx5_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx5_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(5)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx4_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx4_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx4_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(4)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx3_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx3_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx3_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(3)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx2_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx2_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx2_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(2)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif

#ifdef K_bli_dgemv_t_zen_int_mx1_avx2
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx1_zen,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx1_avx2),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalones
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(4),                                  // L4
                           gtint_t(3),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(80),                                 // 5 * L16
                           gtint_t(88),                                 // 5 * L16 + L8
                           gtint_t(92),                                 // 5 * L16 + L8 + L4
                           gtint_t(95)),                                // 5 * L16 + L8 + L4 + Lfringe
        ::testing::Values( gtint_t(1)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
    );
#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
// Unit-tests
#ifdef K_bli_dgemv_t_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_primary_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Range( gtint_t(1), gtint_t(16), gtint_t(1)),         // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x (non-unit incx is handled by frame)
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx7_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx7_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx7_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(7)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx6_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx6_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx6_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(6)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx5_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx5_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx5_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(5)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx4_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx4_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx4_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(4)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx3_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx3_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx3_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(3)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx2_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx2_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx2_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalone
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(2)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif

#ifdef K_bli_dgemv_t_zen_int_mx1_avx512
INSTANTIATE_TEST_SUITE_P(
    dgemv_t_mx1_zen4,
    dgemvGeneric,
    ::testing::Combine(
        ::testing::Values(bli_dgemv_t_zen_int_mx1_avx512),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( // Testing the loops standalones
                           gtint_t(32),                                 // L32
                           gtint_t(16),                                 // L16
                           gtint_t(8),                                  // L8
                           gtint_t(7),                                  // Lfringe
                           // Testing the loops in combinations
                           gtint_t(160),                                // 5 * L32
                           gtint_t(176),                                // 5 * L32 + L16
                           gtint_t(184),                                // 5 * L32 + L16 + L8
                           gtint_t(191)),                               // 5 * L32 + L16 + L8 + Lfringe
        ::testing::Values( gtint_t(1)),                                 // n
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // alpha
        ::testing::Values( double(0.0), double(1.0), double(2.0)),      // beta
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1), gtint_t(3)),                      // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(7)),                      // increment to the leading dim of a
        ::testing::Values(false, true)                                  // is_memory_test
    ),
    (::gemvUKRPrint<double, dgemv_ker_ft>())
);
#endif
#endif
