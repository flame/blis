/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

using T = scomplex;

class cgemvGenericKernel :
        public ::testing::TestWithParam<std::tuple<cgemv_ker,
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

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cgemvGenericKernel);

TEST_P( cgemvGenericKernel, UKR )
{
    cgemv_ker ukr_fp = std::get<0>(GetParam());
    char storage = std::get<1>(GetParam());
    char transa = std::get<2>(GetParam());
    char conjx = std::get<3>(GetParam());
    gtint_t m  = std::get<4>(GetParam());
    gtint_t n  = std::get<5>(GetParam());
    T alpha = std::get<6>(GetParam());
    T beta = std::get<7>(GetParam());
    gtint_t incx = std::get<8>(GetParam());
    gtint_t incy = std::get<9>(GetParam());
    gtint_t lda_inc = std::get<10>(GetParam());
    bool is_memory_test = std::get<11>(GetParam());

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

    test_gemv_ukr_transa<T, cgemv_ker>( ukr_fp, storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test );
}

// =============================================================================
// ZEN (AVX2) kernels
// =============================================================================
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

// T-direction entry-point: bli_cgemv_t_zen_int
#ifdef K_bli_cgemv_t_zen_int
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen_int,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen_int),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n', 'c'),                                                // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(21), gtint_t(59), gtint_t(120) ),               // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(4),
                           gtint_t(7), gtint_t(8), gtint_t(20) ),                  // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// T-direction ST caller: bli_cgemv_t_zen_int_20x4
#ifdef K_bli_cgemv_t_zen_int_20x4
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen_int_20x4,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen_int_20x4),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(21), gtint_t(59), gtint_t(120) ),               // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(6), gtint_t(12) ),                  // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1) ),                               // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// T-direction MT wrapper: bli_cgemv_t_zen_int_20x4_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_cgemv_t_zen_int_20x4_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen_int_20x4_mt,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen_int_20x4_mt),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(59), gtint_t(192), gtint_t(384) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(4), gtint_t(10),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction entry-point: bli_cgemv_n_zen_int
#ifdef K_bli_cgemv_n_zen_int
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen_int,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen_int),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n', 'c'),                                                // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(21), gtint_t(59), gtint_t(120) ),               // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(5), gtint_t(15) ),                  // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy (non-unit handled by frame)
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction ST caller: bli_cgemv_n_zen_int_20x5 (MR=20, NR=5)
#ifdef K_bli_cgemv_n_zen_int_20x5
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen_int_20x5,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen_int_20x5),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(21), gtint_t(59), gtint_t(120) ),               // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(5), gtint_t(6), gtint_t(10) ),                  // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction MT wrapper: bli_cgemv_n_zen_int_20x5_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_cgemv_n_zen_int_20x5_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen_int_20x5_mt,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen_int_20x5_mt),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(20),
                           gtint_t(59), gtint_t(192), gtint_t(384) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(4), gtint_t(10),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

#endif // BLIS_KERNELS_ZEN && GTEST_AVX2FMA3

// =============================================================================
// ZEN4 (AVX-512) kernels
// =============================================================================
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)

// T-direction entry-point: bli_cgemv_t_zen4_int
#ifdef K_bli_cgemv_t_zen4_int
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen4_int,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen4_int),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n', 'c'),                                                // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(41), gtint_t(119), gtint_t(240) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(7),
                           gtint_t(8), gtint_t(20), gtint_t(50) ),                 // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(8) ),                               // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// T-direction ST caller: bli_cgemv_t_zen4_int_40x8
#ifdef K_bli_cgemv_t_zen4_int_40x8
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen4_int_40x8,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen4_int_40x8),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(41), gtint_t(119), gtint_t(240) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(7),
                           gtint_t(8), gtint_t(12), gtint_t(20) ),                 // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1) ),                               // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// T-direction MT wrapper: bli_cgemv_t_zen4_int_40x8_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_cgemv_t_zen4_int_40x8_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_t_zen4_int_40x8_mt,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_t_zen4_int_40x8_mt),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('t', 'c'),                                                // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(119), gtint_t(384), gtint_t(768) ),             // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction entry-point: bli_cgemv_n_zen4_int
#ifdef K_bli_cgemv_n_zen4_int
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen4_int,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen4_int),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n', 'c'),                                                // conjx
        ::testing::Values( gtint_t(1), gtint_t(8), gtint_t(15),
                           gtint_t(58), gtint_t(159), gtint_t(231) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(5), gtint_t(12),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy (non-unit handled by frame)
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction ST caller: bli_cgemv_n_zen4_int_40x10 (MR=40, NR=10)
#ifdef K_bli_cgemv_n_zen4_int_40x10
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen4_int_40x10,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen4_int_40x10),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(58), gtint_t(159), gtint_t(231) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(6),
                           gtint_t(10), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

// N-direction MT wrapper: bli_cgemv_n_zen4_int_40x10_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_cgemv_n_zen4_int_40x10_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_cgemv_n_zen4_int_40x10_mt,
    cgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_cgemv_n_zen4_int_40x10_mt),
        ::testing::Values('c'),                                                     // storage format
        ::testing::Values('n'),                                                     // transa
        ::testing::Values('n'),                                                     // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(58), gtint_t(159), gtint_t(231) ),              // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),              // n
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // alpha
        ::testing::Values( scomplex{0.0f,0.0f}, scomplex{1.0f,0.0f},
                           scomplex{1.0f,2.0f} ),                                  // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                               // incx
        ::testing::Values( gtint_t(1) ),                                           // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                               // lda_inc
        ::testing::Values( false, true )                                           // is_memory_test
    ),
    (::gemvUKRPrint<scomplex, cgemv_ker>())
);
#endif

#endif // BLIS_KERNELS_ZEN4 && GTEST_AVX512
