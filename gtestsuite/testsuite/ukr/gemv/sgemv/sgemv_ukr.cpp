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

using T = float;

class sgemvGenericKernel :
        public ::testing::TestWithParam<std::tuple<sgemv_ker,
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

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemvGenericKernel);

TEST_P( sgemvGenericKernel, UKR )
{
    sgemv_ker ukr_fp = std::get<0>(GetParam());
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

    test_gemv_ukr_transa<T, sgemv_ker>( ukr_fp, storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test );
}

// =============================================================================
// ZEN (AVX2) kernels
// =============================================================================
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

// T-direction entry-point: bli_sgemv_t_zen_int
#ifdef K_bli_sgemv_t_zen_int
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen_int,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen_int),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(24),
                           gtint_t(25), gtint_t(71), gtint_t(144) ),   // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(4),
                           gtint_t(7), gtint_t(8), gtint_t(20) ),      // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// T-direction ST caller: bli_sgemv_t_zen_int_24x4
#ifdef K_bli_sgemv_t_zen_int_24x4
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen_int_24x4,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen_int_24x4),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(24),
                           gtint_t(25), gtint_t(71), gtint_t(144) ),   // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(6), gtint_t(12) ),      // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1) ),                   // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// T-direction MT wrapper: bli_sgemv_t_zen_int_24x4_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_t_zen_int_24x4_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen_int_24x4_mt,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen_int_24x4_mt),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(24),
                           gtint_t(71), gtint_t(192), gtint_t(384) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(4), gtint_t(10),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction entry-point: bli_sgemv_n_zen_int
#ifdef K_bli_sgemv_n_zen_int
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen_int,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen_int),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(41), gtint_t(119), gtint_t(240) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(5), gtint_t(15) ),      // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy (non-unit handled by frame)
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction ST caller: bli_sgemv_n_zen_int_40x4 (MR=40, NR=1)
#ifdef K_bli_sgemv_n_zen_int_40x4
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen_int_40x4,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen_int_40x4),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(41), gtint_t(119), gtint_t(240) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(6), gtint_t(10) ),      // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction MT wrapper: bli_sgemv_n_zen_int_40x4_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_n_zen_int_40x4_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen_int_40x4_mt,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen_int_40x4_mt),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(119), gtint_t(240), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(4), gtint_t(10),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction ST caller: bli_sgemv_m_zen_int_40x4 (MR=40, NR=4)
#ifdef K_bli_sgemv_m_zen_int_40x4
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen_int_40x4,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen_int_40x4),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(41), gtint_t(119), gtint_t(240) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(3),
                           gtint_t(4), gtint_t(6), gtint_t(10) ),      // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction MT Mdiv: bli_sgemv_m_zen_int_40x4_mt_Mdiv
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_m_zen_int_40x4_mt_Mdiv)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen_int_40x4_mt_Mdiv,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen_int_40x4_mt_Mdiv),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(119), gtint_t(240), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction MT Ndiv: bli_sgemv_m_zen_int_40x4_mt_Ndiv
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_m_zen_int_40x4_mt_Ndiv)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen_int_40x4_mt_Ndiv,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen_int_40x4_mt_Ndiv),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(40),
                           gtint_t(119), gtint_t(240), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

#endif // BLIS_KERNELS_ZEN && GTEST_AVX2FMA3

// =============================================================================
// ZEN4 (AVX-512) kernels
// =============================================================================
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)

// T-direction entry-point: bli_sgemv_t_zen4_int
#ifdef K_bli_sgemv_t_zen4_int
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen4_int,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen4_int),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(48),
                           gtint_t(49), gtint_t(143), gtint_t(288) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(7),
                           gtint_t(8), gtint_t(20), gtint_t(50) ),     // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(8) ),                   // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// T-direction ST caller: bli_sgemv_t_zen4_int_48x8
#ifdef K_bli_sgemv_t_zen4_int_48x8
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen4_int_48x8,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen4_int_48x8),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(48),
                           gtint_t(49), gtint_t(143), gtint_t(288) ),  // m
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(7),
                           gtint_t(8), gtint_t(12), gtint_t(20) ),     // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1) ),                   // incx
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// T-direction MT wrapper: bli_sgemv_t_zen4_int_48x8_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_t_zen4_int_48x8_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_t_zen4_int_48x8_mt,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_t_zen4_int_48x8_mt),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('t'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(48),
                           gtint_t(143), gtint_t(384), gtint_t(768) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction entry-point: bli_sgemv_n_zen4_int
#ifdef K_bli_sgemv_n_zen4_int
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen4_int,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen4_int),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(8), gtint_t(15),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(5), gtint_t(12),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy (non-unit handled by frame)
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction ST caller: bli_sgemv_n_zen4_int_80x8 (MR=80, NR=8)
#ifdef K_bli_sgemv_n_zen4_int_80x8
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen4_int_80x8,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen4_int_80x8),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(80),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(2), gtint_t(6),
                           gtint_t(16), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// N-direction MT wrapper: bli_sgemv_n_zen4_int_80x8_mt
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_n_zen4_int_80x8_mt)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_n_zen4_int_80x8_mt,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_n_zen4_int_80x8_mt),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(80),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction ST caller: bli_sgemv_m_zen4_int_80x8
#ifdef K_bli_sgemv_m_zen4_int_80x8
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen4_int_80x8,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen4_int_80x8),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(80),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction MT Ndiv: bli_sgemv_m_zen4_int_80x8_mt_Ndiv
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_m_zen4_int_80x8_mt_Ndiv)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen4_int_80x8_mt_Ndiv,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen4_int_80x8_mt_Ndiv),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(80),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

// M-direction MT Mdiv: bli_sgemv_m_zen4_int_80x8_mt_Mdiv
#if defined(BLIS_ENABLE_OPENMP) && defined(K_bli_sgemv_m_zen4_int_80x8_mt_Mdiv)
INSTANTIATE_TEST_SUITE_P(
    bli_sgemv_m_zen4_int_80x8_mt_Mdiv,
    sgemvGenericKernel,
    ::testing::Combine(
        ::testing::Values(bli_sgemv_m_zen4_int_80x8_mt_Mdiv),
        ::testing::Values('c'),                                         // storage format
        ::testing::Values('n'),                                         // transa
        ::testing::Values('n'),                                         // conjx
        ::testing::Values( gtint_t(1), gtint_t(3), gtint_t(80),
                           gtint_t(116), gtint_t(318), gtint_t(462) ), // m
        ::testing::Values( gtint_t(1), gtint_t(6), gtint_t(16),
                           gtint_t(84), gtint_t(132), gtint_t(271) ),  // n
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // alpha
        ::testing::Values( float(0.0f), float(1.0f), float(2.0f) ),   // beta
        ::testing::Values( gtint_t(1), gtint_t(3) ),                   // incx
        ::testing::Values( gtint_t(1) ),                               // incy
        ::testing::Values( gtint_t(0), gtint_t(7) ),                   // lda_inc
        ::testing::Values( false, true )                               // is_memory_test
    ),
    (::gemvUKRPrint<float, sgemv_ker>())
);
#endif

#endif // BLIS_KERNELS_ZEN4 && GTEST_AVX512
