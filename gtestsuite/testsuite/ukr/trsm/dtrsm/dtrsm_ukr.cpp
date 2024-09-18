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
#include "common/testing_helpers.h"
#include "level3/ref_gemm.h"
#include "ukr/trsm/test_trsm_ukr.h"
#include "level3/trsm/test_trsm.h"

class dtrsmGenericNat :
    public ::testing::TestWithParam<std::tuple< dgemmtrsm_ukr_ft,  // Function pointer type for dtrsm kernels
                                                char,              // storage
                                                char,              // uploa
                                                char,              // diaga
                                                gtint_t,           // m
                                                gtint_t,           // n
                                                gtint_t,           // k
                                                double,            // alpha
                                                gtint_t,           // ldc_inc
                                                bool      >> {};   // is_memory_test

class dtrsmGenericSmall :
    public ::testing::TestWithParam<std::tuple< trsm_small_ker_ft,  // Function pointer type for dtrsm kernels
                                                char,               // side
                                                char,               // uploa
                                                char,               // diaga
                                                char,               // transa
                                                gtint_t,            // m
                                                gtint_t,            // n
                                                double,             // alpha
                                                gtint_t,            // lda_inc
                                                gtint_t,            // ldb_inc
                                                bool      >> {};    // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dtrsmGenericNat);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dtrsmGenericSmall);

TEST_P( dtrsmGenericNat, native_kernel)
{
    using T = double;
    dgemmtrsm_ukr_ft ukr_fp = std::get<0>(GetParam());
    char storage            = std::get<1>(GetParam());
    char uploa              = std::get<2>(GetParam());
    char diaga              = std::get<3>(GetParam());
    gtint_t m               = std::get<4>(GetParam());
    gtint_t n               = std::get<5>(GetParam());
    gtint_t k               = std::get<6>(GetParam());
    T   alpha               = std::get<7>(GetParam());
    gtint_t ldc             = std::get<8>(GetParam());
    bool is_memory_test     = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trsm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = 3*m*testinghelpers::getEpsilon<T>();

    test_trsm_ukr<T, dgemmtrsm_ukr_ft>( ukr_fp, storage, uploa, diaga, m, n, k, alpha, ldc, thresh, is_memory_test );
}

TEST_P( dtrsmGenericSmall, small_kernel)
{
    using T = double;
    trsm_small_ker_ft ukr_fp  = std::get<0>(GetParam());
    char side                 = std::get<1>(GetParam());
    char uploa                = std::get<2>(GetParam());
    char diaga                = std::get<3>(GetParam());
    char transa               = std::get<4>(GetParam());
    gtint_t m                 = std::get<5>(GetParam());
    gtint_t n                 = std::get<6>(GetParam());
    T   alpha                 = std::get<7>(GetParam());
    gtint_t lda               = std::get<8>(GetParam());
    gtint_t ldb               = std::get<9>(GetParam());
    bool is_memory_test       = std::get<10>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite trsm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        thresh = 3*m*testinghelpers::getEpsilon<T>();

    test_trsm_small_ukr<T, trsm_small_ker_ft>( ukr_fp, side, uploa, diaga, transa, m, n, alpha, lda, ldb, thresh, is_memory_test, BLIS_DOUBLE);
}

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_l_zen4_asm_8x24,
    dtrsmGenericNat,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_l_zen4_asm_8x24),  // ker_ptr
        ::testing::Values('c', 'r', 'g'),                  // stor
        ::testing::Values('l'),                            // uplo
        ::testing::Values('u', 'n'),                       // diaga
        ::testing::Values(8),                              // m
        ::testing::Values(24),                             // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),   // k
        ::testing::Values(-1, -5.2, 1, 8.9),               // alpha
        ::testing::Values(0, 9, 53),                       // ldc_inc
        ::testing::Values(false, true)                     // is_memory_test
    ),
    (::trsmNatUKRPrint<double,dgemmtrsm_ukr_ft>())
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_u_zen4_asm_8x24,
    dtrsmGenericNat,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_u_zen4_asm_8x24),  // ker_ptr
        ::testing::Values('c', 'r', 'g'),                  // stor
        ::testing::Values('u'),                            // uplo
        ::testing::Values('u', 'n'),                       // diaga
        ::testing::Values(8),                              // m
        ::testing::Values(24),                             // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),   // k
        ::testing::Values(-1, -5.2, 1, 8.9),               // alpha
        ::testing::Values(0, 9, 53),                       // ldc_inc
        ::testing::Values(false, true)                     // is_memory_test
   ),
    (::trsmNatUKRPrint<double,dgemmtrsm_ukr_ft>())
);

INSTANTIATE_TEST_SUITE_P (
    bli_trsm_small_AVX512,
    dtrsmGenericSmall,
    ::testing::Combine(
        ::testing::Values(bli_trsm_small_AVX512),     // ker_ptr
        ::testing::Values('l', 'r'),                  // side
        ::testing::Values('l', 'u'),                  // uplo
        ::testing::Values('n', 'u'),                  // diaga
        ::testing::Values('n', 't'),                  // transa
        ::testing::Range(gtint_t(1), gtint_t(9), 1),  // m
        ::testing::Range(gtint_t(1), gtint_t(9), 1),  // n
        ::testing::Values(-3, 3),                     // alpha
        ::testing::Values(0, 10),                     // lda_inc
        ::testing::Values(0, 10),                     // ldb_inc
        ::testing::Values(false, true)                // is_memory_test
    ),
    (::trsmSmallUKRPrint<double,trsm_small_ker_ft>())
);
#endif


#if defined(BLIS_KERNELS_HASWELL) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_l_haswell_asm_6x8,
    dtrsmGenericNat,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_l_haswell_asm_6x8), // ker_ptr
        ::testing::Values('c', 'r', 'g'),                   // stor
        ::testing::Values('l'),                             // uplo
        ::testing::Values('u', 'n'),                        // diaga
        ::testing::Values(6),                               // m
        ::testing::Values(8),                               // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),    // k
        ::testing::Values(-1, -5.2, 1, 8.9),                // alpha
        ::testing::Values(0, 9, 53),                        // ldc_inc
        ::testing::Values(false, true)                      // is_memory_test
    ),
    (::trsmNatUKRPrint<double,dgemmtrsm_ukr_ft>())
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_u_haswell_asm_6x8,
    dtrsmGenericNat,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_u_haswell_asm_6x8), // ker_ptr
        ::testing::Values('c', 'r', 'g'),                   // stor
        ::testing::Values('u'),                             // uplo
        ::testing::Values('u', 'n'),                        // diaga
        ::testing::Values(6),                               // m
        ::testing::Values(8),                               // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),    // k
        ::testing::Values(-1, -5.2, 1, 8.9),                // alpha
        ::testing::Values(0, 9, 53),                        // ldc_inc
        ::testing::Values(false, true)                      // is_memory_test
    ),
    (::trsmNatUKRPrint<double,dgemmtrsm_ukr_ft>())
);
#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
INSTANTIATE_TEST_SUITE_P (
    bli_trsm_small,
    dtrsmGenericSmall,
    ::testing::Combine(
        ::testing::Values(bli_trsm_small),            // ker_ptr
        ::testing::Values('l', 'r'),                  // side
        ::testing::Values('l', 'u'),                  // uplo
        ::testing::Values('n', 'u'),                  // diaga
        ::testing::Values('n', 't'),                  // transa
        ::testing::Range(gtint_t(1), gtint_t(9), 1),  // m
        ::testing::Range(gtint_t(1), gtint_t(9), 1),  // n
        ::testing::Values(-3, 3),                     // alpha
        ::testing::Values(0, 10),                     // lda_inc
        ::testing::Values(0, 10),                     // ldb_inc
        ::testing::Values(false, true)                // is_memory_test
    ),
    (::trsmSmallUKRPrint<double,trsm_small_ker_ft>())
);
#endif
#endif
