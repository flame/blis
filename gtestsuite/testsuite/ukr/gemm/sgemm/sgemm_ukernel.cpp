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
#include "blis.h"
#include "common/testing_helpers.h"
#include "ukr/gemm/test_gemm_ukr.h"

/*******************************************************/
/*                 SUP Kernel testing                  */
/*******************************************************/
class sgemmGenericSUP :
        public ::testing::TestWithParam<std::tuple< gtint_t,                // m
                                                    gtint_t,                // n
                                                    gtint_t,                // k
                                                    float,                  // alpha
                                                    float,                  // beta
                                                    char,                   // storage of C matrix
                                                    sgemmsup_ker_ft,        // Function pointer type for sgemm kernel
                                                    gtint_t,                // micro-kernel MR block
                                                    char,                   // transa
                                                    char,                   // transb
                                                    bool,                   // row preferred kernel
                                                    bool                    // is_memory_test
                                                    >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmGenericSUP);

TEST_P( sgemmGenericSUP, functionality_testing)
{
    using T = float;
    gtint_t m                = std::get<0>(GetParam());                     // dimension m
    gtint_t n                = std::get<1>(GetParam());                     // dimension n
    gtint_t k                = std::get<2>(GetParam());                     // dimension k
    T alpha                  = std::get<3>(GetParam());                     // alpha
    T beta                   = std::get<4>(GetParam());                     // beta
    char storageC            = std::get<5>(GetParam());                     // storage scheme for C matrix
    sgemmsup_ker_ft kern_ptr = std::get<6>(GetParam());                     // pointer to the gemm kernel
    gtint_t MR               = std::get<7>(GetParam());                     // Micro-kernel tile size
    char transa              = std::get<8>(GetParam());                     // transa
    char transb              = std::get<9>(GetParam());                     // transb
    bool row_pref            = std::get<10>(GetParam());                    // kernel transpose
    bool is_memory_test      = std::get<11>(GetParam());                    // memory test

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
              beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    test_gemmsup_ukr(kern_ptr, transa, transb, m, n, k, alpha, beta, storageC, MR, row_pref, thresh, is_memory_test);

}// end of function


class sgemmGenericSUPPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, float, float, char,
                                          sgemmsup_ker_ft, gtint_t, char, char, bool, bool>>  str) const {

        gtint_t m           = std::get<0>(str.param);
        gtint_t n           = std::get<1>(str.param);
        gtint_t k           = std::get<2>(str.param);
        float alpha         = std::get<3>(str.param);
        float beta          = std::get<4>(str.param);
        char storageC       = std::get<5>(str.param);
        char transa         = std::get<8>(str.param);
        char transb         = std::get<9>(str.param);
        bool is_memory_test = std::get<11>(str.param);

        std::string str_name;
        str_name += "_stor_" + std::string(&storageC, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16m_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16m_col_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x16m_col_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16n_col_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(false),                               // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16n_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x16n_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(false),                               // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64m_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64m_avx512), // sgemm_sup kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa
            ::testing::Values('n'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64m_col_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(1), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('c'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64m_avx512), // sgemm_sup_kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('n'),                                  // transa
            ::testing::Values('t'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

/*
    The bli_sgemmsup_rd_zen_asm_6x64m_avx512(standalone), accepts inputs with the
    following contingency for n.
        n <= NR, where NR is 64
    The code structure for the sgemm_sup rd kernels(m-var) are as follows: 
    In m direction :
        Main kernel    : Blocks of 6(L6_M)
        Fringe kernels : 5 ... 1(L5_M ... L1_M)
    In k direction :
        Main loop   : Blocks of 64(L64_K)
        Fringe loop : Blocks of 32, 8, 1(L32_K ... L1_K)
    In n direction :
        Main kernel    : NR = 64(L64_N)
        Fringe kernels : With n being 48, 32(AVX512 kernels)(L48_N, L32_N)
                         With n being 16, 8, 4, 2, 1(Reusing AVX2 kernels)(L16_N ... L1_N)

    The inherent storage scheme format for the kernel is RRC, for C, A and B.
    The testing interface allows for testing row-storage(inherent) and col-storage(operation transpose)
    of C. We still need to pass the right transpose value pair for A and B, as per the kernel requirement. 
*/

// Checking with row storage of C
INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x64m_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), gtint_t(1)),    // values of m(L6_M to L1_M)
            ::testing::Values(gtint_t(64),                           // values of n, L64_N
                              gtint_t(48),                           // L48_N
                              gtint_t(32),                           // L32_N
                              gtint_t(8),                            // L8_N
                              gtint_t(7),                            // 7 * L1_N
                              gtint_t(63)),                          // Combination of fringe cases for N
            ::testing::Values(gtint_t(64),                           // values of k, L64_K
                              gtint_t(32),                           // L32_K
                              gtint_t(8),                            // L8_K
                              gtint_t(7),                            // 7 * L1_K
                              gtint_t(256),                          // 4 * L64_K
                              gtint_t(303)),                         // Combination of main and fringe cases for K
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x64m_avx512), // sgemm_sup_kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('n'),                                  // transa, has to be N for row storage
            ::testing::Values('t'),                                  // transb, has to be T for row storage
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

// Checking with col storage of C
// NOTE : Since we are inducing transpose at opertaion level, for code coverage, we
//        have to interchange m and n instantiations
INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x64m_col_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(64),                           // values of m, L64_N
                              gtint_t(48),                           // L48_N
                              gtint_t(32),                           // L32_N
                              gtint_t(8),                            // L8_N
                              gtint_t(7),                            // 7 * L1_N
                              gtint_t(63)),                          // Combination of fringe cases
            ::testing::Range(gtint_t(1), gtint_t(7), gtint_t(1)),    // values of n(L6_M to L1_M)
            ::testing::Values(gtint_t(64),                           // values of k, L64_K
                              gtint_t(32),                           // L32_K
                              gtint_t(8),                            // L8_K
                              gtint_t(7),                            // 7 * L1_K
                              gtint_t(256),                          // 4 * L64_K
                              gtint_t(303)),                         // Combination of main and fringe cases for K
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('c'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x64m_avx512), // sgemm_sup_kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa, has to be T for row storage
            ::testing::Values('n'),                                  // transb, has to be N for row storage
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64n_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64n_avx512), // sgemm_sup_kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa
            ::testing::Values('n'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x64n_row_stored_c,
        sgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x64n_avx512), // sgemm_sup_kernel
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('n'),                                  // transa
            ::testing::Values('t'),                                  // transb
            ::testing::Values(false),                                // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmGenericSUPPrint()
    );
#endif

/*******************************************************/
/*              Native Kernel testing                  */
/*******************************************************/
class sgemmGenericNat :
//        public ::testing::TestWithParam<std::tuple<sgemm_ukr_ft, gtint_t, float, float, char, gtint_t, gtint_t, bool>> {};
//sgemm native kernel, k, alpha, beta, storage of c, m, n, memory test

        public ::testing::TestWithParam<std::tuple<gtint_t,                 // k
                                                   float,                   // alpha
                                                   float,                   // beta
                                                   char,                    // storage of C matrix
                                                   gtint_t,                 // m
                                                   gtint_t,                 // n
                                                   sgemm_ukr_ft,            // pointer to the gemm kernel
                                                   bool                     // is_memory_test
                                                   >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmGenericNat);

TEST_P( sgemmGenericNat, functionality_testing)
{
    using T = float;
    gtint_t k             = std::get<0>(GetParam());                        // dimension k
    T alpha               = std::get<1>(GetParam());                        // alpha
    T beta                = std::get<2>(GetParam());                        // beta
    char storageC         = std::get<3>(GetParam());                        // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m             = std::get<4>(GetParam());                        // m
    gtint_t n             = std::get<5>(GetParam());                        // n
    sgemm_ukr_ft kern_ptr = std::get<6>(GetParam());                        // pointer to the gemm kernel
    bool is_memory_test   = std::get<7>(GetParam());                        // is_memory_test

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
              beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    test_gemmnat_ukr(storageC, m, n, k, alpha, beta, kern_ptr, thresh, is_memory_test);

}// end of function



class sgemmGenericNatPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, float, float, char, gtint_t, gtint_t, sgemm_ukr_ft, bool>>  str) const {

        gtint_t k           = std::get<0>(str.param);
        float alpha         = std::get<1>(str.param);
        float beta          = std::get<2>(str.param);
        char storageC       = std::get<3>(str.param);
        bool is_memory_test = std::get<7>(str.param);

        std::string str_name;
        str_name += "_stor_" + std::string(&storageC, 1);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_sgemm_skx_asm_32x12_l2,
    sgemmGenericNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(32),                          // values of m
        ::testing::Values(12),                          // values of n
        ::testing::Values(bli_sgemm_skx_asm_32x12_l2),
        ::testing::Values(true, false)                  // memory test
    ),
    ::sgemmGenericNatPrint()
);


#endif

#if defined(BLIS_KERNELS_HASWELL) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_sgemm_haswell_asm_6x16,
    sgemmGenericNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(6),                           // values of m
        ::testing::Values(16),                          // values of n
        ::testing::Values(bli_sgemm_haswell_asm_6x16),
        ::testing::Values(true, false)                  // memory test
    ),
    ::sgemmGenericNatPrint()
);
#endif

#if 0
/**
 * sgemm_small microkernel testing disable because sgemm_small is static local
 * function. Once it is made global, this testcase can be enabled.
 * As of now for the compilation sake, this testcase is kept disabled.
*/
#ifdef BLIS_ENABLE_SMALL_MATRIX

class sgemmGenericSmallTest :
        public ::testing::TestWithParam<std::tuple< gtint_t,                // m
                                                    gtint_t,                // n
                                                    gtint_t,                // k
                                                    float,                  // alpha
                                                    float,                  // beta
                                                    char                    // storage of C matrix
                                                    >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmGenericSmallTest);

TEST_P( sgemmGenericSmallTest, gemm_small)
{
    using T = float;
    gtint_t m      = std::get<0>(GetParam());  // dimension m
    gtint_t n      = std::get<1>(GetParam());  // dimension n
    gtint_t k      = std::get<2>(GetParam());  // dimension k
    T alpha        = std::get<3>(GetParam());  // alpha
    T beta         = std::get<4>(GetParam());  // beta
    char storageC  = std::get<5>(GetParam());  // indicates storage of all matrix operands


    gtint_t lda = testinghelpers::get_leading_dimension( storageC, 'n', m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storageC, 'n', k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storageC, 'n', m, n, 0 );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storageC, 'n', m, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storageC, 'n', k, n, ldb );
    std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storageC, 'n', m, n, ldc );

    std::vector<T> c_ref(c);

    const num_t dt = BLIS_FLOAT;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao = BLIS_OBJECT_INITIALIZER;
    obj_t       bo = BLIS_OBJECT_INITIALIZER;
    obj_t       betao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co = BLIS_OBJECT_INITIALIZER;

    dim_t       m0_a, n0_a;
    dim_t       m0_b, n0_b;

    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, m, k, &m0_a, &n0_a);
    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, k, n, &m0_b, &n0_b);

    bli_obj_init_finish_1x1(dt, (float*)&alpha, &alphao);
    bli_obj_init_finish_1x1(dt, (float*)&beta, &betao);

    bli_obj_init_finish(dt, m0_a, n0_a, (float*)a.data(), 1, lda, &ao);
    bli_obj_init_finish(dt, m0_b, n0_b, (float*)b.data(), 1, ldb, &bo);
    bli_obj_init_finish(dt, m, n, (float*)c.data(), 1, ldc, &co);

    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &ao);
    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &bo);


    bli_sgemm_small ( &alphao,
                      &ao,
                      &bo,
                      &betao,
                      &co,
                      NULL,
                      NULL
                    );


    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
              beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storageC, 'n', 'n', m, n, k, alpha,
                                 a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc);

    // Check component-wise error
    computediff<T>( "C", storageC, m, n, c.data(), c_ref.data(), ldc, thresh );

}// end of function



class sgemmGenericSmallTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, float, float, char>>  str) const {

        gtint_t m     = std::get<0>(str.param);
        gtint_t n     = std::get<1>(str.param);
        gtint_t k     = std::get<2>(str.param);
        float alpha   = std::get<3>(str.param);
        float beta    = std::get<4>(str.param);
        char storageC = std::get<5>(str.param);

        std::string str_name;
        str_name += "_stor_" + std::string(&storageC, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);

        return str_name;
    }
};


INSTANTIATE_TEST_SUITE_P (
        bli_sgemm_small,
        sgemmGenericSmallTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(71), 1), // values of m
            ::testing::Range(gtint_t(1), gtint_t(21), 1), // values of n
            ::testing::Range(gtint_t(1), gtint_t(20), 1), // values of k
            ::testing::Values(2.0, 1.0, -1.0),            // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),       // beta value
            ::testing::Values('c')                        // storage
        ),
        ::sgemmGenericSmallTestPrint()
    );

#endif
#endif
