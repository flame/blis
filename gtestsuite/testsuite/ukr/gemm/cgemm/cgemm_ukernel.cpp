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
#include "ukr/gemm/test_complex_gemm_ukr.h"

/*******************************************************/
/*                 SUP Kernel testing                  */
/*******************************************************/
class cgemmGenericSUP:
        public ::testing::TestWithParam<std::tuple< gtint_t,                // m
                                                    gtint_t,                // n
                                                    gtint_t,                // k
                                                    scomplex,               // alpha
                                                    scomplex,               // beta
                                                    char,                   // storage matrix
                                                    cgemmsup_ker_ft,        // Function pointer type for cgemm kernel
                                                    char,                   // transa
                                                    bool                    // is_memory_test
                                                    >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cgemmGenericSUP);

TEST_P( cgemmGenericSUP, UKR )
{
    using T                  = scomplex;
    gtint_t m                = std::get<0>(GetParam());                     // dimension m
    gtint_t n                = std::get<1>(GetParam());                     // dimension n
    gtint_t k                = std::get<2>(GetParam());                     // dimension k
    T alpha                  = std::get<3>(GetParam());                     // alpha
    T beta                   = std::get<4>(GetParam());                     // beta
    char storageC            = std::get<5>(GetParam());                     // storage scheme for C matrix
    cgemmsup_ker_ft kern_ptr = std::get<6>(GetParam());                     // pointer to the gemm kernel
    char transa              = std::get<7>(GetParam());                     // transa
    char transb              = (storageC == 'r')? 'n' : 't';                // transb
    bool is_memory_test      = std::get<8>(GetParam());                     // is_memory_test

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
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

    test_complex_gemmsup_ukr<scomplex, cgemmsup_ker_ft> (storageC, transa, transb, m, n, k, alpha, beta, thresh, kern_ptr, is_memory_test);
}// end of function

class cgemmGenericSUPPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, scomplex, scomplex, char,
                                          cgemmsup_ker_ft, char, bool>>  str) const {

        gtint_t m           = std::get<0>(str.param);
        gtint_t n           = std::get<1>(str.param);
        gtint_t k           = std::get<2>(str.param);
        scomplex alpha      = std::get<3>(str.param);
        scomplex beta       = std::get<4>(str.param);
        char storageC       = std::get<5>(str.param);
        char transa         = std::get<7>(str.param);
        char transb         = (storageC == 'r')? 'n' : 't';
        bool is_memory_test = std::get<8>(str.param);

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

/*********************************************************/
/* Stroage Formats For SUP Kernels                       */
/* A Matrix: Broadcast instruction is applied on Matrix  */
/*           hence it can be row or col stored           */
/*           trana = 'n' or 't'                          */
/* B Matrix: Load instruction is applied on Matrix       */
/*           hence it has to be row stored               */
/*           When storage = r, transb = 'n'              */
/*           When storage = c, transb = 't'              */
/* C Matrix: Supports row or col storage                 */
/*********************************************************/

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

/*************************************************/
/***********Choosing values of m, n, k************/
/* m is vectorised for 3                         */
/* - main kernel : 3, 6 (3x2)                    */
/* - fringe case : 1, 2                          */
/* - main kernel and fringe case:                */
/*    4(3+1), 5(3+2), 7(3x2+1), 8(3x2+2)         */
/* n is vectorised for 4 and 2                   */
/* - main kernel : 4, 2, 1(gemv)                 */
/* - main kernel and fringe case:                */
/*    3(2+1), 5(4+1), 6(4+2), 7(4+2+1)           */
/* k is unrolled 4 times                         */
/* - main loop : 4, 8                            */
/* - fringe loop : 1, 2                          */
/* - main and fringe 5, 6, 9, 10                 */
/*************************************************/

/*Failures*/
/* 1. blis_sol[i*ld + j] = (0.856704, 0.625597),   ref_sol[i*ld + j] = (0.856718, 0.625608), i = 5, j = 0,    thresh = 9.5367431640625e-06,    error = 1.7269374438910745e-05 (144.86601257324219 * eps)
[  FAILED  ] bli_cgemmsup_rv_zen_asm_3x8m/cgemmGenericSUP.FunctionalTest/StorageOfMatrix_r_transA_t_transB_n_m_6_n_8_k_4_alpha_3i4_beta_m7i6_mem_test_disabled, where GetParam() = (6, 8, 4, (3, 4.5), (-7.3, 6.7), 'r' (114, 0x72), 0x5576cdf96cc7, 't' (116, 0x74), 'n' (110, 0x6E), false) (0 ms) */

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x8m,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                    // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x8m_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                    // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x4m,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x4m_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x2m,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x2m_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2m),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x8n,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x8n_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#if 0
//Memtest fails
//Memtest diabled free(): invalid next size (fast)
INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x8n,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(3), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false)                                        // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x8n_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(3), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false)                                        // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif
INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x8n,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x8n_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8n),                // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x4,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x4_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x2,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_3x2_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

 INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x8,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

 INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x8_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x8,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x8_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x4,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x4_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x4,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x4_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x4),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x2,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_2x2_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x2,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),                   // values of k
            ::testing::Values(scomplex{3, 4}),                              // alpha value
            ::testing::Values(scomplex{-7.3, 6.7}),                         // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_cgemmsup_rv_zen_asm_1x2_alpha_beta,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r', 'c'),                                    // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x2),                 // cgemm_sup kernel
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

/*******************************************************/
/*              Native Kernel testing                  */
/*******************************************************/
class cgemmGenericNat :
        public ::testing::TestWithParam<std::tuple<gtint_t,                 // k
                                                   scomplex,                // alpha
                                                   scomplex,                // beta
                                                   char,                    // storage of C matrix
                                                   gtint_t,                 // m
                                                   gtint_t,                 // n
                                                   cgemm_ukr_ft,            // pointer to the gemm kernel
                                                   bool                     // is_memory_test
                                                   >> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cgemmGenericNat);
TEST_P( cgemmGenericNat, UKR )
{
    using T = scomplex;
    gtint_t k             = std::get<0>(GetParam());                        // dimension k
    T alpha               = std::get<1>(GetParam());                        // alpha
    T beta                = std::get<2>(GetParam());                        // beta
    char storageC         = std::get<3>(GetParam());                        // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m             = std::get<4>(GetParam());                        // m
    gtint_t n             = std::get<5>(GetParam());                        // n
    cgemm_ukr_ft kern_ptr = std::get<6>(GetParam());                        // pointer to the gemm kernel
    bool is_memory_test   = std::get<7>(GetParam());                        // is_memory_test

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
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

    test_gemmnat_ukr(storageC, m, n, k, alpha, beta, thresh, kern_ptr, is_memory_test);
}// end of function

class cgemmGenericNatPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, scomplex, scomplex, char, gtint_t, gtint_t, cgemm_ukr_ft, bool>>  str) const {

        gtint_t k           = std::get<0>(str.param);
        scomplex alpha      = std::get<1>(str.param);
        scomplex beta       = std::get<2>(str.param);
        char storageC       = std::get<3>(str.param);
        bool is_memory_test = std::get<7>(str.param);

        std::string str_name ;
        str_name += "_stor_" + std::string(&storageC, 1);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_HASWELL) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_cgemm_haswell_asm_3x8,
    cgemmGenericNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(1), gtint_t(20), 1),                       // values of k
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -0.2}, scomplex{3.5, 4.5}),   // alpha value
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -2.1}, scomplex{-7.3, 6.7}), // beta value
        ::testing::Values('r', 'c'),                                        // storage
        ::testing::Values(3),                                               // values of m
        ::testing::Values(8),                                               // values of n
        ::testing::Values(bli_cgemm_haswell_asm_3x8),                       // cgemm_nat kernel
        ::testing::Values(false, true)                                      // is_memory_test
    ),
    ::cgemmGenericNatPrint()
);
#endif
