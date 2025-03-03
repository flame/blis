/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "common/blis_version_defs.h"

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
                                                    gtint_t,                // MR
                                                    char,                   // transa
                                                    char,                   // transb
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
    gtint_t MR               = std::get<7>(GetParam());                     // ukr dimension MR
    char transa              = std::get<8>(GetParam());                     // transa
    char transb              = std::get<9>(GetParam());                     // transb
    bool is_memory_test      = std::get<10>(GetParam());                    // is_memory_test

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
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.7;
#else
        double adj = 9.4;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<T>();
    }
    test_complex_gemmsup_ukr<scomplex, cgemmsup_ker_ft> (storageC, transa, transb, m, n, k, MR, alpha, beta, thresh, kern_ptr, is_memory_test);
}// end of function

class cgemmGenericSUPPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, scomplex, scomplex, char,
                                          cgemmsup_ker_ft, gtint_t, char, char, bool>>  str) const {

        gtint_t m           = std::get<0>(str.param);
        gtint_t n           = std::get<1>(str.param);
        gtint_t k           = std::get<2>(str.param);
        scomplex alpha      = std::get<3>(str.param);
        scomplex beta       = std::get<4>(str.param);
        char storageC       = std::get<5>(str.param);
        char transa         = std::get<8>(str.param);
        char transb         = std::get<9>(str.param);
        bool is_memory_test = std::get<10>(str.param);

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

// Testing the kernels without in-register transpose
#ifdef K_bli_cgemmsup_rv_zen_asm_3x8m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x8m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                    // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x4m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x4m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x2m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x2m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x8n_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#if 0
//Memtest fails
//Memtest diabled free(): invalid next size (fast)
#ifdef K_bli_cgemmsup_rv_zen_asm_2x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x8n_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(3), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false)                                        // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#endif // disable memtest

#ifdef K_bli_cgemmsup_rv_zen_asm_1x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x8n_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x8
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x8_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x8
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x8_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('n'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

// Testing the kernels with in-register transpose
#ifdef K_bli_cgemmsup_rv_zen_asm_3x8m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x8m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),                    // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x4m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x4m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x2m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x2m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(8), 1),                    // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2m),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x8n_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#if 0
//Memtest fails
//Memtest diabled free(): invalid next size (fast)
#ifdef K_bli_cgemmsup_rv_zen_asm_2x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x8n_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(3), 1),                    // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false)                                        // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#endif // disable memtest

#ifdef K_bli_cgemmsup_rv_zen_asm_1x8n
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x8n_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Range(gtint_t(1), gtint_t(16), 1),                   // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8n),                // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_3x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_3x2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_3x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x8
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x8_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x8),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x8
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x8_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(8)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x8),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x4),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_2x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_2x2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_2x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );
#endif

#ifdef K_bli_cgemmsup_rv_zen_asm_1x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_rv_zen_asm_1x2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(10)),                                 // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_rv_zen_asm_1x2),                 // cgemm_sup kernel
            ::testing::Values(gtint_t(3)),                                  // Micro kernel block MR
            ::testing::Values('n', 't'),                                    // transa
            ::testing::Values('t'),                                         // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/* The AVX512 CGEMM SUP kernels are column-preferential in nature
   Thus, the computational model is as follows :
   Load      : A matrix(column storage)
   Broadcast : B matrix(row/column storage)
   Store     : C matrix(row/column storage)

   Thus, the supported storage schemes(in the order C,A,B) are :
   CCC, CCR, RCC, RCR.
   Every other storage scheme is converted to one of these at the
   framework layer(through packing and/or operation transpose).

   Every kernel is tested with B/C being in row/colum storage, and A
   strictly being in column storage(this is controlled through the transpose
   values).
*/
// NOTE : The values for k in 24x4m kernel tests are such that they test for
//        loops taken before/during/after prefetch of C. For the other kernels,
//        we test for the main(unrolled) adn the fringe loops of k.
#ifdef K_bli_cgemmsup_cv_zen4_asm_24x4m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x4m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Range(gtint_t(1), gtint_t(4), gtint_t(1)),           // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(32),
                              gtint_t(40), gtint_t(67)),                    // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x4m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x4m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Range(gtint_t(1), gtint_t(4), gtint_t(1)),           // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(32),
                              gtint_t(40), gtint_t(67)),                    // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x4m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_24x3m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x3m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x3m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x3m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x3m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_24x2m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x2m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x2m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x2m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x2m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_24x1m
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x1m_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x1m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_24x1m_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(24), gtint_t(49), gtint_t(1)),         // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_24x1m),              // cgemm_sup kernel
            ::testing::Values(gtint_t(24)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_16x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x4),              // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x4),              // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_16x3
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x3_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x3),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x3_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x3),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_16x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x2),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x2),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_16x1
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x1_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x1),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_16x1_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(16)),                                 // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_16x1),               // cgemm_sup kernel
            ::testing::Values(gtint_t(16)),                                 // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_8x4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                 // values of m
            ::testing::Values(gtint_t(4)),                                 // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),       // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x4),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                 // values of m
            ::testing::Values(gtint_t(4)),                                 // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),       // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x4),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_8x3
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x3_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x3),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x3_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x3),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_8x2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x2),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x2),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_8x1
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x1_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x1),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_8x1_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(8)),                                  // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_8x1),                // cgemm_sup kernel
            ::testing::Values(gtint_t(8)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_fx4
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx4_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx4),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx4_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(4)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx4),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_fx3
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx3_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx3),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx3_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(3)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx3),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_fx2
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx2_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx2),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx2_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(2)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx2),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif

#ifdef K_bli_cgemmsup_cv_zen4_asm_fx1
INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx1_col_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx1),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('n'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_cgemmsup_cv_zen4_asm_fx1_row_stored,
        cgemmGenericSUP,
        ::testing::Combine(
            ::testing::Values(gtint_t(5)),                                  // values of m
            ::testing::Values(gtint_t(1)),                                  // values of n
            ::testing::Values(gtint_t(3), gtint_t(16), gtint_t(67)),        // values of k
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -5.0}, scomplex{3, 4}), // alpha value
            ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -5.0}, scomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                         // storage
            ::testing::Values(bli_cgemmsup_cv_zen4_asm_fx1),                // cgemm_sup kernel
            ::testing::Values(gtint_t(5)),                                  // Micro kernel block MR
            ::testing::Values('t'),                                         // transa
            ::testing::Values('n', 't'),                                    // transb
            ::testing::Values(false, true)                                  // is_memory_test
        ),
        ::cgemmGenericSUPPrint()
    );

#endif
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
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
              beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 3.0;
#else
        double adj = 7.1;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<T>();
    }
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
#ifdef K_bli_cgemm_haswell_asm_3x8
INSTANTIATE_TEST_SUITE_P(
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
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
#ifdef K_bli_cgemm_zen4_asm_24x4
INSTANTIATE_TEST_SUITE_P(
    bli_cgemm_zen4_asm_24x4,
    cgemmGenericNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(1), gtint_t(20), 1),                       // values of k
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -0.2}, scomplex{3.2, 4.5}),   // alpha value
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -2.1}, scomplex{-7.3, 6.7}), // beta value
        ::testing::Values('r', 'c'),                                        // storage
        ::testing::Values(24),                                              // values of m
        ::testing::Values(4),                                               // values of n
        ::testing::Values(bli_cgemm_zen4_asm_24x4),                         // cgemm_nat kernel
        ::testing::Values(false, true)                                      // is_memory_test
    ),
    ::cgemmGenericNatPrint()
);
#endif

#ifdef K_bli_cgemm_zen4_asm_4x24
INSTANTIATE_TEST_SUITE_P(
    bli_cgemm_zen4_asm_4x24,
    cgemmGenericNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(1), gtint_t(20), 1),                       // values of k
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{4.0, 0.0}, scomplex{0.0, -0.2}, scomplex{3.2, 4.5}),   // alpha value
        ::testing::Values(scomplex{0.0, 0.0}, scomplex{1.0, 0.0}, scomplex{-1.0, 0.0}, scomplex{-5.0, 0.0}, scomplex{0.0, -2.1}, scomplex{-7.3, 6.7}), // beta value
        ::testing::Values('r', 'c'),                                        // storage
        ::testing::Values(4),                                               // values of m
        ::testing::Values(24),                                              // values of n
        ::testing::Values(bli_cgemm_zen4_asm_4x24),                         // cgemm_nat kernel
        ::testing::Values(false, true)                                      // is_memory_test
    ),
    ::cgemmGenericNatPrint()
);
#endif
#endif
