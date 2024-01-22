
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
#include "test_gemm_ukr.h"

/*******************************************************/
/*                 SUP Kernel testing                  */
/*******************************************************/
class ZGEMMUkrSUPTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, dcomplex, dcomplex, char, zgemmsup_ker_ft, char, char >> {};
                                                  //  m, n, k, alpha, beta, storage of c, zgemm sup kernel, transa, transb

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ZGEMMUkrSUPTest);
TEST_P(ZGEMMUkrSUPTest, sup_kernel)
{
    using T = dcomplex;
    gtint_t m      = std::get<0>(GetParam());  // dimension m
    gtint_t n      = std::get<1>(GetParam());  // dimension n
    gtint_t k      = std::get<2>(GetParam());  // dimension k
    T alpha        = std::get<3>(GetParam());  // alpha
    T beta         = std::get<4>(GetParam());  // beta
    char storageC   = std::get<5>(GetParam()); // storage scheme for C matrix
    zgemmsup_ker_ft kern_ptr = std::get<6>(GetParam()); //pointer to the gemm kernel
    char transa = std::get<7>(GetParam());
    char transb = std::get<8>(GetParam());
    test_zgemmsup_ukr(storageC, transa, transb, m, n, k, alpha, beta, kern_ptr);
}// end of function

class ZGEMMukrsupTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, dcomplex, dcomplex, char, zgemmsup_ker_ft, char, char>>  str) const {
        gtint_t m       = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t k       = std::get<2>(str.param);
        dcomplex alpha  = std::get<3>(str.param);
        dcomplex beta   = std::get<4>(str.param);
        char storageC  = std::get<5>(str.param);
        char trnsa     = std::get<7>(str.param);
        char trnsb     = std::get<8>(str.param);
        std::string str_name = "zgemmsup_ukr";
        str_name = str_name + "_" + trnsa;
        str_name = str_name + "_" + trnsb;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(n);
        str_name = str_name + "_" + std::to_string(k);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + storageC;
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x4m_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(10), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(5), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(15), 1),           // values of k
            //alpha values dcomplex{0.0, 0.0} failure observed
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -5.0}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -5.0}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x4m),        // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x4_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(19), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 5.0}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 5.0}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x4),         // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x4_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(18), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 5.5}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 5.4}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x4),         // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x2m_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(20), 1),           // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(13), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 2}, dcomplex{3.5, 4.5}),    // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x2m),        // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x2_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(5), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0,15.0}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 2.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x2),         // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x2_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(12), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 12}, dcomplex{3.5, 4.5}),    // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 13}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x2),         // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x2_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(8), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 6}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x2),         // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x4m_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(14), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(5), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(22), 1),           // values of k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -15.0}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x4m),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x2m_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(14), 1),           // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 3.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x2m),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x2_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(3)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(19), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 2.3}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.4}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x2),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x4_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(7), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 19.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.99}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x4),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x4_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(8), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0},dcomplex{0.0, 1.9},  dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x4),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x2_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.5}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -1.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x2),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x2_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(8), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 2.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x2),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_3x4m_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(12), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(5), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.5}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 2.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_3x4m),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                 // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_3x2m_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(11), 1),           // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.19}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_3x2m),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_3x4n_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(10), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(16),1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.0}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 2.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_3x4n),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                 // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_2x4n_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Range(gtint_t(1), gtint_t(12), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(14), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.23}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_2x4n),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_2x4_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(14), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.34}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 2.9}, dcomplex{-7.3, 6.7}),  // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_2x4),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_1x4_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(4)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(9), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.56}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 21.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_1x4),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_1x2_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(8), 1),            // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.99}, dcomplex{3.5, 4.5}),    // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -21.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_1x2),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rd_zen_asm_2x2_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Values(gtint_t(2)),                          // values of n
            ::testing::Range(gtint_t(0), gtint_t(10), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 91.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -2.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rd_zen_asm_2x2),         // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x4n_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(15), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(12), 1),           // values of k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -2}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x4n),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x4n_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Range(gtint_t(1), gtint_t(13), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 8.9}, dcomplex{3.5, 4.5}),    // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -1.9}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x4n),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x4n_col_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Range(gtint_t(1), gtint_t(8), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.3}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 5.6}, dcomplex{-7.3, 6.7}),  // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x4n),        // zgemm_sup kernel
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_3x4n_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(4), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(18), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0,.0}, dcomplex{0.0, 2.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_3x4n),        // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_2x4n_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(2)),                          // values of m
            ::testing::Range(gtint_t(1), gtint_t(6), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -5.6}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{-7.3, 6.7}),  // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_2x4n),        // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_zgemmsup_rv_zen_asm_1x4n_row_stored_c,
        ZGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(1)),                          // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(20), 1),           // values of k
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.9}, dcomplex{3.5, 4.5}),   // alpha value
            ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -1.3}, dcomplex{-7.3, 6.7}), // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_zgemmsup_rv_zen_asm_1x4n),        // zgemm_sup kernel
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n')                                  // transb
        ),
        ::ZGEMMukrsupTestPrint()
    );
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x4m_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(28), 1),           // values of m
             ::testing::Range(gtint_t(1), gtint_t(5), 1),            // values of n
             ::testing::Range(gtint_t(0), gtint_t(19), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -8}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x4m),      // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x3m_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
             ::testing::Values(gtint_t(3)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(10), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 9}, dcomplex{-7.3, 6.7}),   // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x3m),      // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x2m_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(20), 1),           // values of m
             ::testing::Values(gtint_t(2)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(13), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -0.9}, dcomplex{3.5, 4.5}),    // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -21.9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x2m),      // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x1m_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
             ::testing::Values(gtint_t(1)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(22), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -31.9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.4}, dcomplex{-7.3, 6.7}),   // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x1m),      // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_8x4_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(8)),                          // values of m
             ::testing::Values(gtint_t(4)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 8}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_8x4),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_8x3_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(8)),                          // values of m
             ::testing::Values(gtint_t(3)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(16), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.2}, dcomplex{3.5, 4.5}),    // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -1.8}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_8x3),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_8x2_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(8)),                          // values of m
             ::testing::Values(gtint_t(2)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(14), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 9}, dcomplex{-7.3, 6.7}),  // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_8x2),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_8x1_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(8)),                          // values of m
             ::testing::Values(gtint_t(1)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(10), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -2}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_8x1),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_4x4_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(4)),                          // values of m
             ::testing::Values(gtint_t(4)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(9), 1),            // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 3}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_4x4),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_4x3_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(4)),                          // values of m
             ::testing::Values(gtint_t(3)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(19), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -1.9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.5}, dcomplex{-7.3, 6.7}),  // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_4x3),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_4x2_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(4)),                          // values of m
             ::testing::Values(gtint_t(2)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(14), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -19}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -9}, dcomplex{-7.3, 6.7}),  // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_4x2),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_4x1_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(4)),                          // values of m
             ::testing::Values(gtint_t(1)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(12), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -19}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1}, dcomplex{-7.3, 6.7}),   // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_4x1),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_2x4_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(2)),                          // values of m
             ::testing::Values(gtint_t(4)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(16), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.9}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.8}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_2x4),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_2x3_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(2)),                          // values of m
             ::testing::Values(gtint_t(3)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(5), 1),            // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 18}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1}, dcomplex{-7.3, 6.7}),  // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_2x3),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_2x2_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(2)),                          // values of m
             ::testing::Values(gtint_t(2)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(9), 1),            // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -19}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 9}, dcomplex{-7.3, 6.7}),   // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_2x2),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_2x1_col_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Values(gtint_t(2)),                          // values of m
             ::testing::Values(gtint_t(1)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(15), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 9}, dcomplex{3.5, 4.5}),    // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('c'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_2x1),        // zgemm_sup kernel
             ::testing::Values('n'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x4m_row_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(13), 1),           // values of m
             ::testing::Range(gtint_t(1), gtint_t(5), 1),            // values of n
             ::testing::Range(gtint_t(0), gtint_t(14), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 7}, dcomplex{3.5, 4.5}),    // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('r'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x4m),      // zgemm_sup kernel
             ::testing::Values('t'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x3m_row_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(33), 1),           // values of m
             ::testing::Values(gtint_t(3)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(12), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -9.7}, dcomplex{3.5, 4.5}),  // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.2}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('r'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x3m),      // zgemm_sup kernel
             ::testing::Values('t'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x2m_row_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(21), 1),           // values of m
             ::testing::Values(gtint_t(2)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(12), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 1.4}, dcomplex{3.5, 4.5}),   // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 8.9}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('r'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x2m),      // zgemm_sup kernel
             ::testing::Values('t'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_zgemmsup_cv_zen4_asm_12x1m_row_stored_c,
         ZGEMMUkrSUPTest,
         ::testing::Combine(
             ::testing::Range(gtint_t(1), gtint_t(20), 1),           // values of m
             ::testing::Values(gtint_t(1)),                          // values of n
             ::testing::Range(gtint_t(0), gtint_t(10), 1),           // values of k
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 9}, dcomplex{3.5, 4.5}),    // alpha value
             ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 19}, dcomplex{-7.3, 6.7}), // beta value
             ::testing::Values('r'),                                 // storage of c
             ::testing::Values(bli_zgemmsup_cv_zen4_asm_12x1m),      // zgemm_sup kernel
             ::testing::Values('t'),                                 // transa
             ::testing::Values('n')                                  // transb
         ),
         ::ZGEMMukrsupTestPrint()
     );
#endif

/*******************************************************/
/*              Native Kernel testing                  */
/*******************************************************/
class ZGEMMUkrNatTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, dcomplex, dcomplex, char, gtint_t, gtint_t, zgemm_ukr_ft>> {};
// k, alpha, beta, storage of c, m, n, zgemm native kernel

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ZGEMMUkrNatTest);
TEST_P(ZGEMMUkrNatTest, native_kernel_testing)
{
    using T = dcomplex;
    gtint_t k      = std::get<0>(GetParam());  // dimension k
    T alpha        = std::get<1>(GetParam());  // alpha
    T beta         = std::get<2>(GetParam());  // beta
    char storage   = std::get<3>(GetParam());  // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<4>(GetParam());
    gtint_t n = std::get<5>(GetParam());
    zgemm_ukr_ft kern_ptr = std::get<6>(GetParam());
    test_gemmnat_ukr(storage, m, n, k, alpha, beta, kern_ptr);
}// end of function

class ZGEMMukrnatTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, dcomplex, dcomplex, char, gtint_t, gtint_t, zgemm_ukr_ft>>  str) const {
        gtint_t k       = std::get<0>(str.param);
        dcomplex alpha  = std::get<1>(str.param);
        dcomplex beta   = std::get<2>(str.param);
        char storage    = std::get<3>(str.param);
        std::string str_name = "zgemmnat_ukr";
        str_name = str_name + "_" + std::to_string(k);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + storage;
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_zgemm_zen4_asm_12x4,
    ZGEMMUkrNatTest,
    ::testing::Combine( //Failure observed for this case zgemmnat_ukr_1_a0pi2_bm7pi6_r
        ::testing::Range(gtint_t(0), gtint_t(15), 1),    // values of k
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 2.3}, dcomplex{3.5, 4.5}),   // alpha value
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 1.0}, dcomplex{-3, 6.7}), // beta value
        ::testing::Values('r', 'c'),      // storage
        ::testing::Values(12),            // values of m
        ::testing::Values(4),             // values of n
        ::testing::Values(bli_zgemm_zen4_asm_12x4)
    ),
    ::ZGEMMukrnatTestPrint()
);

/*Kernel reqired for trsm computation*/
INSTANTIATE_TEST_SUITE_P (
    bli_zgemm_zen4_asm_4x12,
    ZGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(10), 1),   // values of k
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, 2.3}, dcomplex{3.5, 4.5}),     // alpha value
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, 3.3}, dcomplex{-7.3, 6.7}),   // beta value
        ::testing::Values('r', 'c'),      // storage
        ::testing::Values(4),             // values of m
        ::testing::Values(12),            // values of n
        ::testing::Values(bli_zgemm_zen4_asm_4x12)
    ),
    ::ZGEMMukrnatTestPrint()
);
#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_zgemm_zen_asm_3x4,
    ZGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(20), 1),  // values of k
        ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -0.2}, dcomplex{3.5, 4.5}),   // alpha value
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -2.1}, dcomplex{-7.3, 6.7}), // beta value
        ::testing::Values('r', 'c'),      // storage
        ::testing::Values(3),             // values of m
        ::testing::Values(4),             // values of n
        ::testing::Values(bli_zgemm_haswell_asm_3x4)
    ),
    ::ZGEMMukrnatTestPrint()
);

/*Kernel reqired for trsm computation*/
INSTANTIATE_TEST_SUITE_P (
    bli_zgemm_zen_asm_2x6,
    ZGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(10), 1),  // values of k
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{4.0, 0.0}, dcomplex{0.0, -0.3}, dcomplex{3.5, 4.5}),   // alpha value
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}, dcomplex{-5.0, 0.0}, dcomplex{0.0, -2.0}, dcomplex{-7.3, 6.7}), // beta value
        ::testing::Values('r', 'c'),      // storage
        ::testing::Values(2),             // values of m
        ::testing::Values(6),             // values of n
        ::testing::Values(bli_zgemm_zen_asm_2x6)
    ),
    ::ZGEMMukrnatTestPrint()
);
#endif