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

class sgemmUkrSUP :
        public ::testing::TestWithParam<std::tuple<sgemmsup_ker_ft, gtint_t, gtint_t, gtint_t, float, float, char, gtint_t, char, char, bool, bool>> {};
// m, n, k, alpha, beta,  storage of c, sgemm sup kernel, micro-kernel MR block, transa, transb, kernel transpose, memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmUkrSUP);

TEST_P(sgemmUkrSUP, functionality_testing)
{
    using T = float;
    sgemmsup_ker_ft kern_ptr = std::get<0>(GetParam()); //pointer to the gemm kernel
    gtint_t m      = std::get<1>(GetParam());           // dimension m
    gtint_t n      = std::get<2>(GetParam());           // dimension n
    gtint_t k      = std::get<3>(GetParam());           // dimension k
    T alpha        = std::get<4>(GetParam());           // alpha
    T beta         = std::get<5>(GetParam());           // beta
    char storageC   = std::get<6>(GetParam());          // storage scheme for C matrix
    gtint_t MR  = std::get<7>(GetParam());              // Micro-kernel tile size
    char transa = std::get<8>(GetParam());              // A transopse
    char transb = std::get<9>(GetParam());              // B transpose
    bool kern_trans = std::get<10>(GetParam());         // kernel transpose
    bool memory_test = std::get<11>(GetParam());        // memory test

    test_gemmsup_ukr(kern_ptr, transa, transb, m, n, k, alpha, beta, storageC, MR, kern_trans, memory_test);

}// end of function


class sgemmUkrSUPPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sgemmsup_ker_ft, gtint_t, gtint_t, gtint_t, float, float, char, gtint_t, char, char, bool, bool>>  str) const {

        gtint_t m        = std::get<1>(str.param);
        gtint_t n        = std::get<2>(str.param);
        gtint_t k        = std::get<3>(str.param);
        float alpha      = std::get<4>(str.param);
        float beta       = std::get<5>(str.param);
        char storageC    = std::get<6>(str.param);
        char trnsa       = std::get<8>(str.param);
        char trnsb       = std::get<9>(str.param);
        bool memory_test = std::get<11>(str.param);
        std::string str_name;
        str_name = str_name + "_transa" + trnsa;
        str_name = str_name + "_transb" + trnsb;
        str_name = str_name + "_m" + std::to_string(m);
        str_name = str_name + "_n" + std::to_string(n);
        str_name = str_name + "_k" + std::to_string(k);
        str_name = str_name + "_alpha" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_beta" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_storage" + storageC;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16m_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16m_col_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x16m_col_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x16m),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16n_col_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(false),                               // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x16n_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x16n_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x16n),       // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(17), 1),           // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(false),                               // kernel pref
            ::testing::Values(true, false)                          // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64m_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64m_avx512), // sgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa
            ::testing::Values('n'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64m_col_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64m_avx512), // dgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(1), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('c'),                                  // storage of c
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('n'),                                  // transa
            ::testing::Values('t'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x64m_col_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x64m_avx512), // dgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('c'),                                  // storage of c
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa
            ::testing::Values('n'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rv_zen_asm_6x64n_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rv_zen_asm_6x64n_avx512), // dgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('t'),                                  // transa
            ::testing::Values('n'),                                  // transb
            ::testing::Values(true),                                 // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_sgemmsup_rd_zen_asm_6x64n_row_stored_c,
        sgemmUkrSUP,
        ::testing::Combine(
            ::testing::Values(bli_sgemmsup_rd_zen_asm_6x64n_avx512), // dgemm_sup kernel
            ::testing::Range(gtint_t(1), gtint_t(7), 1),             // values of m
            ::testing::Range(gtint_t(1), gtint_t(65), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),            // values of k
            ::testing::Values(2.0, 1.0, -1.0),                       // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                  // beta value
            ::testing::Values('r'),                                  // storage of c
            ::testing::Values(gtint_t(6)),                           // Micro kernel block MR
            ::testing::Values('n'),                                  // transa
            ::testing::Values('t'),                                  // transb
            ::testing::Values(false),                                // kernel pref
            ::testing::Values(true, false)                           // memory test
        ),
        ::sgemmUkrSUPPrint()
    );
#endif



class sgemmUkrNat :
        public ::testing::TestWithParam<std::tuple<sgemm_ukr_ft, gtint_t, float, float, char, gtint_t, gtint_t, bool>> {};
//sgemm native kernel, k, alpha, beta, storage of c, m, n, memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmUkrNat);

TEST_P(sgemmUkrNat, functionality_testing)
{
    using T = float;
    gtint_t k      = std::get<1>(GetParam());        // dimension k
    T alpha        = std::get<2>(GetParam());        // alpha
    T beta         = std::get<3>(GetParam());        // beta
    char storage   = std::get<4>(GetParam());        // indicates storage of all matrix operands
                                                     // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<5>(GetParam());             // MR of native kernel
    gtint_t n = std::get<6>(GetParam());             // NR of native kernel
    bool memory_test = std::get<7>(GetParam());      // memory test
    sgemm_ukr_ft kern_ptr = std::get<0>(GetParam()); //kernel's function pointer
    test_gemmnat_ukr(storage, m, n, k, alpha, beta, kern_ptr, memory_test);
}// end of function



class sgemmUkrNatPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sgemm_ukr_ft, gtint_t, float, float, char, gtint_t, gtint_t, bool>>  str) const {
        gtint_t k    = std::get<1>(str.param);
        float alpha  = std::get<2>(str.param);
        float beta   = std::get<3>(str.param);
        char storage = std::get<4>(str.param);
        bool memory_test = std::get<7>(str.param);
        std::string str_name;
        str_name = str_name + "_k" + std::to_string(k);
        str_name = str_name + "_alpha" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_beta" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_storage" + storage;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_sgemm_skx_asm_32x12_l2,
    sgemmUkrNat,
    ::testing::Combine(
        ::testing::Values(bli_sgemm_skx_asm_32x12_l2),
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(32),                          // values of m
        ::testing::Values(12),                          // values of n
        ::testing::Values(true, false)                  // memory test
    ),
    ::sgemmUkrNatPrint()
);


#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_sgemm_haswell_asm_6x16,
    sgemmUkrNat,
    ::testing::Combine(
        ::testing::Values(bli_sgemm_haswell_asm_6x16),
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(6),                           // values of m
        ::testing::Values(16),                          // values of n
        ::testing::Values(true, false)                  // memory test
    ),
    ::sgemmUkrNatPrint()
);
#endif

#if 0
/**
 * sgemm_small microkernel testing disable because sgemm_small is static local
 * function. Once it is made global, this testcase can be enabled.
 * As of now for the compilation sake, this testcase is kept disabled.
*/
#ifdef BLIS_ENABLE_SMALL_MATRIX

class SGemmSmallUkernelTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, float, float, char>> {};

//m, n, k, alpha, beta, storage scheme

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SGemmSmallUkernelTest);

TEST_P(SGemmSmallUkernelTest, gemm_small)
{
    using T = float;
    gtint_t m      = std::get<0>(GetParam());  // dimension m
    gtint_t n      = std::get<1>(GetParam());  // dimension n
    gtint_t k      = std::get<2>(GetParam());  // dimension k
    T alpha        = std::get<3>(GetParam());  // alpha
    T beta         = std::get<4>(GetParam());  // beta
    char storage   = std::get<5>(GetParam());  // indicates storage of all matrix operands


    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, 'n', k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, 'n', m, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, 'n', k, n, ldb );
    std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storage, 'n', m, n, ldc );

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
    double thresh = 10 * std::max(n,std::max(k,m)) * testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                 a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc);

    // Check component-wise error
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldc, thresh );

}// end of function



class SGemmSmallUkernelTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, float, float, char>>  str) const {
        gtint_t m     = std::get<0>(str.param);
        gtint_t n     = std::get<1>(str.param);
        gtint_t k     = std::get<2>(str.param);
        float alpha   = std::get<3>(str.param);
        float beta    = std::get<4>(str.param);
        char storage  = std::get<5>(str.param);
        std::string str_name;
        str_name = str_name + "_m" + std::to_string(m);
        str_name = str_name + "_n" + std::to_string(n);
        str_name = str_name + "_k" + std::to_string(k);
        str_name = str_name + "_alpha" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_beta" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_storage" + storage;

        return str_name;
    }
};


INSTANTIATE_TEST_SUITE_P (
        bli_sgemm_small,
        SGemmSmallUkernelTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(71), 1), // values of m
            ::testing::Range(gtint_t(1), gtint_t(21), 1), // values of n
            ::testing::Range(gtint_t(1), gtint_t(20), 1), // values of k
            ::testing::Values(2.0, 1.0, -1.0),            // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),       // beta value
            ::testing::Values('c')                        // storage
        ),
        ::SGemmSmallUkernelTestPrint()
    );

#endif
#endif
