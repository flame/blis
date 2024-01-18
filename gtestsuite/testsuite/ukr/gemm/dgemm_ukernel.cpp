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

class DGEMMUkrSUPTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, dgemmsup_ker_ft, gtint_t, char, char, bool>> {};
// m, n, k, alpha, beta,  storage of c, dgemm sup kernel, micro-kernel MR block, transa, transb

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DGEMMUkrSUPTest);

TEST_P(DGEMMUkrSUPTest, sup_kernel)
{
    using T = double;
    gtint_t m      = std::get<0>(GetParam());  // dimension m
    gtint_t n      = std::get<1>(GetParam());  // dimension n
    gtint_t k      = std::get<2>(GetParam());  // dimension k
    T alpha        = std::get<3>(GetParam());  // alpha
    T beta         = std::get<4>(GetParam());  // beta
    char storageC   = std::get<5>(GetParam()); // storage scheme for C matrix
    dgemmsup_ker_ft kern_ptr = std::get<6>(GetParam()); //pointer to the gemm kernel
    gtint_t MR  = std::get<7>(GetParam());
    char transa = std::get<8>(GetParam());
    char transb = std::get<9>(GetParam());
    bool row_pref = std::get<10>(GetParam());

    test_dgemmsup_ukr(kern_ptr, transa, transb, m, n, k, alpha, beta, storageC, MR, row_pref);

}// end of function


class DGEMMukrsupTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, dgemmsup_ker_ft, gtint_t, char, char, bool>>  str) const {

        gtint_t m       = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t k       = std::get<2>(str.param);
        double alpha    = std::get<3>(str.param);
        double beta     = std::get<4>(str.param);
        char storageC    = std::get<5>(str.param);
        char trnsa       = std::get<8>(str.param);
        char trnsb       = std::get<9>(str.param);

        std::string str_name = "dgemmsup_ukr";
        str_name = str_name + "_" + trnsa;
        str_name = str_name + "_" + trnsb;
        str_name = str_name + "_m" + std::to_string(m);
        str_name = str_name + "_n" + std::to_string(n);
        str_name = str_name + "_k" + std::to_string(k);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + storageC;

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8m_row_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8m_col_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rd_haswell_asm_6x8m_col_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rd_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );


INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8n_col_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8n_row_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rd_haswell_asm_6x8n_col_stored_c,
        DGEMMUkrSUPTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rd_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true)                                 // row preferred kernel?
        ),
        ::DGEMMukrsupTestPrint()
    );
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)

 INSTANTIATE_TEST_SUITE_P (
         bli_dgemmsup_rv_zen4_asm_24x8m_col_stored_c,
         DGEMMUkrSUPTest,
         ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(25), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_zen4_asm_24x8m),      // dgemm_sup kernel
            ::testing::Values(gtint_t(8)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(false)                                // row preferred kernel?
         ),
         ::DGEMMukrsupTestPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_dgemmsup_rv_zen4_asm_24x8m_row_stored_c,
         DGEMMUkrSUPTest,
         ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(25), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_zen4_asm_24x8m),      // dgemm_sup kernel
            ::testing::Values(gtint_t(8)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(false)                                // row preferred kernel?
         ),
         ::DGEMMukrsupTestPrint()
     );
#endif

class DGEMMUkrNatTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, double, double, char, gtint_t, gtint_t, dgemm_ukr_ft>> {};
// k, alpha, beta, storage of c, m, n, dgemm native kernel

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DGEMMUkrNatTest);

TEST_P(DGEMMUkrNatTest, native_kernel_testing)
{
    using T = double;
    gtint_t k      = std::get<0>(GetParam());  // dimension k
    T alpha        = std::get<1>(GetParam());  // alpha
    T beta         = std::get<2>(GetParam());  // beta
    char storage   = std::get<3>(GetParam());  // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<4>(GetParam());
    gtint_t n = std::get<5>(GetParam());
    dgemm_ukr_ft kern_ptr = std::get<6>(GetParam());
    test_gemmnat_ukr(storage, m, n, k, alpha, beta, kern_ptr);
}// end of function



class DGEMMukrnatTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, double, double, char, gtint_t, gtint_t, dgemm_ukr_ft>>  str) const {
        gtint_t k       = std::get<0>(str.param);
        double alpha    = std::get<1>(str.param);
        double beta     = std::get<2>(str.param);
        char storage    = std::get<3>(str.param);

        std::string str_name = "dgemmnat_ukr";
        str_name = str_name + "_" + std::to_string(k);
               str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);;
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);;
        str_name = str_name + "_" + storage; //std::to_string(storage);

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_zen4_asm_32x6,
    DGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(32),                          // values of m
        ::testing::Values(6),                           // values of n
        ::testing::Values(bli_dgemm_zen4_asm_32x6)
    ),
    ::DGEMMukrnatTestPrint()
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_zen4_asm_8x24,
    DGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1), // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(8),                           // values of m
        ::testing::Values(24),                          // values of n
        ::testing::Values(bli_dgemm_zen4_asm_8x24)
    ),
    ::DGEMMukrnatTestPrint()
);
#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_haswell_asm_6x8,
    DGEMMUkrNatTest,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1), // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(6),                           // values of m
        ::testing::Values(8),                           // values of n
        ::testing::Values(bli_dgemm_haswell_asm_6x8)
    ),
    ::DGEMMukrnatTestPrint()
);
#endif

//Function pointer specific to dgemm kernel that handles
//special case where k=1.
typedef err_t (*gemm_k1_kernel)
     (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        double*    alpha,
        double*    a, const inc_t lda,
        double*    b, const inc_t ldb,
        double*    beta,
        double*    c, const inc_t ldc
    );

//Since AOCL BLAS is having separate kernel optimized to handle k=1 cases
//dgemm computation, a micro-kernel testing added that validates dgemm kernel
//for k=1 case.

class DGEMMUkrk1Test :
        public ::testing::TestWithParam<std::tuple<double, double, char, gtint_t, gtint_t, gemm_k1_kernel>> {};
// k, alpha, beta, storage of c, m, n, dgemm k1 kernel

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DGEMMUkrk1Test);

TEST_P(DGEMMUkrk1Test, k1_kernel_testing)
{
    using T = double;
    gtint_t k      = 1;
    T alpha        = std::get<0>(GetParam());  // alpha
    T beta         = std::get<1>(GetParam());  // beta
    char storage   = std::get<2>(GetParam());  // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<3>(GetParam());
    gtint_t n = std::get<4>(GetParam());
    gemm_k1_kernel kern_ptr = std::get<5>(GetParam());
    test_gemmk1_ukr(kern_ptr, m, n, k, storage, alpha, beta);
}// end of function



class DGEMMukrk1TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<double, double, char, gtint_t, gtint_t, gemm_k1_kernel>>  str) const {
        gtint_t k       = 1;
        double alpha    = std::get<0>(str.param);
        double beta     = std::get<1>(str.param);
        char storage    = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);

        std::string str_name = "dgemmk1_ukr";
        str_name = str_name + "_" + std::to_string(k);
        str_name = str_name + "_a" + testinghelpers::get_value_string(alpha);;
        str_name = str_name + "_b" + testinghelpers::get_value_string(beta);;
        str_name = str_name + "_m" + std::to_string(m);
        str_name = str_name + "_n" + std::to_string(n);
        str_name = str_name + "_" + storage; //std::to_string(storage);

        return str_name;
    }
};


#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_24x8_avx512_k1_nn,
    DGEMMUkrk1Test,
    ::testing::Combine(

        ::testing::Values(2.0, 1.0, -1.0),             // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),        // beta value
        ::testing::Values('c'),                        // storage
        ::testing::Range(gtint_t(1), gtint_t(25), 1),  // values of m
        ::testing::Range(gtint_t(1), gtint_t(9), 1),   // values of n
        ::testing::Values(bli_dgemm_24x8_avx512_k1_nn)
    ),
    ::DGEMMukrk1TestPrint()
);

#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_8x6_avx2_k1_nn,
    DGEMMUkrk1Test,
    ::testing::Combine(
        ::testing::Values(2.0, 1.0, -1.0),           // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),      // beta value
        ::testing::Values('c'),                      // storage
        ::testing::Range(gtint_t(1), gtint_t(9), 1), // values of m
        ::testing::Range(gtint_t(1), gtint_t(7), 1), // values of n
        ::testing::Values(bli_dgemm_8x6_avx2_k1_nn)
    ),
    ::DGEMMukrk1TestPrint()
);
#endif


#ifdef BLIS_ENABLE_SMALL_MATRIX

class DGemmSmallUkernelTest :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DGemmSmallUkernelTest);

//m, n, k, alpha, beta, storage scheme
TEST_P(DGemmSmallUkernelTest, gemm_small)
{
    using T = double;
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

    const num_t dt = BLIS_DOUBLE;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao = BLIS_OBJECT_INITIALIZER;
    obj_t       bo = BLIS_OBJECT_INITIALIZER;
    obj_t       betao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co = BLIS_OBJECT_INITIALIZER;

    dim_t       m0_a, n0_a;
    dim_t       m0_b, n0_b;

    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, m, k, &m0_a, &n0_a);
    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, k, n, &m0_b, &n0_b);

    bli_obj_init_finish_1x1(dt, (double*)&alpha, &alphao);
    bli_obj_init_finish_1x1(dt, (double*)&beta, &betao);

    bli_obj_init_finish(dt, m0_a, n0_a, (double*)a.data(), 1, lda, &ao);
    bli_obj_init_finish(dt, m0_b, n0_b, (double*)b.data(), 1, ldb, &bo);
    bli_obj_init_finish(dt, m, n, (double*)c.data(), 1, ldc, &co);

    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &ao);
    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &bo);


    bli_dgemm_small ( &alphao,
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



class DGemmSmallUkernelTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char>>  str) const {
        gtint_t m       = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t k       = std::get<2>(str.param);
        double alpha    = std::get<3>(str.param);
        double beta     = std::get<4>(str.param);
        char storage    = std::get<5>(str.param);

        std::string str_name = "gemmsmall_ukr";
        str_name = str_name + "_m" + std::to_string(m);
        str_name = str_name + "_n" + std::to_string(n);
        str_name = str_name + "_k" + std::to_string(k);
        std::string alpha_str = ( alpha > 0) ? std::to_string(int(alpha)) : "m" + std::to_string(int(std::abs(alpha)));
        str_name = str_name + "_a" + alpha_str;
        std::string beta_str = ( beta > 0) ? std::to_string(int(beta)) : "m" + std::to_string(int(std::abs(beta)));
        str_name = str_name + "_b" + beta_str;
        str_name = str_name + "_" + storage; //std::to_string(storage);

        return str_name;
    }
};


INSTANTIATE_TEST_SUITE_P (
        bli_dgemm_small,
        DGemmSmallUkernelTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(21), 1), // values of m
            ::testing::Range(gtint_t(1), gtint_t(11), 1), // values of n
            ::testing::Range(gtint_t(1), gtint_t(20), 1), // values of k
            ::testing::Values(2.0, 1.0, -1.0),            // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),       // beta value
            ::testing::Values('c')                        // storage
        ),
        ::DGemmSmallUkernelTestPrint()
    );

#endif