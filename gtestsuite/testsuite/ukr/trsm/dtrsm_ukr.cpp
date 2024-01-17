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
#include "test_trsm_ukr.h"
#include "level3/trsm/test_trsm.h"


class DTrsmUkrTest :
    public ::testing::TestWithParam<std::tuple< dgemmtrsm_ukr_ft,  // Function pointer type for daxpyv kernels
                                                char,              // storage
                                                char,              // uploa
                                                char,              // diaga
                                                gtint_t,           // m
                                                gtint_t,           // n
                                                gtint_t,           // k
                                                double,            // alpha
                                                gtint_t >> {};     // ldc_inc


TEST_P(DTrsmUkrTest, native)
{
    using   T = double;
    dgemmtrsm_ukr_ft ukr_fp = std::get<0>(GetParam());
    char storage            = std::get<1>(GetParam());
    char uploa              = std::get<2>(GetParam());
    char diaga              = std::get<3>(GetParam());
    gtint_t m               = std::get<4>(GetParam());
    gtint_t n               = std::get<5>(GetParam());
    gtint_t k               = std::get<6>(GetParam());
    T   alpha               = std::get<7>(GetParam());
    gtint_t ldc             = std::get<8>(GetParam());

    double thresh = 2 * m * testinghelpers::getEpsilon<T>();
    test_trsm_ukr<T, dgemmtrsm_ukr_ft>( ukr_fp, storage, uploa, diaga, m, n, k, alpha, ldc, thresh );
}

class DTrsmUkrTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<dgemmtrsm_ukr_ft, char, char, char, gtint_t,
                                            gtint_t, gtint_t, double, gtint_t>> str) const{
        char storage            = std::get<1>(str.param);
        char uploa              = std::get<2>(str.param);
        char diaga              = std::get<3>(str.param);
        gtint_t k               = std::get<6>(str.param);
        double  alpha           = std::get<7>(str.param);
        gtint_t ldc             = std::get<8>(str.param);
        return std::string("dgemmtrsm_ukr") + "_s" + storage + "_d" +  diaga + "_u" + uploa +
                "_k" + std::to_string(k) + "_a" +
                (alpha > 0 ? std::to_string(int(alpha)) : std::string("m") + std::to_string(int(alpha*-1))) +
                "_c" + std::to_string(ldc);
    }
};

#ifdef BLIS_KERNELS_ZEN4
INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_l_zen4_asm_8x24,
    DTrsmUkrTest,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_l_zen4_asm_8x24),  // ker_ptr
        ::testing::Values('c', 'r', 'g'),                  // stor
        ::testing::Values('l'),                            // uplo
        ::testing::Values('u', 'n'),                       // diaga
        ::testing::Values(8),                              // m
        ::testing::Values(24),                             // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),   // k
        ::testing::Values(-1, -5.2, 1, 8.9),               // alpha
        ::testing::Values(0, 9, 53)                        // ldc
    ),
    ::DTrsmUkrTestPrint()
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_u_zen4_asm_8x24,
    DTrsmUkrTest,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_u_zen4_asm_8x24),  // ker_ptr
        ::testing::Values('c', 'r', 'g'),                  // stor
        ::testing::Values('u'),                            // uplo
        ::testing::Values('u', 'n'),                       // diaga
        ::testing::Values(8),                              // m
        ::testing::Values(24),                             // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),   // k
        ::testing::Values(-1, -5.2, 1, 8.9),               // alpha
        ::testing::Values(0, 9, 53)                        // ldc
   ),
    ::DTrsmUkrTestPrint()
);
#endif


#ifdef BLIS_KERNELS_HASWELL
INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_l_haswell_asm_6x8,
    DTrsmUkrTest,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_l_haswell_asm_6x8), // ker_ptr
        ::testing::Values('c', 'r', 'g'),                   // stor
        ::testing::Values('l'),                             // uplo
        ::testing::Values('u', 'n'),                        // diaga
        ::testing::Values(6),                               // m
        ::testing::Values(8),                               // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),    // k
        ::testing::Values(-1, -5.2, 1, 8.9),                // alpha
        ::testing::Values(0, 9, 53)                         // ldc
    ),
    ::DTrsmUkrTestPrint()
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemmtrsm_u_haswell_asm_6x8,
    DTrsmUkrTest,
    ::testing::Combine(
        ::testing::Values(bli_dgemmtrsm_u_haswell_asm_6x8), // ker_ptr
        ::testing::Values('c', 'r', 'g'),                   // stor
        ::testing::Values('u'),                             // uplo
        ::testing::Values('u', 'n'),                        // diaga
        ::testing::Values(6),                               // m
        ::testing::Values(8),                               // n
        ::testing::Values(0, 1, 2, 8, 9, 10, 500, 1000),    // k
        ::testing::Values(-1, -5.2, 1, 8.9),                // alpha
        ::testing::Values(0, 9, 53)                         // ldc
    ),
    ::DTrsmUkrTestPrint()
);
#endif