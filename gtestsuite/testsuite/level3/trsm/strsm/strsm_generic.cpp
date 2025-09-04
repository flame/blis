/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level3/trsm/test_trsm.h"

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(strsmGeneric);

/**
 * @brief Test STRSM native path, which starts from size 1000 for BLAS api
 *        and starts from size 0 for BLIS api.
 */
INSTANTIATE_TEST_SUITE_P(
        Native,
        strsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 2, 112, 1200),                              // m
            ::testing::Values(1, 2, 154, 1317),                              // n
            ::testing::Values(-2.0f),                                        // alpha
            ::testing::Values(gtint_t(45)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(38))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<float>()
    );

/**
 * @brief Test STRSM with differnt values of alpha
 *      code paths covered:
 *          TRSV              -> 1
 *          TRSM_AVX2_small   -> 3
 *          TRSM_NATIVE       -> 1001
 */
INSTANTIATE_TEST_SUITE_P(
        Alpha,
        strsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 3, 1001),                                   // n
            ::testing::Values(1, 3, 1001),                                   // m
            ::testing::Values(-2.4f, 0.0f, 1.0f, 3.1f),                      // alpha
            ::testing::Values(gtint_t(0), gtint_t(35)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(39))                       // increment to the leading dim of b
        ),
        ::trsmGenericPrint<float>()
    );
