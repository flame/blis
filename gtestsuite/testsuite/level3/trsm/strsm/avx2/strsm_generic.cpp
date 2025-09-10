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
 * @brief Test STRSM small avx2 path all fringe cases
 *        Kernel size for avx2 small path is 16x6, testing in range of
 *        1 to 16 ensures all finge cases are being tested.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2_fringe,
        strsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(19), 3),                    // m
            ::testing::Range(gtint_t(1), gtint_t(22), 4),                    // n
            ::testing::Values(-2.4f),                                        // alpha
            ::testing::Values(gtint_t(58)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(31))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<float>()
    );


/**
 * @brief Test STRSM small avx2 path, this code path is used in range 0 to 1000
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2,
        strsmGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 110, 51, 1000),                            // m
            ::testing::Values(17, 48 , 51, 1000),                            // n
            ::testing::Values(-2.4f),                                        // alpha
            ::testing::Values(gtint_t(95)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(83))                                   // increment to the leading dim of b
        ),
        ::trsmGenericPrint<float>()
    );
