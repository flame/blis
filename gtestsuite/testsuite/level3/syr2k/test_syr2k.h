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

#pragma once

#include "syr2k.h"
#include "level3/ref_syr2k.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_syr2k( char storage, char uplo, char transa, char transb, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc, T alpha,
    T beta, double thresh )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, n, k, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, transa, n, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, transb, n, k, ldb );
    std::vector<T> c( testinghelpers::matsize( storage, 'n', n, n, ldc ) );
    if (beta != testinghelpers::ZERO<T>())
        // Since matrix C, stored in c, is symmetric and we only use the upper or lower
        // part in the computation of her2k and zero-out the rest to ensure
        // that code operates as expected.
        testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, uplo, n, c.data(), ldc );
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, n, c.data(), uplo, ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    syr2k<T>( storage, uplo, transa, transb, n, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_syr2k<T>( storage, uplo, transa, transb, n, k, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "C", storage, n, n, c.data(), c_ref.data(), ldc, thresh );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class syr2kGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, char, gtint_t, gtint_t, T, T, gtint_t, gtint_t, gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char uplo       = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha  = std::get<6>(str.param);
        T beta   = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uplo_" + std::string(&uplo, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, n, k, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};
