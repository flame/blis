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

#include "gemm.h"
#include "level3/ref_gemm.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include <cfloat>

template<typename T>
void test_gemm( char storage, char trnsa, char trnsb, gtint_t m, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc, T alpha,
    T beta, double thresh )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, trnsa, m, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, trnsb, k, n, ldb );
    std::vector<T> c( testinghelpers::matsize( storage, 'n', m, n, ldc ) );
    if (beta != testinghelpers::ZERO<T>())
        testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, m, n, c.data(), 'n', ldc );
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, m, n, c.data(), 'n', ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemm<T>( storage, trnsa, trnsb, m, n, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemm<T>( storage, trnsa, trnsb, m, n, k, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "c", storage, m, n, c.data(), c_ref.data(), ldc, thresh );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test body used for exception value testing, by inducing an exception value
// in the index that is passed for each of the matrices.
/*
  (ai, aj) is the index with corresponding exception value aexval in matrix A.
  The index is with respect to the assumption that the matrix is column stored,
  without any transpose. In case of the row-storage and/or transpose, the index
  is translated from its assumption accordingly.
  Ex : (2, 3) with storage 'c' and transpose 'n' becomes (3, 2) if storage becomes
  'r' or transpose becomes 't'.
*/
// (bi, bj) is the index with corresponding exception value bexval in matrix B.
// (ci, cj) is the index with corresponding exception value cexval in matrix C.
template<typename T>
void test_gemm( char storage, char trnsa, char trnsb, gtint_t m, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc, T alpha,
    T beta, gtint_t ai, gtint_t aj, T aexval, gtint_t bi, gtint_t bj, T bexval,
    gtint_t ci, gtint_t cj, T cexval, double thresh )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, trnsa, m, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, trnsb, k, n, ldb );
    std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storage, 'n', m, n, ldc );

    // Inducing exception values onto the matrices based on the indices passed as arguments.
    // Assumption is that the indices are with respect to the matrices in column storage without
    // any transpose. In case of difference in storage scheme or transposition, the row and column
    // indices are appropriately swapped.
    testinghelpers::set_ev_mat( storage, trnsa, lda, ai, aj, aexval, a.data() );
    testinghelpers::set_ev_mat( storage, trnsb, ldb, bi, bj, bexval, b.data() );
    testinghelpers::set_ev_mat( storage, 'n', ldc, ci, cj, cexval, c.data() );

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemm<T>( storage, trnsa, trnsb, m, n, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemm( storage, trnsa, trnsb, m, n, k, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "c", storage, m, n, c.data(), c_ref.data(), ldc, thresh, true );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test body used for overflow and underflow checks
template<typename T>
void test_gemm( char storage, char trnsa, char trnsb, gtint_t over_under, gtint_t input_range,
                gtint_t m, gtint_t n, gtint_t k, gtint_t lda_inc, gtint_t ldb_inc,
                gtint_t ldc_inc, gtint_t ai, gtint_t aj, gtint_t bi, gtint_t bj,  T alpha,
                T beta, double thresh )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrices with random numbers
    //----------------------------------------------------------
    std::vector<T> a,b,c;

    /*
      Testing for Overflow
      ======================
      For double-precision floating point, the maximum representable number is
        DBL_MAX = 1.7976931348623158e+308

      Any value higher than DBL_MAX is considered to be an overflow.

      over_under=0 indicates Overflow testing
      The input matrices are populated with 3 different value ranges based on input_range

      |****************************************************************|
      | input_range |     Expected Input      |     Expected Output    |
      |*************|*************************|************************|
      |      -1     | Values much less than   | Exact floating point   |
      |             | DBL_MAX                 | values                 |
      |*************|*************************|************************|
      |      0      | Values close to         | Exact floating point   |
      |             | DBL_MAX                 | values upto DBL_MAX    |
      |             |                         |                        |
      |             |                         | +/-INF for values      |
      |             |                         | higher than +/-DBL_MAX |
      |*************|*************************|************************|
      |      1      | Values much higher than | +/-INF for values      |
      |             | DBL_MAX                 | higher than +/-DBL_MAX |
      |             |                         |                        |
      ******************************************************************

      Testing for Underflow
      ========================
      For double-precision floating point, the minimum representable number is
        DBL_MIN = 2.2250738585072014e-308

      Any value lower than DBL_MIN is considered to be an underflow

      over_under=1 indicates Underflow testing
      The input matrices are populated with 3 different value ranges based on input_range

      |******************************************************************|
      | input_range |     Expected Input       |     Expected Output     |
      |*************|**************************|*************************|
      |      -1     | Values much larger       | Exact floating point    |
      |             | than DBL_MIN             | values                  |
      |*************|**************************|*************************|
      |      0      | Values close to          | Exact floating point    |
      |             | DBL_MIN                  | values upto DBL_MIN     |
      |             |                          |                         |
      |             |                          | +0 for values           |
      |             |                          | lower than DBL_MIN      |
      |*************|**************************|*************************|
      |      1      | Values much smaller than | +0 for values           |
      |             | DBL_MIN                  | smaller than +/-DBL_MIN |
      |             |                          |                         |
      ********************************************************************

    */
    a = testinghelpers::get_random_matrix<T>( 5.5, 10.5, storage, trnsa, m, k, lda, 1,
                                              testinghelpers::datagenerators::ElementType::FP );
    b = testinghelpers::get_random_matrix<T>( 3.2, 5.6, storage, trnsb, k, n, ldb, 1,
                                              testinghelpers::datagenerators::ElementType::FP );
    c = testinghelpers::get_random_matrix<T>( -5, -2, storage, 'n', m, n, ldc, 1,
                                              testinghelpers::datagenerators::ElementType::FP );
    /*
      Based on the value of over_under, overflow/underflow values are inserted to the input matrices
      at the indices passed as arguments.
    */
    testinghelpers::set_overflow_underflow_mat( storage, trnsa, lda, ai, aj, a.data(), over_under, input_range);
    testinghelpers::set_overflow_underflow_mat( storage, trnsb, lda, bi, bj, b.data(), over_under, input_range);

    std::vector<T> c_ref(c);

    // Create a copy of c so that we can check reference results.
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemm<T>( storage, trnsa, trnsb, m, n, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemm<T>( storage, trnsa, trnsb, m, n, k, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "C", storage, m, n, c.data(), c_ref.data(), ldc, thresh, true );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gemmGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, T, T, gtint_t, gtint_t, gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char transa     = std::get<1>(str.param);
        char transb     = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha  = std::get<6>(str.param);
        T beta   = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, m, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};

template <typename T>
class gemmEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, gtint_t,
                                          gtint_t, T, gtint_t, gtint_t, T, gtint_t, gtint_t, T,
                                          T, T, gtint_t, gtint_t, gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char transa     = std::get<1>(str.param);
        char transb     = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        gtint_t ai, aj, bi, bj, ci, cj;
        T aex, bex, cex;
        ai  = std::get<6>(str.param);
        aj  = std::get<7>(str.param);
        aex = std::get<8>(str.param);

        bi  = std::get<9>(str.param);
        bj  = std::get<10>(str.param);
        bex = std::get<11>(str.param);

        ci  = std::get<12>(str.param);
        cj  = std::get<13>(str.param);
        cex = std::get<14>(str.param);

        T alpha  = std::get<15>(str.param);
        T beta   = std::get<16>(str.param);
        gtint_t lda_inc = std::get<17>(str.param);
        gtint_t ldb_inc = std::get<18>(str.param);
        gtint_t ldc_inc = std::get<19>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name = str_name + "_A" + std::to_string(ai) + std::to_string(aj);
        str_name = str_name + "_" + testinghelpers::get_value_string(aex);
        str_name = str_name + "_B" + std::to_string(bi) + std::to_string(bj);
        str_name = str_name + "_" + testinghelpers::get_value_string(bex);
        str_name = str_name + "_C" + std::to_string(ci) + std::to_string(cj);
        str_name = str_name + "_" + testinghelpers::get_value_string(cex);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, m, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};

template <typename T>
class gemmOUTPrint {
    public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t, T, T, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t, gtint_t>> str) const {
        char storage          = std::get<0>(str.param);
        char transa           = std::get<1>(str.param);
        char transb           = std::get<2>(str.param);
        gtint_t over_under    = std::get<3>(str.param);
        gtint_t input_range   = std::get<4>(str.param);
        gtint_t m             = std::get<5>(str.param);
        gtint_t n             = std::get<6>(str.param);
        gtint_t k             = std::get<7>(str.param);
        T alpha               = std::get<8>(str.param);
        T beta                = std::get<9>(str.param);
        gtint_t lda_inc       = std::get<10>(str.param);
        gtint_t ldb_inc       = std::get<11>(str.param);
        gtint_t ldc_inc       = std::get<12>(str.param);
        gtint_t ai            = std::get<13>(str.param);
        gtint_t aj            = std::get<14>(str.param);
        gtint_t bi            = std::get<15>(str.param);
        gtint_t bj            = std::get<16>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        std::string over_under_str = ( over_under > 0) ? "underflow": "overflow";
        str_name = str_name + "_" + over_under_str;
        std::string input_range_str = (input_range < 0) ? "within_limit": (input_range > 0) ? "beyond_limit" : "close_to_limit";
        str_name = str_name + "_" + input_range_str;
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name = str_name + "_A_" + std::to_string(ai) + "_" + std::to_string(aj);
        str_name = str_name + "_B_" + std::to_string(bi) + "_" + std::to_string(bj);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, m, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};
