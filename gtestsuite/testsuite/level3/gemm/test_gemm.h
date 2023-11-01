/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
    std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storage, 'n', m, n, ldc );

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
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldc, thresh );
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
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldc, thresh, true );
}
