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

#include "trsm.h"
#include "level3/ref_trsm.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_trsm( char storage, char side, char uploa, char transa,
    char diaga, gtint_t m, gtint_t n, T alpha, gtint_t lda_inc,
    gtint_t ldb_inc, double thresh, char datatype ) {

    gtint_t mn;
    testinghelpers::set_dim_with_side( side, m, n, &mn );
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, mn, mn, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldb_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random values.
    //----------------------------------------------------------
    gtint_t lower = (diaga = 'n')||(diaga = 'N') ? 3 : 0;
    gtint_t upper = (diaga = 'n')||(diaga = 'N') ? 10 : 1;
    std::vector<T> a = testinghelpers::get_random_matrix<T>(lower, upper, storage, transa, mn, mn, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(3, 10, storage, 'n', m, n, ldb, datatype);

    // Making A diagonally dominant so that the condition number is good and
    // the algorithm doesn't diverge.
    for (gtint_t i=0; i<mn; i++)
    {
        a[i+i*lda] = T{float(mn)}*a[i+i*lda];
    }
    // Create a copy of v so that we can check reference results.
    std::vector<T> b_ref(b);

    mktrim<T>( storage, uploa, mn, a.data(), lda );
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    trsm<T>( storage, side, uploa, transa, diaga, m, n, &alpha, a.data(), lda, b.data(), ldb );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_trsm( storage, side, uploa, transa, diaga, m, n, alpha, a.data(), lda, b_ref.data(), ldb );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, b.data(), b_ref.data(), ldb, thresh );
}
