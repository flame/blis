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

#include "symv.h"
#include "level2/ref_symv.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
    double thresh, char datatype ) {

    // Compute the leading dimensions of a.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', n, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 5, storage, 'n', n, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(-3, 3, n, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(-2, 5, n, incy, datatype);

    mksymm<T>( storage, uploa, n, a.data(), lda );
    mktrim<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    symv<T>( storage, uploa, conja, conjx, n, &alpha, a.data(), lda,
                                  x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_symv<T>( storage, uploa, conja, conjx, n, &alpha,
                 a.data(), lda, x.data(), incx, &beta, y_ref.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy, thresh );
}
