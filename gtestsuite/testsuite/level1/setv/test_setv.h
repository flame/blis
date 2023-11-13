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

#include "setv.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for setv operation.
 */

template<typename T>
void test_setv( char conjalpha, gtint_t n, T alpha, gtint_t incx )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x( testinghelpers::buff_dim(n, incx), T{-1} );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    T alpha_ref = alpha;
    if( testinghelpers::chkconj( conjalpha ) ) {
        alpha_ref = testinghelpers::conj<T>( alpha );
    }

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    setv<T>( conjalpha, n, &alpha, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    gtint_t i,idx;
    for( idx = 0 ; idx < n ; idx++ )
    {
        i = (incx > 0) ? (idx * incx) : ( - ( n - idx - 1 ) * incx );
        EXPECT_EQ(x[i], alpha_ref) << "blis_sol[" << i << "]="<< x[i] <<"   ref = "  << alpha_ref;
    }
}
