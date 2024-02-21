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

#pragma once

#include "level1/scalv/scalv.h"
#include "level1/ref_scalv.h"
#include "inc/check_error.h"

/**
 * @brief Microkernel test body for scalv operation.
 */
template<typename T, typename U, typename FT>
static void test_scalv_ukr( FT ukr, char conja_alpha, gtint_t n, gtint_t incx, T alpha, double thresh, bool nan_inf_check )
{
    //----------------------------------------------------------
    //        Initialize vector with random numbers.
    //----------------------------------------------------------
    T *x, *x_ref;

    gtint_t size_x = testinghelpers::buff_dim( n, incx );

    x     = ( T* )malloc( sizeof( T ) * size_x );
    x_ref = ( T* )malloc( sizeof( T ) * size_x );

    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );

    // Copying x to x_ref, for comparision after computation
    memcpy( x_ref, x, size_x * sizeof( T ) );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    if constexpr ( testinghelpers::type_info<T>::is_complex && testinghelpers::type_info<U>::is_real )
        testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha.real, x_ref, incx );
    else    // if constexpr ( std::is_same<T,U>::value )
        testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha, x_ref, incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    conj_t blis_conjalpha;
    testinghelpers::char_to_blis_conj( conja_alpha, &blis_conjalpha );
    ukr( blis_conjalpha, n, &alpha, x, incx, nullptr );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, x, x_ref, incx, thresh, nan_inf_check );

    free( x );
    free( x_ref );
}
