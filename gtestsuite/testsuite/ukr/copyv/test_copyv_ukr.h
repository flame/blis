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

#include "level1/copyv/copyv.h"
#include "level1/ref_copyv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for copyv operation.
 */

template<typename T, typename FT>
static void test_copyv_ukr( FT ukr_fp, char conjx, gtint_t n, gtint_t incx, gtint_t incy, double thresh )
{
    //----------------------------------------------------------
    //        Allocate the fixed memory and initialize 
    //        vectors with random numbers.
    //----------------------------------------------------------

    T *x, *y, *y_ref;
    gtint_t size_x = testinghelpers::buff_dim( n, incx );
    gtint_t size_y = testinghelpers::buff_dim( n, incy );
    x = ( T* )malloc( sizeof( T ) * size_x );
    y = ( T* )malloc( sizeof( T ) * size_y );
    y_ref = ( T* )malloc( sizeof( T ) * size_y );

    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incy, y );

    // Copying y to y_ref, for comparision after computation
    for( gtint_t i = 0; i < size_y; i += 1 )
      *( y_ref + i ) = *( y + i );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------

    testinghelpers::ref_copyv<T>( conjx, n, x, incx, y_ref, incy );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    conj_t blis_conjx;
    testinghelpers::char_to_blis_conj( conjx, &blis_conjx );
    ukr_fp( blis_conjx, n, x, incx, y, incy, nullptr );

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( n, y, y_ref, incy );

    free( x );
    free( y );
    free( y_ref );
}