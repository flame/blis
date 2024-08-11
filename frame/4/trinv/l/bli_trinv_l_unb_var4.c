/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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

#include "blis.h"

#ifdef BLIS_ENABLE_LEVEL4

err_t bli_trinv_l_unb_var4
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm,
             l4_cntl_t* cntl
     )
{
	const dim_t m = bli_obj_length( a );

	obj_t    a00;
	obj_t    a10,   alpha11;
	obj_t    a20,   a21,       a22;

	for ( dim_t ij = 0; ij < m; ij += 1 )
	{
		bli_acquire_mparts_tl2br( ij, 1, a,
		                          &a00, NULL,     NULL,
		                          &a10, &alpha11, NULL,
		                          &a20, &a21,     &a22 );

		// a21 = -tril( A22 ) \ a21;
		bli_trsv_ex( &BLIS_MINUS_ONE, &a22, &a21, cntx, rntm );

		// A20 = -a21 * a10 + A20;
		bli_obj_set_struc( BLIS_GENERAL, &a20 );
		bli_ger_ex( &BLIS_MINUS_ONE, &a21, &a10, &a20, cntx, rntm );

		// a10   = a10 * tril( A00 );
		// a10^T = tril( A00 )^T * a10^T;
		bli_obj_set_onlytrans( BLIS_TRANSPOSE, &a00 );
		bli_trmv_ex( &BLIS_ONE, &a00, &a10, cntx, rntm );

		// alpha11 = 1.0 / alpha11;
		if ( bli_obj_has_nonunit_diag( a ) )
			bli_invertsc( &alpha11, &alpha11 );
	}

	return BLIS_SUCCESS;
}

#endif
