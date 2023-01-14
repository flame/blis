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

err_t bli_chol_l_blk_var3
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm,
             cntl_t* cntl
     )
{
	err_t r_val = BLIS_SUCCESS;

	const dim_t m = bli_obj_length( a );

	dim_t b_alg;

	      /* a00 */
	obj_t /* a10 */ a11;
	obj_t /* a20 */ a21,   a22;

	for ( dim_t ij = 0; ij < m; ij += b_alg )
	{
		b_alg = bli_chol_determine_blocksize( ij, m, a, cntx, cntl );

		const dim_t mn_behind = ij;

		bli_acquire_mparts_tl2br( ij, b_alg, a,
		                          NULL, NULL, NULL,
		                          NULL, &a11, NULL,
		                          NULL, &a21, &a22 );

		// A11 = chol( A11 );
		// NOTE: If r_val is not BLIS_SUCCESS, it will be on the interval [1,b_alg].
		r_val = bli_chol_int( &a11, cntx, rntm, bli_cntl_sub_node( cntl ) );

		if ( r_val != BLIS_SUCCESS )
			return ( mn_behind + r_val );

		// A21 = A21 * tril( A11 )';
		bli_obj_set_conjtrans( BLIS_CONJ_TRANSPOSE, &a11 );
		bli_obj_set_struc( BLIS_TRIANGULAR, &a11 );
		bli_obj_set_struc( BLIS_GENERAL, &a21 );
		bli_trsm_ex( BLIS_RIGHT, &BLIS_ONE, &a11, &a21, cntx, rntm );

		// A22 = A22 - A21 * A21';
		bli_herk_ex( &BLIS_MINUS_ONE, &a21, &BLIS_ONE, &a22, cntx, rntm );
	}

	return r_val;
}

#endif
