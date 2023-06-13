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

err_t bli_chol_l_unb_var1
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm,
             cntl_t* cntl
     )
{
	const dim_t m = bli_obj_length( a );

	obj_t    a00;
	obj_t    a10,   alpha11;
	      /* a20,   a21,       a22 */

	for ( dim_t ij = 0; ij < m; ij += 1 )
	{
		const dim_t mn_behind = ij;

		bli_acquire_mparts_tl2br( ij, 1, a,
		                          &a00, NULL,     NULL,
		                          &a10, &alpha11, NULL,
		                          NULL, NULL,     NULL );

		// a10   = a10 / tril( A00 )';
		// a10^T = conj( tril( A00 ) ) / a10^T;
		bli_obj_set_conj( BLIS_CONJUGATE, &a00 );
		bli_obj_set_struc( BLIS_TRIANGULAR, &a00 );
		bli_trsv_ex( &BLIS_ONE, &a00, &a10, cntx, rntm );

		// alpha11 = alpha11 - a10 * a10';
		obj_t a10_conj;
		bli_obj_alias_with_conj( BLIS_CONJUGATE, &a10, &a10_conj );
		bli_dotxv_ex( &BLIS_MINUS_ONE, &a10, &a10_conj, &BLIS_ONE, &alpha11, cntx, rntm );

		// Check if alpha11 is positive. If it is positive, we may proceed
		// with the square root. If alpha11 is not positive, then the matrix
		// is not Hermitian positive definite, and so we must return an error
		// code.
		bool is_lte0; bli_ltesc( &alpha11, &BLIS_ZERO, &is_lte0 );
		if ( is_lte0 ) return ( mn_behind + 1 );

		// [ alpha11, 0.0 ] = sqrt( real(alpha11) );
		bli_sqrtrsc( &alpha11, &alpha11 );
	}

	return BLIS_SUCCESS;
}

#endif
