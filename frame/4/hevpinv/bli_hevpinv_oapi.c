/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

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

//
// Define object-based interfaces.
//

#if 0

err_t bli_hevpinv
     (
       const obj_t* a,
       const obj_t* pinv
     )
{
	const double thresh = 1.0e-13;

	obj_t v, e;

	num_t dt   = bli_obj_dt( a );
	num_t dt_r = bli_obj_dt_proj_to_real( a );
	dim_t m    = bli_obj_length( a );

	// Create a temporary matrix for the eigenvectors and a vector for
	// eigenvalues. (Specifying column storage for V and unit stride for e
	// saves the underlying bli_?hevd() implementation a little bit of work.)
	bli_obj_create( dt,   m, m, 0, 0, &v );
	bli_obj_create( dt_r, m, 1, 0, 0, &e );

	// Perform a Hermitian EVD: A -> V * Ie * V^H.
	bli_hevd( a, &v, &e );

	// Invert the eigenvalues above a threshold (and zero out the ones
	// below the threshold).
	bli_inverttv( thresh, &e );

	// Perform a reverse Hermitian EVD: Pinv := V * Ie * V^H.
	// Pinv now contains the psuedo-inverse of A (or the true inverse if
	// all of the eigenvalues were above the threshold).
	bli_rhevd( &v, &e, pinv );

	// Make the matrix densely/explicitly Hermitian.
	bli_mkherm( pinv );

	// Free the 
	bli_obj_free( &v );
	bli_obj_free( &e );

	return BLIS_SUCCESS;
}

#else

err_t bli_hevpinv
     (
             double  thresh,
       const obj_t*  a,
       const obj_t*  p
     )
{
	return bli_hevpinv_ex( thresh, a, p, NULL, NULL );
}

err_t bli_hevpinv_ex
     (
             double  thresh,
       const obj_t*  a,
       const obj_t*  p,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_hevpinv_check( thresh, a, p, cntx );

	num_t     dt        = bli_obj_dt( a );

	uplo_t    uploa     = bli_obj_uplo( a );
	dim_t     m         = bli_obj_length( a );
	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     rs_a      = bli_obj_row_stride( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	void*     buf_p     = bli_obj_buffer_at_off( p );
	inc_t     rs_p      = bli_obj_row_stride( p );
	inc_t     cs_p      = bli_obj_col_stride( p );

	// Query a type-specific function pointer, except one that uses
	// void* for function arguments instead of typed pointers.
	hevpinv_ex_vft f = bli_hevpinv_ex_qfp( dt );

	return f
	(
	  thresh,
	  uploa,
	  m,
	  buf_a, rs_a, cs_a,
	  buf_p, rs_p, cs_p,
	  cntx,
	  rntm
	);
}

#endif

#endif

