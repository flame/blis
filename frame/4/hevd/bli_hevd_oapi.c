/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

err_t bli_hevd
     (
       const obj_t*  a,
       const obj_t*  v,
       const obj_t*  e
     )
{
	return bli_hevd_ex( a, v, e, NULL, NULL );
}

err_t bli_hevd_ex
     (
       const obj_t*  a,
       const obj_t*  v,
       const obj_t*  e,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_hevd_check( a, v, e, cntx );

	num_t     dt        = bli_obj_dt( a );

	uplo_t    uploa     = bli_obj_uplo( a );
	dim_t     m         = bli_obj_length( a );
	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     rs_a      = bli_obj_row_stride( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	void*     buf_v     = bli_obj_buffer_at_off( v );
	inc_t     rs_v      = bli_obj_row_stride( v );
	inc_t     cs_v      = bli_obj_col_stride( v );
	void*     buf_e     = bli_obj_buffer_at_off( e );
	inc_t     ince      = bli_obj_vector_inc( e );

	// Query a type-specific function pointer, except one that uses
	// void* for function arguments instead of typed pointers.
	hevd_ex_vft f = bli_hevd_ex_qfp( dt );

	return f
	(
	  TRUE,  // Always compute eigenvectors.
	  uploa,
	  m,
	  buf_a, rs_a, cs_a,
	  buf_v, rs_v, cs_v,
	  buf_e, ince,
	  NULL,  // work:  Request optimal size allocation for work array.
	  0,     // lwork: ignored if work == NULL.
	  NULL,  // rwork: Request optimal size allocation for rwork array.
	  NULL,
	  NULL
	);
}

#endif

