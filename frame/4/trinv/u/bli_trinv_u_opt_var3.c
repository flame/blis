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

err_t bli_trinv_u_opt_var3
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm,
             l4_cntl_t* cntl
     )
{
	num_t     dt        = bli_obj_dt( a );

	uplo_t    uploa     = bli_obj_uplo( a );
	diag_t    diaga     = bli_obj_diag( a );
	dim_t     m         = bli_obj_length( a );
	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     rs_a      = bli_obj_row_stride( a );
	inc_t     cs_a      = bli_obj_col_stride( a );

	if ( bli_error_checking_is_enabled() )
		PASTEMAC(trinv,_check)( a, cntx );

	// Query a type-specific function pointer, except one that uses
	// void* for function arguments instead of typed pointers.
	trinv_opt_vft f = PASTEMAC(trinv_u_opt_var3,_qfp)( dt );

	return
	f
	(
	  uploa,
	  diaga,
	  m,
	  buf_a, rs_a, cs_a,
	  cntx,
	  rntm
	);
}

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
err_t PASTEMAC(ch,varname) \
     ( \
            uplo_t  uploa, \
            diag_t  diaga, \
            dim_t   m, \
            ctype*  a, inc_t rs_a, inc_t cs_a, \
      const cntx_t* cntx, \
            rntm_t* rntm  \
     ) \
{ \
\
	const ctype one       = *PASTEMAC(ch,1); \
	const ctype minus_one = *PASTEMAC(ch,m1); \
\
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		const dim_t mn_behind = i; \
		const dim_t mn_ahead  = m - i - 1; \
\
		/* Identify subpartitions: /  ---  a01      a02  \
		                           |  0    alpha11  a12  |
		                           \  0    0        ---  / */ \
		ctype*   a01       = a + (0  )*rs_a + (i  )*cs_a; \
		ctype*   a02       = a + (0  )*rs_a + (i+1)*cs_a; \
		ctype*   alpha11   = a + (i  )*rs_a + (i  )*cs_a; \
		ctype*   a12       = a + (i  )*rs_a + (i+1)*cs_a; \
\
		/* a12 = -a12 / alpha11; */ \
		if ( bli_is_nonunit_diag( diaga ) ) \
		{ \
			ctype alpha11_m1; \
\
			PASTEMAC(ch,scal2s)( minus_one, *alpha11, alpha11_m1 ) \
			PASTEMAC(ch,invscalv,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  mn_ahead, \
			  &alpha11_m1, \
			  a12, cs_a, \
			  cntx, \
			  rntm \
			); \
		} \
		else \
		{ \
			PASTEMAC(ch,scalv,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  mn_ahead, \
			  &minus_one, \
			  a12, cs_a, \
			  cntx, \
			  rntm \
			); \
		} \
\
		/* A02 = a01 * a12 + A02; */ \
		PASTEMAC(ch,ger,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  BLIS_NO_CONJUGATE, \
		  mn_behind, \
		  mn_ahead, \
		  &one, \
		  a01, rs_a, \
		  a12, cs_a, \
		  a02, rs_a, cs_a, \
		  cntx, \
		  rntm \
		); \
\
		/* a01 = a01 / alpha11; */ \
		if ( bli_is_nonunit_diag( diaga ) ) \
			PASTEMAC(ch,invscalv,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  mn_behind, \
			  alpha11, \
			  a01, rs_a, \
			  cntx, \
			  rntm \
			); \
\
		/* alpha11 = 1.0 / alpha11; */ \
		if ( bli_is_nonunit_diag( diaga ) ) \
			PASTEMAC(ch,inverts)( *alpha11 ); \
	} \
\
	return BLIS_SUCCESS; \
}

INSERT_GENTFUNCR_BASIC( trinv_u_opt_var3 )

#endif
