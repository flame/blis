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

err_t bli_ttmm_u_opt_var1
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm,
             cntl_t* cntl
     )
{
	num_t     dt        = bli_obj_dt( a );

	uplo_t    uploa     = bli_obj_uplo( a );
	dim_t     m         = bli_obj_length( a );
	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     rs_a      = bli_obj_row_stride( a );
	inc_t     cs_a      = bli_obj_col_stride( a );

	if ( bli_error_checking_is_enabled() )
		PASTEMAC(ttmm,_check)( a, cntx );

	// Query a type-specific function pointer, except one that uses
	// void* for function arguments instead of typed pointers.
	ttmm_opt_vft f = PASTEMAC(ttmm_u_opt_var1,_qfp)( dt );

	return
	f
	(
	  uploa,
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
            dim_t   m, \
            ctype*  a, inc_t rs_a, inc_t cs_a, \
      const cntx_t* cntx, \
            rntm_t* rntm  \
     ) \
{ \
	const ctype_r one_r = *PASTEMAC(chr,1); \
\
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		const dim_t mn_behind = i; \
\
		/* Identify subpartitions: /  a00  a01      ---  \
		                           |       alpha11  ---  |
		                           \                ---  / */ \
		ctype*   a00       = a + (0  )*rs_a + (0  )*cs_a; \
		ctype*   a01       = a + (0  )*rs_a + (i  )*cs_a; \
		ctype*   alpha11   = a + (i  )*rs_a + (i  )*cs_a; \
\
		/* A00 = A00 + a01 * a01'; */ \
		PASTEMAC2(ch,her,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_UPPER, \
		  BLIS_NO_CONJUGATE, \
		  mn_behind, \
		  &one_r, \
		  a01, rs_a, \
		  a00, rs_a, cs_a, \
		  cntx, \
		  rntm \
		); \
\
		/* a01 = alpha11 * a01; */ \
		PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  mn_behind, \
		  alpha11, \
		  a01, rs_a, \
		  cntx, \
		  rntm \
		); \
\
		/* alpha11 = alpha11 * alpha11'; */ \
		PASTEMAC(ch,absq2s)( *alpha11, *alpha11 ); \
	} \
\
	return BLIS_SUCCESS; \
}

INSERT_GENTFUNCR_BASIC( ttmm_u_opt_var1 )

#endif
