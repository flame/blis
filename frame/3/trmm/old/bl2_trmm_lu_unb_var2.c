/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"

#define FUNCPTR_T trmm_fp

typedef void (*FUNCPTR_T)(
                           trans_t transa,
                           diag_t  diag,
                           dim_t   m,
                           dim_t   n,
                           void*   alpha,
                           void*   a, inc_t rs_a, inc_t cs_a,
                           void*   b, inc_t rs_b, inc_t cs_b
                         );

static FUNCPTR_T ftypes[BLIS_NUM_FP_TYPES] =
{
	bl2_strmm_lu_unb_var2,
	bl2_ctrmm_lu_unb_var2,
	bl2_dtrmm_lu_unb_var2,
	bl2_ztrmm_lu_unb_var2
};

void bl2_trmm_lu_unb_var2( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           trmm_t* cntl )
{
	num_t     dt_a      = bl2_obj_datatype( *a );

	trans_t   transa    = bl2_obj_conjtrans_status( *a );
	diag_t    diag      = bl2_obj_diag( *a );

	dim_t     m         = bl2_obj_length( *b );
	dim_t     n         = bl2_obj_width( *b );

	void*     buf_a     = bl2_obj_buffer_at_off( *a );
	inc_t     rs_a      = bl2_obj_row_stride( *a );
	inc_t     cs_a      = bl2_obj_col_stride( *a );

	void*     buf_b     = bl2_obj_buffer_at_off( *b );
	inc_t     rs_b      = bl2_obj_row_stride( *b );
	inc_t     cs_b      = bl2_obj_col_stride( *b );

	void*     buf_alpha = bl2_obj_scalar_buffer( dt_a, *alpha );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_a];

	// Invoke the function.
	f( transa,
	   diag,
	   m,
	   n,
	   buf_alpha,
	   buf_a, rs_a, cs_a,
	   buf_b, rs_b, cs_b );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname)( \
                           trans_t transa, \
                           diag_t  diag, \
                           dim_t   m, \
                           dim_t   n, \
                           void*   alpha, \
                           void*   a, inc_t rs_a, inc_t cs_a, \
                           void*   b, inc_t rs_b, inc_t cs_b  \
                         ) \
{ \
	ctype* alpha_cast = alpha; \
	ctype* a_cast     = a; \
	ctype* b_cast     = b; \
	ctype* a01; \
	ctype* alpha11; \
	ctype* b0; \
	ctype* b1; \
	ctype  alpha_alpha11_conj; \
	dim_t  iter, i; \
	dim_t  n_behind; \
	conj_t conja; \
\
	if ( bl2_zero_dim2( m, n ) ) return; \
\
	conja = bl2_extract_conj( transa ); \
\
	for ( iter = 0; iter < m; ++iter ) \
	{ \
		i        = iter; \
		n_behind = i; \
		a01      = a_cast + (0  )*rs_a + (i  )*cs_a; \
		alpha11  = a_cast + (i  )*rs_a + (i  )*cs_a; \
		b0       = b_cast + (0  )*rs_b + (0  )*cs_b; \
		b1       = b_cast + (i  )*rs_b + (0  )*cs_b; \
\
		/* B0 = B0 + alpha * a01 * b1; */ \
		PASTEMAC(ch,ger)( conja, \
		                  BLIS_NO_CONJUGATE, \
		                  n_behind, \
		                  n, \
		                  alpha_cast, \
		                  a01, rs_a, \
		                  b1,  cs_b, \
		                  b0,  rs_b, cs_b ); \
\
		/* b1 = alpha * alpha11 * b1; */ \
		PASTEMAC2(ch,ch,copys)( *alpha_cast, alpha_alpha11_conj ); \
\
		if ( bl2_is_nonunit_diag( diag ) ) \
			PASTEMAC2(ch,ch,scalcjs)( conja, *alpha11, alpha_alpha11_conj ); \
\
		PASTEMAC2(ch,ch,scalv)( BLIS_NO_CONJUGATE, \
		                        n, \
		                        &alpha_alpha11_conj, \
		                        b1, cs_b ); \
	} \
}

INSERT_GENTFUNC_BASIC( trmm, trmm_lu_unb_var2 )

