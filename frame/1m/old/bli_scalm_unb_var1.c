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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#define FUNCPTR_T scalm_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conjalpha,
                           doff_t  diagoffx,
                           diag_t  diagx,
                           uplo_t  uplox,
                           dim_t   m,
                           dim_t   n,
                           void*   alpha,
                           void*   x, inc_t rs_x, inc_t cs_x
                         );

static FUNCPTR_T GENARRAY_MIN(ftypes,scalm_unb_var1);


void bli_scalm_unb_var1( obj_t*  alpha,
                         obj_t*  x,
                         cntx_t* cntx )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	uplo_t    diagx     = bli_obj_diag( *x );
	uplo_t    uplox     = bli_obj_uplo( *x );

	dim_t     m         = bli_obj_length( *x );
	dim_t     n         = bli_obj_width( *x );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );

	void*     buf_alpha;

	obj_t     x_local;

	FUNCPTR_T f;

	// Alias x to x_local so we can apply alpha if it is non-unit.
	bli_obj_alias_to( *x, x_local );

	// If alpha is non-unit, apply it to the scalar attached to x.
	if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
	{
		bli_obj_scalar_apply_scalar( alpha, &x_local );
	}

	// Grab the address of the internal scalar buffer for the scalar
	// attached to x.
	buf_alpha_x = bli_obj_internal_scalar_buffer( *x );

	// Index into the type combination array to extract the correct
	// function pointer.
	// NOTE: We use dt_x for both alpha and x because alpha was obtained
	// from the attached scalar of x, which is guaranteed to be of the
	// same datatype as x.
	f = ftypes[dt_x][dt_x];

	// Invoke the function.
	// NOTE: We unconditionally pass in BLIS_NO_CONJUGATE for alpha
	// because it would have already been conjugated by the front-end.
	f( BLIS_NO_CONJUGATE,
	   diagoffx,
	   diagx,
	   uplox,
	   m,
	   n,
	   buf_alpha,
	   buf_x, rs_x, cs_x );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conjalpha, \
                           doff_t  diagoffx, \
                           doff_t  diagx, \
                           uplo_t  uplox, \
                           dim_t   m, \
                           dim_t   n, \
                           void*   alpha, \
                           void*   x, inc_t rs_x, inc_t cs_x \
                         ) \
{ \
	ctype* alpha_cast = alpha; \
	ctype* x_cast     = x; \
	ctype* x1; \
	uplo_t uplox_eff; \
	dim_t  n_iter; \
	dim_t  n_elem, n_elem_max; \
	inc_t  ldx, incx; \
	dim_t  j, i; \
	dim_t  ij0, n_shift; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If alpha is unit, the entire operation is a no-op. */ \
	if ( PASTEMAC(chb,eq1)( *alpha_cast ) ) return; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_1m( diagoffx, diagx, \
	                           uplox, m, n, rs_x, cs_x, \
	                           uplox_eff, n_elem_max, n_iter, incx, ldx, \
	                           ij0, n_shift ); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x1     = x_cast + (j  )*ldx + (0  )*incx; \
\
			PASTEMAC(ch,kername)( conjalpha, \
			                      n_elem, \
			                      alpha_cast, \
			                      x1, incx ); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				x1     = x_cast + (ij0+j  )*ldx + (0  )*incx; \
\
				PASTEMAC(ch,kername)( conjalpha, \
				                      n_elem, \
				                      alpha_cast, \
				                      x1, incx ); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				x1     = x_cast + (j  )*ldx + (ij0+i  )*incx; \
\
				PASTEMAC(ch,kername)( conjalpha, \
				                      n_elem, \
				                      alpha_cast, \
				                      x1, incx ); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( scalm_unb_var1 )

