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

#define FUNCPTR_T normfm_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffx,
                           diag_t  diagx,
                           uplo_t  uplox,
                           dim_t   m,
                           dim_t   n,
                           void*   x, inc_t rs_x, inc_t cs_x,
                           void*   norm
                         );

static FUNCPTR_T GENARRAY(ftypes,normfm_unb_var1);


void bli_normfm_unb_var1( obj_t*  x,
                          obj_t*  norm )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	diag_t    diagx     = bli_obj_diag( *x );
	uplo_t    uplox     = bli_obj_uplo( *x );

	dim_t     m         = bli_obj_length( *x );
	dim_t     n         = bli_obj_width( *x );

	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	void*     buf_norm  = bli_obj_buffer_at_off( *norm );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( diagoffx,
	   diagx,
	   uplox,
	   m,
	   n,
	   buf_x, rs_x, cs_x,
	   buf_norm );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, varname, kername ) \
\
void PASTEMAC(chx,varname)( \
                            doff_t  diagoffx, \
                            diag_t  diagx, \
                            uplo_t  uplox, \
                            dim_t   m, \
                            dim_t   n, \
                            void*   x, inc_t rs_x, inc_t cs_x, \
                            void*   norm \
                          ) \
{ \
	ctype_x*  x_cast     = x; \
	ctype_xr* norm_cast  = norm; \
	ctype_x*  one        = PASTEMAC(chx,1); \
	ctype_xr* one_r      = PASTEMAC(chxr,1); \
	ctype_xr* zero_r     = PASTEMAC(chxr,0); \
	ctype_x*  x0; \
	ctype_x*  chi1; \
	ctype_x*  x2; \
	ctype_xr  scale; \
	ctype_xr  sumsq; \
	ctype_xr  sqrt_sumsq; \
	uplo_t    uplox_eff; \
	dim_t     n_iter; \
	dim_t     n_elem, n_elem_max; \
	inc_t     ldx, incx; \
	dim_t     j, i; \
	dim_t     ij0, n_shift; \
\
	/* Return a norm of zero if either dimension is zero. */ \
	if ( bli_zero_dim2( m, n ) ) \
	{ \
		PASTEMAC(chxr,set0s)( *norm_cast ); \
		return; \
	} \
\
	/* Set various loop parameters. Here, we pretend that diagx is equal to
	   BLIS_NONUNIT_DIAG because we handle the unit diagonal case manually. */ \
	bli_set_dims_incs_uplo_1m( diagoffx, BLIS_NONUNIT_DIAG, \
	                           uplox, m, n, rs_x, cs_x, \
	                           uplox_eff, n_elem_max, n_iter, incx, ldx, \
	                           ij0, n_shift ); \
\
	/* Check the effective uplo; if it's zeros, then our norm is zero. */ \
	if ( bli_is_zeros( uplox_eff ) ) \
	{ \
		PASTEMAC(chxr,set0s)( *norm_cast ); \
		return; \
	} \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	PASTEMAC2(chxr,chxr,copys)( *zero_r, scale ); \
	PASTEMAC2(chxr,chxr,copys)( *one_r,  sumsq ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x0     = x_cast + (j  )*ldx + (0  )*incx; \
\
			/* Compute the norm of the current column. */ \
			PASTEMAC(chx,kername)( n_elem, \
			                       x0, incx, \
			                       &scale, \
			                       &sumsq ); \
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
				x0     = x_cast + (ij0+j  )*ldx + (0       )*incx; \
				chi1   = x_cast + (ij0+j  )*ldx + (n_elem-1)*incx; \
\
				/* Sum the squares of the super-diagonal elements. */ \
				PASTEMAC(chx,kername)( n_elem - 1, \
				                       x0, incx, \
				                       &scale, \
				                       &sumsq ); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				PASTEMAC(chx,kername)( 1, \
				                       chi1, incx, \
				                       &scale, \
				                       &sumsq ); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				chi1   = x_cast + (j  )*ldx + (ij0+i  )*incx; \
				x2     = x_cast + (j  )*ldx + (ij0+i+1)*incx; \
\
				/* Sum the squares of the sub-diagonal elements. */ \
				PASTEMAC(chx,kername)( n_elem - 1, \
				                       x2, incx, \
				                       &scale, \
				                       &sumsq ); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				PASTEMAC(chx,kername)( 1, \
				                       chi1, incx, \
				                       &scale, \
				                       &sumsq ); \
			} \
		} \
	} \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	PASTEMAC2(chxr,chxr,sqrt2s)( sumsq, sqrt_sumsq ); \
	PASTEMAC2(chxr,chxr,scals)( scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	PASTEMAC2(chxr,chxr,copys)( sqrt_sumsq, *norm_cast ); \
}

INSERT_GENTFUNCR_BASIC( normfm_unb_var1, sumsqv_unb_var1 )

