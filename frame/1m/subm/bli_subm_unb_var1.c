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

#define FUNCPTR_T subm_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffx,
                           diag_t  diagx,
                           uplo_t  uplox,
                           trans_t transx,
                           dim_t   m,
                           dim_t   n,
                           void*   x, inc_t rs_x, inc_t cs_x,
                           void*   y, inc_t rs_y, inc_t cs_y
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,subm_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,subm_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,subm_unb_var1);
#endif
#endif


void bli_subm_unb_var1( obj_t*  x,
                         obj_t*  y )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	diag_t    diagx     = bli_obj_diag( *x );
	uplo_t    uplox     = bli_obj_uplo( *x );
	trans_t   transx    = bli_obj_conjtrans_status( *x );

	dim_t     m         = bli_obj_length( *y );
	dim_t     n         = bli_obj_width( *y );

	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	inc_t     rs_y      = bli_obj_row_stride( *y );
	inc_t     cs_y      = bli_obj_col_stride( *y );
	void*     buf_y     = bli_obj_buffer_at_off( *y );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x][dt_y];

	// Invoke the function.
	f( diagoffx,
	   diagx,
	   uplox,
	   transx,
	   m,
	   n,
	   buf_x, rs_x, cs_x,
	   buf_y, rs_y, cs_y );
}


#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, varname, kername ) \
\
void PASTEMAC2(chx,chy,varname)( \
                                 doff_t  diagoffx, \
                                 diag_t  diagx, \
                                 uplo_t  uplox, \
                                 trans_t transx, \
                                 dim_t   m, \
                                 dim_t   n, \
                                 void*   x, inc_t rs_x, inc_t cs_x, \
                                 void*   y, inc_t rs_y, inc_t cs_y \
                               ) \
{ \
	ctype_x* x_cast     = x; \
	ctype_y* y_cast     = y; \
	ctype_x* x1; \
	ctype_y* y1; \
	uplo_t   uplox_eff; \
	conj_t   conjx; \
	dim_t    n_iter; \
	dim_t    n_elem, n_elem_max; \
	inc_t    ldx, incx; \
	inc_t    ldy, incy; \
	dim_t    j, i; \
	dim_t    ij0, n_shift; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* When the diagonal of x is implicitly unit, we first update only the
	   region strictly above or below the diagonal of y, and then update the
	   diagonal of y. */ \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m( diagoffx, diagx, transx, \
	                           uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	                           uplox_eff, n_elem_max, n_iter, incx, ldx, incy, ldy, \
	                           ij0, n_shift ); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	conjx = bli_extract_conj( transx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x1     = x_cast + (j  )*ldx + (0  )*incx; \
			y1     = y_cast + (j  )*ldy + (0  )*incy; \
\
			PASTEMAC2(chx,chy,kername)( conjx, \
			                            n_elem, \
			                            x1, incx, \
			                            y1, incy ); \
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
				y1     = y_cast + (ij0+j  )*ldy + (0  )*incy; \
\
				PASTEMAC2(chx,chy,kername)( conjx, \
				                            n_elem, \
				                            x1, incx, \
				                            y1, incy ); \
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
				y1     = y_cast + (j  )*ldy + (ij0+i  )*incy; \
\
				PASTEMAC2(chx,chy,kername)( conjx, \
				                            n_elem, \
				                            x1, incx, \
				                            y1, incy ); \
			} \
		} \
\
		/* When the diagonal is unit, we handle it separately. */ \
		if ( bli_is_unit_diag( diagx ) ) \
		{ \
			PASTEMAC2(chy,chy,addd)( diagoffx, \
			                         diagx, \
			                         transx, \
			                         m, \
			                         n, \
			                         x_cast, rs_x, cs_x, \
			                         y_cast, rs_y, cs_y ); \
		} \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( subm_unb_var1, SUBV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( subm_unb_var1, SUBV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( subm_unb_var1, SUBV_KERNEL )
#endif

