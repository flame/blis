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

#define FUNCPTR_T subd_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffx,
                           diag_t  diagx,
                           trans_t transx,
                           dim_t   m,
                           dim_t   n,
                           void*   x, inc_t rs_x, inc_t cs_x,
                           void*   y, inc_t rs_y, inc_t cs_y
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,subd_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,subd_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,subd_unb_var1);
#endif
#endif


void bli_subd_unb_var1( obj_t*  x,
                         obj_t*  y )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	diag_t    diagx     = bli_obj_diag( *x );
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
	conj_t   conjx; \
	dim_t    n_elem; \
	dim_t    offx, offy; \
	inc_t    incx, incy; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, transx, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_2d( diagoffx, transx, \
	                      m, n, rs_x, cs_x, rs_y, cs_y, \
	                      offx, offy, n_elem, incx, incy ); \
\
	conjx = bli_extract_conj( transx ); \
\
	if ( bli_is_nonunit_diag( diagx ) ) \
	{ \
		x1   = x_cast + offx; \
		y1   = y_cast + offy; \
	} \
	else /* if ( bli_is_unit_diag( diagx ) ) */ \
	{ \
		/* Simulate a unit diagonal for x with a zero increment over a unit
		   scalar. */ \
		x1   = PASTEMAC(chx,1); \
		incx = 0; \
		y1   = y_cast + offy; \
	} \
\
	PASTEMAC2(chx,chy,kername)( conjx, \
	                            n_elem, \
	                            x1, incx, \
	                            y1, incy ); \
}


// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( subd_unb_var1, SUBV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( subd_unb_var1, SUBV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( subd_unb_var1, SUBV_KERNEL )
#endif

