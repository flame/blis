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

static ger_vft GENARRAY(ftypes,ger_unb_var1);

void bli_ger_unb_var1( obj_t*  alpha,
                       obj_t*  x,
                       obj_t*  y,
                       obj_t*  a,
                       cntx_t* cntx,
                       ger_t*  cntl )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );
	num_t     dt_a      = bli_obj_datatype( *a );

	conj_t    conjx     = bli_obj_conj_status( *x );
	conj_t    conjy     = bli_obj_conj_status( *y );

	dim_t     m         = bli_obj_length( *a );
	dim_t     n         = bli_obj_width( *a );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     incx      = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     incy      = bli_obj_vector_inc( *y );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;

	// The datatype of alpha MUST be the type union of x and y. This is to
	// prevent any unnecessary loss of information during computation.
	dt_alpha  = bli_datatype_union( dt_x, dt_y );
	buf_alpha = bli_obj_buffer_for_1x1( dt_alpha, *alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_a];

	// Invoke the function.
	f( conjx,
	   conjy,
	   m,
	   n,
	   buf_alpha,
	   buf_x, incx,
	   buf_y, incy,
	   buf_a, rs_a, cs_a,
       cntx );
}


#undef  GENTFUNC3U12
#define GENTFUNC( ctype, ch, varname, kername, kerid ) \
\
void PASTEMAC(cha,varname) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       void*   alpha, \
       void*   x, inc_t incx, \
       void*   y, inc_t incy, \
       void*   a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype_xy* alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_a*  a_cast     = a; \
	ctype_a*  a1t; \
	ctype_x*  chi1; \
	ctype_y*  y1; \
	ctype_xy  alpha_chi1; \
	dim_t     i; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( PASTEMAC(chxy,eq0)( *alpha_cast ) ) return; \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		a1t  = a_cast + (i  )*rs_a + (0  )*cs_a; \
		chi1 = x_cast + (i  )*incx; \
		y1   = y_cast + (0  )*incy; \
\
		/* a1t = a1t + alpha * chi1 * y; */ \
		PASTEMAC2(chx,chxy,copycjs)( conjx, *chi1, alpha_chi1 ); \
		PASTEMAC2(chxy,chxy,scals)( *alpha_cast, alpha_chi1 ); \
\
		PASTEMAC(cha,kername) \
		( \
		  conjy, \
		  n, \
		  &alpha_chi1, \
		  y1,  incy, \
		  a1t, cs_a, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC3U12_BASIC( ger_unb_var1, AXPYV_KERNEL )

