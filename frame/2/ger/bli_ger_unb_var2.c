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

#define FUNCPTR_T ger_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conjx,
                           conj_t  conjy,
                           dim_t   m,
                           dim_t   n,
                           void*   alpha,
                           void*   x, inc_t incx,
                           void*   y, inc_t incy,
                           void*   a, inc_t rs_a, inc_t cs_a
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,ger_unb_var2);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,ger_unb_var2);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,ger_unb_var2);
#endif
#endif


void bli_ger_unb_var2( obj_t*  alpha,
                       obj_t*  x,
                       obj_t*  y,
                       obj_t*  a,
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
	f = ftypes[dt_x][dt_y][dt_a];

	// Invoke the function.
	f( conjx,
	   conjy,
	   m,
	   n,
	   buf_alpha,
	   buf_x, incx,
	   buf_y, incy,
	   buf_a, rs_a, cs_a );
}


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_a, ctype_xy, chx, chy, cha, chxy, varname, kername ) \
\
void PASTEMAC3(chx,chy,cha,varname)( \
                                     conj_t  conjx, \
                                     conj_t  conjy, \
                                     dim_t   m, \
                                     dim_t   n, \
                                     void*   alpha, \
                                     void*   x, inc_t incx, \
                                     void*   y, inc_t incy, \
                                     void*   a, inc_t rs_a, inc_t cs_a \
                                   ) \
{ \
	ctype_xy* alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_a*  a_cast     = a; \
	ctype_a*  a1; \
	ctype_x*  x1; \
	ctype_y*  psi1; \
	ctype_xy  alpha_psi1; \
	dim_t     j; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( PASTEMAC(chxy,eq0)( *alpha_cast ) ) return; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		a1   = a_cast + (0  )*rs_a + (j  )*cs_a; \
		x1   = x_cast + (0  )*incx; \
		psi1 = y_cast + (j  )*incy; \
\
		/* a1 = a1 + alpha * psi1 * x; */ \
		PASTEMAC2(chy,chxy,copycjs)( conjy, *psi1, alpha_psi1 ); \
		PASTEMAC2(chxy,chxy,scals)( *alpha_cast, alpha_psi1 ); \
\
		PASTEMAC3(chxy,chx,cha,kername)( conjx, \
		                                 m, \
		                                 &alpha_psi1, \
		                                 x1, incx, \
		                                 a1, rs_a ); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( ger_unb_var2, AXPYV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( ger_unb_var2, AXPYV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( ger_unb_var2, AXPYV_KERNEL )
#endif

