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

#define FUNCPTR_T gemv_fp

typedef void (*FUNCPTR_T)(
                           trans_t transa,
                           conj_t  conjx,
                           dim_t   m,
                           dim_t   n,
                           void*   alpha,
                           void*   a, inc_t rs_a, inc_t cs_a,
                           void*   x, inc_t incx,
                           void*   beta,
                           void*   y, inc_t incy
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,gemv_unf_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,gemv_unf_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,gemv_unf_var1);
#endif
#endif


void bli_gemv_unf_var1( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  x,
                        obj_t*  beta,
                        obj_t*  y,
                        gemv_t* cntl )
{
	num_t     dt_a      = bli_obj_datatype( *a );
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	conj_t    transa    = bli_obj_conjtrans_status( *a );
	conj_t    conjx     = bli_obj_conj_status( *x );

	dim_t     m         = bli_obj_length( *a );
	dim_t     n         = bli_obj_width( *a );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     incx      = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     incy      = bli_obj_vector_inc( *y );

	num_t     dt_alpha;
	void*     buf_alpha;

	num_t     dt_beta;
	void*     buf_beta;

	FUNCPTR_T f;

	// The datatype of alpha MUST be the type union of a and x. This is to
	// prevent any unnecessary loss of information during computation.
	dt_alpha  = bli_datatype_union( dt_a, dt_x );
	buf_alpha = bli_obj_buffer_for_1x1( dt_alpha, *alpha );

	// The datatype of beta MUST be the same as the datatype of y.
	dt_beta   = dt_y;
	buf_beta  = bli_obj_buffer_for_1x1( dt_beta, *beta );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_a][dt_x][dt_y];

	// Invoke the function.
	f( transa,
	   conjx,
	   m,
	   n,
	   buf_alpha,
	   buf_a, rs_a, cs_a,
	   buf_x, incx,
	   buf_beta,
	   buf_y, incy );
}


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, varname, kername ) \
\
void PASTEMAC3(cha,chx,chy,varname)( \
                                     trans_t transa, \
                                     conj_t  conjx, \
                                     dim_t   m, \
                                     dim_t   n, \
                                     void*   alpha, \
                                     void*   a, inc_t rs_a, inc_t cs_a, \
                                     void*   x, inc_t incx, \
                                     void*   beta, \
                                     void*   y, inc_t incy  \
                         ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype_ax* alpha_cast = alpha; \
	ctype_y*  beta_cast  = beta; \
	ctype_a*  a_cast     = a; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_a*  A1; \
	ctype_x*  x1; \
	ctype_y*  y1; \
	dim_t     i; \
	dim_t     b_fuse, f; \
	dim_t     n_elem, n_iter; \
	inc_t     rs_at, cs_at; \
	conj_t    conja; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	bli_set_dims_incs_with_trans( transa, \
	                              m, n, rs_a, cs_a, \
	                              n_iter, n_elem, rs_at, cs_at ); \
\
	conja = bli_extract_conj( transa ); \
\
	/* Query the fusing factor for the dotxf implementation. */ \
	b_fuse = PASTEMAC(chax,dotxf_fusefac); \
\
	for ( i = 0; i < n_iter; i += f ) \
	{ \
		f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
		A1 = a_cast + (i  )*rs_at + (0  )*cs_at; \
		x1 = x_cast + (0  )*incy; \
		y1 = y_cast + (i  )*incy; \
\
		/* y1 = beta * y1 + alpha * A1 * x; */ \
		PASTEMAC3(cha,chx,chy,kername)( conja, \
		                                conjx, \
		                                n_elem, \
		                                f, \
		                                alpha_cast, \
		                                A1,   cs_at, rs_at, \
		                                x1,   incx, \
		                                beta_cast, \
		                                y1,   incy ); \
\
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( gemv_unf_var1, DOTXF_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( gemv_unf_var1, DOTXF_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( gemv_unf_var1, DOTXF_KERNEL )
#endif

