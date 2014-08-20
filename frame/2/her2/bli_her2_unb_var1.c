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

#define FUNCPTR_T her2_fp

typedef void (*FUNCPTR_T)(
                           uplo_t  uplo,
                           conj_t  conjx,
                           conj_t  conjy,
                           conj_t  conjh,
                           dim_t   m,
                           void*   alpha,
                           void*   x, inc_t incx,
                           void*   y, inc_t incy,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,her2_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,her2_unb_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,her2_unb_var1);
#endif
#endif


void bli_her2_unb_var1( conj_t   conjh,
                        obj_t*   alpha,
                        obj_t*   alpha_conj,
                        obj_t*   x,
                        obj_t*   y,
                        obj_t*   c,
                        her2_t*  cntl )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );
	num_t     dt_c      = bli_obj_datatype( *c );

	uplo_t    uplo      = bli_obj_uplo( *c );
	conj_t    conjx     = bli_obj_conj_status( *x );
	conj_t    conjy     = bli_obj_conj_status( *y );

	dim_t     m         = bli_obj_length( *c );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     incx      = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     incy      = bli_obj_vector_inc( *y );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;

	// The datatype of alpha MUST be the type union of the datatypes of x and y.
	dt_alpha  = bli_datatype_union( dt_x, dt_y );
	buf_alpha = bli_obj_buffer_for_1x1( dt_alpha, *alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x][dt_y][dt_c];

	// Invoke the function.
	f( uplo,
	   conjx,
	   conjy,
	   conjh,
	   m,
	   buf_alpha,
	   buf_x, incx,
	   buf_y, incy,
	   buf_c, rs_c, cs_c );
}


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_c, ctype_xy, chx, chy, chc, chxy, varname, kername ) \
\
void PASTEMAC3(chx,chy,chc,varname)( \
                                     uplo_t  uplo, \
                                     conj_t  conjx, \
                                     conj_t  conjy, \
                                     conj_t  conjh, \
                                     dim_t   m, \
                                     void*   alpha, \
                                     void*   x, inc_t incx, \
                                     void*   y, inc_t incy, \
                                     void*   c, inc_t rs_c, inc_t cs_c \
                                   ) \
{ \
	ctype_xy* two        = PASTEMAC(chxy,2); \
	ctype_xy* alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_c*  c_cast     = c; \
	ctype_x*  x0; \
	ctype_x*  chi1; \
	ctype_y*  y0; \
	ctype_y*  psi1; \
	ctype_c*  c10t; \
	ctype_c*  gamma11; \
	ctype_xy  alpha0; \
	ctype_xy  alpha1; \
	ctype_xy  alpha0_chi1; \
	ctype_xy  alpha1_psi1; \
	ctype_xy  alpha0_chi1_psi1; \
	ctype_x   conjx0_chi1; \
	ctype_y   conjy1_psi1; \
	ctype_y   conjy0_psi1; \
	dim_t     i; \
	dim_t     n_behind; \
	inc_t     rs_ct, cs_ct; \
	conj_t    conj0, conj1; \
\
	if ( bli_zero_dim1( m ) ) return; \
\
	if ( PASTEMAC(chxy,eq0)( *alpha_cast ) ) return; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_ct = rs_c; \
		cs_ct = cs_c; \
\
		PASTEMAC2(chxy,chxy,copys)( *alpha_cast, alpha0 ); \
		PASTEMAC2(chxy,chxy,copycjs)( conjh, *alpha_cast, alpha1 ); \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_ct = cs_c; \
		cs_ct = rs_c; \
\
		/* Toggle conjugation of conjx/conjy, but only if we are being invoked
		   as her2; for syr2, conjx/conjy are unchanged. */ \
		conjx = bli_apply_conj( conjh, conjx ); \
		conjy = bli_apply_conj( conjh, conjy ); \
\
		PASTEMAC2(chxy,chxy,copycjs)( conjh, *alpha_cast, alpha0 ); \
		PASTEMAC2(chxy,chxy,copys)( *alpha_cast, alpha1 ); \
	} \
\
	/* Apply conjh (which carries the conjugation component of the Hermitian
	   transpose, if applicable) to conjx and/or conjy as needed to arrive at
	   the effective conjugation for the vector subproblems. */ \
	conj0 = bli_apply_conj( conjh, conjy ); \
	conj1 = bli_apply_conj( conjh, conjx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_behind = i; \
		x0       = x_cast + (0  )*incx; \
		chi1     = x_cast + (i  )*incx; \
		y0       = y_cast + (0  )*incy; \
		psi1     = y_cast + (i  )*incy; \
		c10t     = c_cast + (i  )*rs_ct + (0  )*cs_ct; \
		gamma11  = c_cast + (i  )*rs_ct + (i  )*cs_ct; \
\
		/* Apply conjx and/or conjy to chi1 and/or psi1. */ \
		PASTEMAC2(chx,chx,copycjs)( conjx,        *chi1, conjx0_chi1 ); \
		PASTEMAC2(chy,chy,copycjs)( conjy,        *psi1, conjy1_psi1 ); \
		PASTEMAC2(chy,chy,copycjs)( conj0,        *psi1, conjy0_psi1 ); \
\
		/* Compute scalars for vector subproblems. */ \
		PASTEMAC3(chxy,chx,chxy,scal2s)( alpha0, conjx0_chi1, alpha0_chi1 ); \
		PASTEMAC3(chxy,chx,chxy,scal2s)( alpha1, conjy1_psi1, alpha1_psi1 ); \
\
		/* Compute alpha * chi1 * conj(psi1) after both chi1 and psi1 have
		   already been conjugated, if needed, by conjx and conjy. */ \
		PASTEMAC3(chy,chxy,chxy,scal2s)( alpha0_chi1, conjy0_psi1, alpha0_chi1_psi1 ); \
\
		/* c10t = c10t + alpha * chi1 * y0'; */ \
		PASTEMAC3(chxy,chy,chc,kername)( conj0, \
		                                 n_behind, \
		                                 &alpha0_chi1, \
		                                 y0,   incy, \
		                                 c10t, cs_ct ); \
\
		/* c10t = c10t + conj(alpha) * psi1 * x0'; */ \
		PASTEMAC3(chxy,chx,chc,kername)( conj1, \
		                                 n_behind, \
		                                 &alpha1_psi1, \
		                                 x0,   incx, \
		                                 c10t, cs_ct ); \
\
		/* gamma11 = gamma11 +      alpha  * chi1 * conj(psi1) \
		                     + conj(alpha) * psi1 * conj(chi1); */ \
		PASTEMAC3(chxy,chxy,chc,axpys)( *two, alpha0_chi1_psi1, *gamma11 ); \
\
		/* For her2, explicitly set the imaginary component of gamma11 to
           zero. */ \
		if ( bli_is_conj( conjh ) ) \
			PASTEMAC(chc,seti0s)( *gamma11 ); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( her2_unb_var1, AXPYV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( her2_unb_var1, AXPYV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( her2_unb_var1, AXPYV_KERNEL )
#endif

