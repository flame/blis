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

#define FUNCPTR_T her_fp

typedef void (*FUNCPTR_T)(
                           uplo_t  uplo,
                           conj_t  conjx,
                           conj_t  conjh,
                           dim_t   m,
                           void*   alpha,
                           void*   x, inc_t incx,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,her_unb_var2);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,her_unb_var2);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,her_unb_var2);
#endif
#endif


void bli_her_unb_var2( conj_t  conjh,
                       obj_t*  alpha,
                       obj_t*  x,
                       obj_t*  c,
                       her_t*  cntl )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_c      = bli_obj_datatype( *c );

	uplo_t    uplo      = bli_obj_uplo( *c );
	conj_t    conjx     = bli_obj_conj_status( *x );

	dim_t     m         = bli_obj_length( *c );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     incx      = bli_obj_vector_inc( *x );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;


	// If alpha is a scalar constant, use dt_x to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the alpha object and extract the buffer at the alpha offset.
	bli_set_scalar_dt_buffer( alpha, dt_x, dt_alpha, buf_alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x][dt_c];

	// Invoke the function.
	f( uplo,
	   conjx,
	   conjh,
	   m,
	   buf_alpha,
	   buf_x, incx,
	   buf_c, rs_c, cs_c );
}


#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_c, chx, chc, varname, kername ) \
\
void PASTEMAC2(chx,chc,varname)( \
                                 uplo_t  uplo, \
                                 conj_t  conjx, \
                                 conj_t  conjh, \
                                 dim_t   m, \
                                 void*   alpha, \
                                 void*   x, inc_t incx, \
                                 void*   c, inc_t rs_c, inc_t cs_c \
                               ) \
{ \
	ctype_x*  alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_c*  c_cast     = c; \
	ctype_x*  chi1; \
	ctype_x*  x2; \
	ctype_c*  gamma11; \
	ctype_c*  c21; \
	ctype_x   alpha_local; \
	ctype_x   alpha_chi1; \
	ctype_x   alpha_chi1_chi1; \
	ctype_x   conjx0_chi1; \
	ctype_x   conjx1_chi1; \
	dim_t     i; \
	dim_t     n_ahead; \
	inc_t     rs_ct, cs_ct; \
	conj_t    conj0, conj1; \
\
	/* Eliminate unused variable warnings. */ \
	( void )conj0; \
\
	if ( bli_zero_dim1( m ) ) return; \
\
	if ( PASTEMAC(chx,eq0)( *alpha_cast ) ) return; \
\
	/* Make a local copy of alpha and zero out the imaginary component if
	   we are being invoked as her, since her requires alpha to be real. */ \
	PASTEMAC2(chx,chx,copys)( *alpha_cast, alpha_local ); \
	if ( bli_is_conj( conjh ) ) \
	{ \
		PASTEMAC(chx,seti0s)( alpha_local ); \
	} \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_ct = rs_c; \
		cs_ct = cs_c; \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_ct = cs_c; \
		cs_ct = rs_c; \
\
		/* Toggle conjugation of conjx, but only if we are being invoked
		   as her; for syr, conjx is unchanged. */ \
		conjx = bli_apply_conj( conjh, conjx ); \
	} \
\
	/* Apply conjh (which carries the conjugation component of the Hermitian
	   transpose, if applicable) to conjx as needed to arrive at the effective
	   conjugation for the scalar and vector subproblems. */ \
	conj0 = bli_apply_conj( conjh, conjx ); \
	conj1 = conjx; \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_ahead  = m - i - 1; \
		chi1     = x_cast + (i  )*incx; \
		x2       = x_cast + (i+1)*incx; \
		gamma11  = c_cast + (i  )*rs_ct + (i  )*cs_ct; \
		c21      = c_cast + (i+1)*rs_ct + (i  )*cs_ct; \
\
		/* Apply conjx to chi1. */ \
		PASTEMAC2(chx,chx,copycjs)( conj0, *chi1, conjx0_chi1 ); \
		PASTEMAC2(chx,chx,copycjs)( conj1, *chi1, conjx1_chi1 ); \
\
		/* Compute scalar for vector subproblem. */ \
		PASTEMAC3(chx,chx,chx,scal2s)( alpha_local, conjx0_chi1, alpha_chi1 ); \
\
		/* Compute alpha * chi1 * conj(chi1) after chi1 has already been
		   conjugated, if needed, by conjx. */ \
		PASTEMAC3(chx,chx,chx,scal2s)( alpha_chi1, conjx1_chi1, alpha_chi1_chi1 ); \
\
		/* c21 = c21 + alpha * x2 * conj(chi1); */ \
		PASTEMAC3(chx,chx,chc,kername)( conj1, \
		                                n_ahead, \
		                                &alpha_chi1, \
		                                x2,  incx, \
		                                c21, rs_ct ); \
\
		/* gamma11 = gamma11 + alpha * chi1 * conj(chi1); */ \
		PASTEMAC2(chx,chc,adds)( alpha_chi1_chi1, *gamma11 ); \
\
		/* For her, explicitly set the imaginary component of gamma11 to
		   zero. */ \
		if ( bli_is_conj( conjh ) ) \
			PASTEMAC(chc,seti0s)( *gamma11 ); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( her_unb_var2, AXPYV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( her_unb_var2, AXPYV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( her_unb_var2, AXPYV_KERNEL )
#endif

