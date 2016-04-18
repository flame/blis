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

#define FUNCPTR_T sumsqv_fp

typedef void (*FUNCPTR_T)(
                           dim_t  n,
                           void*  x, inc_t incx,
                           void*  scale,
                           void*  sumsq
                         );

/*
// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,sumsqv_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,sumsqv_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,sumsqv_unb_var1);
#endif
#endif
*/
static FUNCPTR_T GENARRAY(ftypes,sumsqv_unb_var1);


void bli_sumsqv_unb_var1( obj_t*  x,
                          obj_t*  scale,
                          obj_t*  sumsq )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	//num_t     dt_s      = bli_obj_datatype( *scale );

	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	void*     buf_scale = bli_obj_buffer_at_off( *scale );

	void*     buf_sumsq = bli_obj_buffer_at_off( *sumsq );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x]; //[dt_s];

	// Invoke the function.
	f( n,
	   buf_x, inc_x,
	   buf_scale,
	   buf_sumsq );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t  n, \
                           void*  x, inc_t incx, \
                           void*  scale, \
                           void*  sumsq  \
                         ) \
{ \
	ctype*        x_cast     = x; \
	ctype_r*      scale_cast = scale; \
	ctype_r*      sumsq_cast = sumsq; \
\
	const ctype_r zero_r     = *PASTEMAC(chr,0); \
	const ctype_r one_r      = *PASTEMAC(chr,1); \
\
	ctype*        chi1; \
	ctype_r       chi1_r; \
	ctype_r       chi1_i; \
	ctype_r       scale_r; \
	ctype_r       sumsq_r; \
	ctype_r       abs_chi1_r; \
	dim_t         i; \
\
	/* NOTE: This function attempts to mimic the algorithm for computing
	   the Frobenius norm in netlib LAPACK's ?lassq(). */ \
\
	/* If x is zero length, return with scale and sumsq unchanged. */ \
	if ( bli_zero_dim1( n ) ) return; \
\
	/* Copy scale and sumsq to local variables. */ \
	PASTEMAC(chr,copys)( *scale_cast, scale_r ); \
	PASTEMAC(chr,copys)( *sumsq_cast, sumsq_r ); \
\
	chi1 = x_cast; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		/* Get the real and imaginary components of chi1. */ \
		PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
		abs_chi1_r = bli_fabs( chi1_r ); \
\
		/* Accumulate real component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_r > zero_r || bli_isnan( abs_chi1_r) ) \
		{ \
			if ( scale_r < abs_chi1_r ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_r ) * \
				                    ( scale_r / abs_chi1_r );  \
\
				PASTEMAC(chr,copys)( abs_chi1_r, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_r / scale_r ) * \
				                    ( abs_chi1_r / scale_r );  \
			} \
		} \
\
		abs_chi1_r = bli_fabs( chi1_i ); \
\
		/* Accumulate imaginary component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_r > zero_r || bli_isnan( abs_chi1_r) ) \
		{ \
			if ( scale_r < abs_chi1_r ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_r ) * \
				                    ( scale_r / abs_chi1_r );  \
\
				PASTEMAC(chr,copys)( abs_chi1_r, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_r / scale_r ) * \
				                    ( abs_chi1_r / scale_r );  \
			} \
		} \
\
		chi1 += incx; \
	} \
\
	/* Store final values of scale and sumsq to output variables. */ \
	PASTEMAC(chr,copys)( scale_r, *scale_cast ); \
	PASTEMAC(chr,copys)( sumsq_r, *sumsq_cast ); \
}

INSERT_GENTFUNCR_BASIC0( sumsqv_unb_var1 )

