/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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

#define FUNCPTR_T absumv_fp

typedef void (*FUNCPTR_T)(
                           dim_t  n,
                           void*  x, inc_t incx,
                           void*  absum
                         );

/*
// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,absumv_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,absumv_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,absumv_unb_var1);
#endif
#endif
*/
static FUNCPTR_T GENARRAY(ftypes,absumv_unb_var1);


void bli_absumv_unb_var1( obj_t*  x,
                          obj_t*  absum )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	void*     buf_absum = bli_obj_buffer_at_off( *absum );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x]; //[dt_s];

	// Invoke the function.
	f( n,
	   buf_x, inc_x,
	   buf_absum );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, opname, varname ) \
\
void PASTEMAC(chx,varname)( \
                            dim_t  n, \
                            void*  x, inc_t incx, \
                            void*  absum  \
                          ) \
{ \
	ctype_x*  x_cast     = x; \
	ctype_xr* absum_cast = absum; \
	ctype_x*  chi1; \
	ctype_xr  chi1_r; \
	ctype_xr  chi1_i; \
	ctype_xr  absum_r; \
	dim_t     i; \
\
	/* Initialize the absolute sum accumulator to zero. */ \
	PASTEMAC(chxr,set0s)( absum_r ); \
\
	/* If x is zero length, return swith absum equal to zero. */ \
	if ( bli_zero_dim1( n ) ) \
	{ \
		PASTEMAC2(chxr,chxr,copys)( absum_r, *absum_cast ); \
		return; \
	} \
\
	chi1 = x_cast; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		/* Get the real and imaginary components of chi1. */ \
		PASTEMAC2(chx,chxr,gets)( *chi1, chi1_r, chi1_i ); \
\
		/* Replace chi1_r and chi1_i with their absolute values. */ \
		chi1_r = bli_fabs( chi1_r ); \
		chi1_i = bli_fabs( chi1_i ); \
\
		/* Accumulate the real and imaginary components into absum. */ \
		PASTEMAC2(chxr,chxr,adds)( chi1_r, absum_r ); \
		PASTEMAC2(chxr,chxr,adds)( chi1_i, absum_r ); \
\
		chi1 += incx; \
	} \
\
	/* Store final values of absum to output variables. */ \
	PASTEMAC2(chxr,chxr,copys)( absum_r, *absum_cast ); \
}

INSERT_GENTFUNCR_BASIC( absumv, absumv_unb_var1 )

