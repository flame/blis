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

#define FUNCPTR_T normfv_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           void*   x, inc_t incx,
                           void*   norm
                         );

static FUNCPTR_T GENARRAY(ftypes,normfv_unb_var1);


void bli_normfv_unb_var1( obj_t*  x,
                          obj_t*  norm )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	dim_t     m         = bli_obj_vector_dim( *x );

	inc_t     incx      = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	void*     buf_norm  = bli_obj_buffer_at_off( *norm );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( m,
	   buf_x, incx,
	   buf_norm );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t   m, \
                           void*   x, inc_t incx, \
                           void*   norm \
                         ) \
{ \
	ctype*   x_cast     = x; \
	ctype_r* norm_cast  = norm; \
	ctype_r* zero       = PASTEMAC(chr,0); \
	ctype_r* one        = PASTEMAC(chr,1); \
	ctype_r  scale; \
	ctype_r  sumsq; \
	ctype_r  sqrt_sumsq; \
\
	/* NOTE: Early returns due to empty dimensions are handled by the
	   caller. */ \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	PASTEMAC(chr,copys)( *zero, scale ); \
	PASTEMAC(chr,copys)( *one,  sumsq ); \
\
	/* Compute the sum of the squares of the vector. */ \
	PASTEMAC(ch,kername) \
	( \
	  m, \
	  x_cast, incx, \
	  &scale, \
	  &sumsq  \
	); \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	PASTEMAC(chr,sqrt2s)( sumsq, sqrt_sumsq ); \
	PASTEMAC(chr,scals)( scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	PASTEMAC(chr,copys)( sqrt_sumsq, *norm_cast ); \
}

INSERT_GENTFUNCR_BASIC( normfv_unb_var1, sumsqv_unb_var1 )

