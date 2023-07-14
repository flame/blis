/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

//
// Define object-based check functions.
//

void bli_inverttv_check
     (
             double thresh,
       const obj_t* x
     )
{
	err_t e_val;

	// NOTE: ADD A CHECK TO ENSURE thresh IS NOT NaN or Inf.

	// NOTE: ADD A CHECK TO ENSURE thresh IS POSITIVE (as in, neither negative
	// nor zero).

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_real_object( x );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );
}

// -----------------------------------------------------------------------------

// Operations with only basic interfaces.

#undef  GENFRONT
#define GENFRONT( opname ) \
\
/*
GENARRAY_FPA( void_fp, opname ); \
*/ \
\
GENARRAY_FPA( PASTECH(opname,_vft), \
              PASTECH0(opname) ); \
\
PASTECH(opname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt ) \
{ \
	return PASTECH(opname,_fpa)[ dt ]; \
}

GENFRONT( inverttv )

// -----------------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
             double  thresh, \
       const obj_t*  x  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( x ); \
\
	dim_t     m         = bli_obj_vector_dim( x ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( thresh, x ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  thresh, \
	  m, \
	  buf_x, incx  \
	); \
}

GENFRONT( inverttv )

// -----------------------------------------------------------------------------

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             double  thresh, \
             dim_t   m, \
             ctype*  x, inc_t incx  \
     ) \
{ \
	bli_init_once(); \
\
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		ctype   absx; \
		ctype_r absxr, absxi; \
\
		/* Compute the absolute value (or complex modulus) of x[i]. */ \
		/* Note: Even when x[i] is complex, the imaginary component of abs(x[i])
		   will be zero, and thus we can ignore it. */ \
		PASTEMAC(ch,abval2s)( *x, absx ); \
		PASTEMAC(ch,gets)( absx, absxr, absxi ); \
\
		if ( PASTEMAC(chr,lt)( absxr, thresh ) ) \
		{ \
			/* If abs(x[i]) falls short of the threshold, we set x[i] to 0
			   instead of inverting it. */ \
			PASTEMAC(ch,set0s)( *x ); \
		} \
		else \
		{ \
			/* If abs(x[i]) meets or exceeds the threshold, then we invert
			   x[i]. */ \
			PASTEMAC(ch,inverts)( *x ); \
		} \
\
		x += incx; \
	} \
}

INSERT_GENTFUNCR_BASIC0( inverttv )

