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

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       FILE*  file, \
       obj_t* x  \
     ) \
{ \
	bli_utilv_fscan_check( file, x ); \
}

GENFRONT( fscanv )

// ---

void bli_utilv_fscan_check
     (
       FILE*  file,
       obj_t* x
     )
{
	err_t e_val;

	// Check argument pointers.

	e_val = bli_check_null_pointer( file );
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

GENFRONT( fscanv )

// -----------------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
       FILE*   file, \
       obj_t*  x  \
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
		PASTEMAC(opname,_check)( file, x ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  file, \
	  m, \
	  buf_x, incx  \
	); \
}

GENFRONT( fscanv )

#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
void PASTEMAC0(opname) \
     ( \
       obj_t*  x  \
     ) \
{ \
	bli_init_once(); \
\
	/* Invoke the typed function. */ \
	PASTEMAC0(varname) \
	( \
	  stdin, \
	  x  \
	); \
}

GENFRONT( scanv, fscanv )

// -----------------------------------------------------------------------------

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t  m, \
       void*  x, inc_t incx  \
     ) \
{ \
	bli_init_once(); \
\
	PASTEMAC(ch,varname) \
	( \
	  stdout, \
	  m, \
	  x, incx  \
	); \
}

INSERT_GENTFUNC_BASIC( scanv, fscanv )

// -----------------------------------------------------------------------------

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       FILE*  file, \
       dim_t  m, \
       ctype* x, inc_t incx  \
     ) \
{ \
	err_t r_val; \
\
	const num_t  dt   = PASTEMAC(ch,type); \
	const ctype* zero = PASTEMAC(ch,0); \
\
	const dim_t chars_per_elem = 100; \
	const dim_t line_buf_len   = chars_per_elem * m; \
\
	/* Allocate a buffer into which a line of chars will be read. */ \
	char* line_buf = bli_malloc_intl( line_buf_len, &r_val ); \
\
	/* Pre-initialize the vector to zeros. */ \
	PASTEMAC(ch,setv) \
	( \
	  BLIS_NO_CONJUGATE, \
	  m, \
	  zero, \
	  x, incx  \
	); \
\
	for ( dim_t i = 0; i < m; ) \
	{ \
		char* r = fgets( line_buf, line_buf_len, file ); \
		if ( r == NULL ) bli_abort(); \
\
		char* start_p = line_buf; \
		char* end_p   = NULL; \
\
		for ( ; i < m; ++i ) \
		{ \
			ctype* xi = x + i*incx; \
\
			/* if ( bli_is_real( dt ) || bli_is_complex( dt ) ) */ \
			{ \
				/* Read the string for the next floating-point number. */ \
				ctype_r chi_real = ( ctype_r )strtod( start_p, &end_p ); \
\
				if ( start_p != end_p ) \
				{ \
					/* Store the converted value to the real part of the jth
					   element of the current row. */ \
					PASTEMAC(ch,setrs)( chi_real, *xi ); \
\
					start_p = end_p; start_p++; \
				} \
				else { break; } \
\
			} \
\
			/* If the matrix is complex, we need to read one more value. */ \
			if ( bli_is_complex( dt ) ) \
			{ \
				/* Read the string for the next floating-point number. */ \
				ctype_r chi_imag = ( ctype_r )strtod( start_p, &end_p ); \
\
				if ( start_p != end_p ) \
				{ \
					/* Store the converted value to the imag part of the jth
					   element of the current row. */ \
					PASTEMAC(ch,setis)( chi_imag, *xi ); \
\
					start_p = end_p; start_p++; \
				} \
				else { ++i; break; } \
			} \
		} \
	} \
\
	/* Free the char buffer. */ \
	bli_free_intl( line_buf ); \
}

INSERT_GENTFUNCR_BASIC0( fscanv )

