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

#ifndef BLIS_MISC_MACRO_DEFS_H
#define BLIS_MISC_MACRO_DEFS_H


// -- Miscellaneous macros --

// min, max, abs
// NOTE: These must remain macros since we don't know the types of a and b.

#define bli_min( a, b )  ( (a) < (b) ? (a) : (b) )
#define bli_max( a, b )  ( (a) > (b) ? (a) : (b) )
#define bli_abs( a )     ( (a) <= 0 ? -(a) : (a) )

// fmin, fmax, fabs
// NOTE: These must remain macros since we don't know the types of a and b.

#define bli_fmin( a, b ) bli_min( a, b )
#define bli_fmax( a, b ) bli_max( a, b )
#define bli_fabs( a )    ( (a) <= 0.0 ? -(a) : (a) )

// fminabs, fmaxabs
// NOTE: These must remain macros since we don't know the types of a and b.

#define bli_fminabs( a, b ) \
\
	bli_fmin( bli_fabs( a ), \
	          bli_fabs( b ) )

#define bli_fmaxabs( a, b ) \
\
	bli_fmax( bli_fabs( a ), \
	          bli_fabs( b ) )

// round

BLIS_INLINE double bli_round( double a )
{
	return round( a );
}

// round_to_mult

BLIS_INLINE guint_t bli_round_to_mult( guint_t val, guint_t mult )
{
	return ( guint_t )
	       ( ( ( ( guint_t )val +
	             ( guint_t )mult / 2
	           ) / mult
	         ) * mult
	       );
}

// isnan, isinf
// NOTE: These must remain macros, since isinf() and isnan() are macros
// (defined in math.h) that likely depend on the type of the argument 'a'
// below.

#define bli_isinf( a )  isinf( a )
#define bli_isnan( a )  isnan( a )

// is_odd, is_even

BLIS_INLINE bool bli_is_odd( gint_t a )
{
	return ( bool )( a % 2 == 1 );
}

BLIS_INLINE bool bli_is_even( gint_t a )
{
	return ( bool )( a % 2 == 0 );
}

// swap_dims

BLIS_INLINE void bli_swap_dims( dim_t* dim1, dim_t* dim2 )
{
	dim_t temp = *dim1;
	*dim1 = *dim2;
	*dim2 = temp;
}

// swap_incs

BLIS_INLINE void bli_swap_incs( inc_t* inc1, inc_t* inc2 )
{
	inc_t temp = *inc1;
	*inc1 = *inc2;
	*inc2 = temp;
}

// toggle_bool

BLIS_INLINE void bli_toggle_bool( bool* b )
{
	if ( *b == TRUE ) *b = FALSE;
	else              *b = TRUE;
}

// return datatype for char

#define bli_stype ( BLIS_FLOAT    )
#define bli_dtype ( BLIS_DOUBLE   )
#define bli_ctype ( BLIS_SCOMPLEX )
#define bli_ztype ( BLIS_DCOMPLEX )

// return C type for char

#define bli_sctype  float
#define bli_dctype  double
#define bli_cctype  scomplex
#define bli_zctype  dcomplex

// return real proj of C type for char

#define bli_sctyper  float
#define bli_dctyper  double
#define bli_cctyper  float
#define bli_zctyper  double


// return default format specifier for char

// NOTE: These must remain macros due to the way they are used to initialize
// local char arrays.

#define bli_sformatspec() "%9.2e"
#define bli_dformatspec() "%9.2e"
#define bli_cformatspec() "%9.2e + %9.2e "
#define bli_zformatspec() "%9.2e + %9.2e "
#define bli_iformatspec() "%6d"


#endif

