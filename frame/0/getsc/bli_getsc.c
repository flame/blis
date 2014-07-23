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

#define FUNCPTR_T getsc_fp

typedef void (*FUNCPTR_T)(
                           void*   chi,
                           double* beta_r,
                           double* beta_i
                         );

static FUNCPTR_T GENARRAY(ftypes,getsc);


void bli_getsc( obj_t*  chi,
                double* beta_r,
                double* beta_i )
{
	num_t     dt_chi   = bli_obj_datatype( *chi );
	num_t     dt_def   = BLIS_DCOMPLEX;
	num_t     dt_use;

	// If chi is a constant object, default to using the dcomplex
	// value within since we don't know if the caller needs just the
	// real or the real and imaginary parts.
	void*     buf_chi  = bli_obj_buffer_for_1x1( dt_def, *chi );

	FUNCPTR_T f;

	if ( bli_error_checking_is_enabled() )
		bli_getsc_check( chi, beta_r, beta_i );

	// The _check() routine prevents integer types, so we know that chi
	// is either a constant or an actual floating-point type.
	if ( bli_is_constant( dt_chi ) ) dt_use = dt_def;
	else                             dt_use = dt_chi;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_use];

	// Invoke the function.
	f( buf_chi,
	   beta_r,
	   beta_i );
}


#undef  GENTFUNC
#define GENTFUNC( ctype_x, ctype_xr, chx, chxr, opname ) \
\
void PASTEMAC(chx,opname)( \
                           void*     chi, \
                           ctype_xr* beta_r, \
                           ctype_xr* beta_i  \
                         ) \
{ \
	ctype_x* chi_cast = chi; \
\
	PASTEMAC2(chx,chxr,gets)( *chi_cast, *beta_r, *beta_i ); \
}

GENTFUNC( float,    double, s, d, getsc )
GENTFUNC( double,   double, d, d, getsc )
GENTFUNC( scomplex, double, c, d, getsc )
GENTFUNC( dcomplex, double, z, d, getsc )

