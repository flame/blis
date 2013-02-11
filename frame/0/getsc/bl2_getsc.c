/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"

#define FUNCPTR_T getsc_fp

typedef void (*FUNCPTR_T)(
                           void*   chi,
                           double* beta_r,
                           double* beta_i
                         );

static FUNCPTR_T GENARRAY(ftypes,getsc);


void bl2_getsc( obj_t*  chi,
                double* beta_r,
                double* beta_i )
{
	num_t     dt_chi   = bl2_obj_datatype( *chi );

	void*     buf_chi  = bl2_obj_buffer_at_off( *chi );

	FUNCPTR_T f;

	if ( bl2_error_checking_is_enabled() )
		bl2_getsc_check( chi, beta_r, beta_i );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_chi];

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
	PASTEMAC2(chx,chxr,getris)( *chi_cast, *beta_r, *beta_i ); \
}

GENTFUNC( float,    double, s, d, getsc )
GENTFUNC( double,   double, d, d, getsc )
GENTFUNC( scomplex, double, c, d, getsc )
GENTFUNC( dcomplex, double, z, d, getsc )

