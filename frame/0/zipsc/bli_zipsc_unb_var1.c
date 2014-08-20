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

#define FUNCPTR_T zipsc_fp

typedef void (*FUNCPTR_T)(
                           void*  beta_r,
                           void*  beta_i,
                           void*  chi
                         );

static FUNCPTR_T GENARRAY(ftypes,zipsc_unb_var1);


void bli_zipsc_unb_var1( obj_t*  beta_r,
                         obj_t*  beta_i,
                         obj_t*  chi )
{
	num_t     dt_beta_r;
	num_t     dt_beta_i;
	num_t     dt_chi     = bli_obj_datatype( *chi );
	num_t     dt_chi_r   = bli_obj_datatype_proj_to_real( *chi );

	void*     buf_beta_r;
	void*     buf_beta_i;

	void*     buf_chi    = bli_obj_buffer_at_off( *chi );

	FUNCPTR_T f;

	// If beta is a scalar constant, use dt_chi_r to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the beta object and extract the buffer at the beta offset.
	bli_set_scalar_dt_buffer( beta_r, dt_chi_r, dt_beta_r, buf_beta_r );
	bli_set_scalar_dt_buffer( beta_i, dt_chi_r, dt_beta_i, buf_beta_i );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_chi];

	// Invoke the function.
	f( buf_beta_r,
	   buf_beta_i,
	   buf_chi );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, varname ) \
\
void PASTEMAC(chx,varname)( \
                            void*  beta_r, \
                            void*  beta_i, \
                            void*  chi \
                          ) \
{ \
	ctype_xr* beta_r_cast = beta_r; \
	ctype_x*  chi_cast    = chi; \
\
	/* Inline casting and dereferencing of beta_i, instead of first assigning
	   to beta_i_cast, so that the compiler can't complain that the beta_i_cast
	   variable is unused for the real-only cases. */ \
	PASTEMAC2(chxr,chx,sets)( *beta_r_cast, \
	                          *( (ctype_xr*) beta_i ), \
	                          *chi_cast ); \
}

INSERT_GENTFUNCR_BASIC0( zipsc_unb_var1 )

