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

#define FUNCPTR_T unzipsc_fp

typedef void (*FUNCPTR_T)(
                           void*  beta,
                           void*  chi_r,
                           void*  chi_i
                         );

static FUNCPTR_T GENARRAY(ftypes,unzipsc_unb_var1);


void bli_unzipsc_unb_var1( obj_t* beta,
                           obj_t* chi_r,
                           obj_t* chi_i )
{
	num_t     dt_beta;
	num_t     dt_chi_c   = bli_obj_datatype_proj_to_complex( *chi_r );

	void*     buf_beta;

	void*     buf_chi_r  = bli_obj_buffer_at_off( *chi_r );
	void*     buf_chi_i  = bli_obj_buffer_at_off( *chi_i );

	FUNCPTR_T f;

	// If beta is a scalar constant, use dt_chi_c to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the beta object and extract the buffer at the beta offset.
	bli_set_scalar_dt_buffer( beta, dt_chi_c, dt_beta, buf_beta );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_beta];

	// Invoke the function.
	f( buf_beta,
	   buf_chi_r,
	   buf_chi_i );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_b, ctype_br, chb, chbr, varname ) \
\
void PASTEMAC(chb,varname)( \
                            void*  beta, \
                            void*  chi_r, \
                            void*  chi_i  \
                          ) \
{ \
	ctype_b*  beta_cast  = beta; \
	ctype_br* chi_r_cast = chi_r; \
	ctype_br* chi_i_cast = chi_i; \
\
	PASTEMAC2(chb,chbr,gets)( *beta_cast, \
	                          *chi_r_cast, \
	                          *chi_i_cast ); \
}

INSERT_GENTFUNCR_BASIC0( unzipsc_unb_var1 )

