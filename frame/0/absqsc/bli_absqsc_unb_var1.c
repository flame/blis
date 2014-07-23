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

#define FUNCPTR_T absqsc_fp

typedef void (*FUNCPTR_T)(
                           void*  chi,
                           void*  absq
                         );

static FUNCPTR_T GENARRAY(ftypes,absqsc_unb_var1);


void bli_absqsc_unb_var1( obj_t* chi,
                          obj_t* absq )
{
	num_t     dt_chi;
	num_t     dt_absq_c  = bli_obj_datatype_proj_to_complex( *absq );

	void*     buf_chi;

	void*     buf_absq   = bli_obj_buffer_at_off( *absq );

	FUNCPTR_T f;

	// If chi is a scalar constant, use dt_absq_c to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the chi object and extract the buffer at the chi offset.
	bli_set_scalar_dt_buffer( chi, dt_absq_c, dt_chi, buf_chi );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_chi];

	// Invoke the function.
	f( buf_chi,
	   buf_absq );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, varname ) \
\
void PASTEMAC(chx,varname)( \
                            void*  chi, \
                            void*  absq \
                          ) \
{ \
	ctype_x*  chi_cast  = chi; \
	ctype_xr* absq_cast = absq; \
	ctype_xr  chi_r; \
	ctype_xr  chi_i; \
\
	PASTEMAC2(chx,chxr,gets)( *chi_cast, \
	                          chi_r, \
	                          chi_i ); \
\
	/* absq = chi_r * chi_r + chi_i * chi_i; */ \
	PASTEMAC2(chxr,chxr,scals)( chi_r, chi_r ); \
	PASTEMAC2(chxr,chxr,scals)( chi_i, chi_i ); \
	PASTEMAC2(chxr,chxr,adds)( chi_i, chi_r ); \
	PASTEMAC2(chxr,chxr,copys)( chi_r, *absq_cast ); \
}

INSERT_GENTFUNCR_BASIC0( absqsc_unb_var1 )

