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

#define FUNCPTR_T mulsc_fp

typedef void (*FUNCPTR_T)(
                           conj_t conjchi,
                           void*  chi,
                           void*  psi
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,mulsc_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,mulsc_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,mulsc_unb_var1);
#endif
#endif


void bli_mulsc_unb_var1( obj_t*  chi,
                         obj_t*  psi )
{
	conj_t    conjchi   = bli_obj_conj_status( *chi );

	num_t     dt_psi     = bli_obj_datatype( *psi );
	void*     buf_psi    = bli_obj_buffer_at_off( *psi );

	num_t     dt_chi;
	void*     buf_chi;

	FUNCPTR_T f;

	// If chi is a scalar constant, use dt_psi to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the chi object and extract the buffer at the chi offset.
	bli_set_scalar_dt_buffer( chi, dt_psi, dt_chi, buf_chi );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_chi][dt_psi];

	// Invoke the function.
	f( conjchi,
	   buf_chi,
	   buf_psi );
}


#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, varname ) \
\
void PASTEMAC2(chx,chy,varname)( \
                                 conj_t conjchi, \
                                 void*  chi, \
                                 void*  psi \
                               ) \
{ \
	ctype_x* chi_cast = chi; \
	ctype_y* psi_cast = psi; \
	ctype_x  chi_conj; \
\
	if ( PASTEMAC(chx,eq0)( *chi_cast ) ) \
	{ \
		PASTEMAC(chy,set0s)( *psi_cast ); \
	} \
	else \
	{ \
		PASTEMAC2(chx,chx,copycjs)( conjchi, *chi_cast, chi_conj ); \
		PASTEMAC2(chx,chy,scals)( chi_conj, *psi_cast ); \
	} \
}


// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC0( mulsc_unb_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D0( mulsc_unb_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P0( mulsc_unb_var1 )
#endif

