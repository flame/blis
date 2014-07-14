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

#include "blis.h"

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conja, \
                           dim_t   n, \
                           void*   beta, \
                           void*   a, inc_t inca, inc_t lda, \
                           void*   p,             inc_t ldp  \
                         ) \
{ \
	ctype* restrict beta_cast = beta; \
	ctype* restrict alpha1    = a; \
	ctype* restrict pi1       = p; \
\
	if ( PASTEMAC(ch,eq1)( *beta_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC2(ch,ch,copyjs)( *(alpha1 + 0*inca), *(pi1 + 0) ); \
				PASTEMAC2(ch,ch,copyjs)( *(alpha1 + 1*inca), *(pi1 + 1) ); \
				PASTEMAC2(ch,ch,copyjs)( *(alpha1 + 2*inca), *(pi1 + 2) ); \
				PASTEMAC2(ch,ch,copyjs)( *(alpha1 + 3*inca), *(pi1 + 3) ); \
\
				alpha1 += lda; \
				pi1    += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC2(ch,ch,copys)( *(alpha1 + 0*inca), *(pi1 + 0) ); \
				PASTEMAC2(ch,ch,copys)( *(alpha1 + 1*inca), *(pi1 + 1) ); \
				PASTEMAC2(ch,ch,copys)( *(alpha1 + 2*inca), *(pi1 + 2) ); \
				PASTEMAC2(ch,ch,copys)( *(alpha1 + 3*inca), *(pi1 + 3) ); \
\
				alpha1 += lda; \
				pi1    += ldp; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC3(ch,ch,ch,scal2js)( *beta_cast, *(alpha1 + 0*inca), *(pi1 + 0) ); \
				PASTEMAC3(ch,ch,ch,scal2js)( *beta_cast, *(alpha1 + 1*inca), *(pi1 + 1) ); \
				PASTEMAC3(ch,ch,ch,scal2js)( *beta_cast, *(alpha1 + 2*inca), *(pi1 + 2) ); \
				PASTEMAC3(ch,ch,ch,scal2js)( *beta_cast, *(alpha1 + 3*inca), *(pi1 + 3) ); \
\
				alpha1 += lda; \
				pi1    += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC3(ch,ch,ch,scal2s)( *beta_cast, *(alpha1 + 0*inca), *(pi1 + 0) ); \
				PASTEMAC3(ch,ch,ch,scal2s)( *beta_cast, *(alpha1 + 1*inca), *(pi1 + 1) ); \
				PASTEMAC3(ch,ch,ch,scal2s)( *beta_cast, *(alpha1 + 2*inca), *(pi1 + 2) ); \
				PASTEMAC3(ch,ch,ch,scal2s)( *beta_cast, *(alpha1 + 3*inca), *(pi1 + 3) ); \
\
				alpha1 += lda; \
				pi1    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_ref_4xk )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conja, \
                           dim_t   n, \
                           void*   beta, \
                           void*   a, inc_t inca, inc_t lda, \
                           void*   p, inc_t psp,  inc_t ldp  \
                         ) \
{ \
	const inc_t       inca2     = 2 * inca; \
	const inc_t       lda2      = 2 * lda; \
\
	ctype*            beta_cast =             beta; \
	ctype_r* restrict beta_r    = ( ctype_r* )beta; \
	ctype_r* restrict beta_i    = ( ctype_r* )beta + 1; \
	ctype_r* restrict alpha1_r  = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i  = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r     = ( ctype_r* )p; \
	ctype_r* restrict pi1_i     = ( ctype_r* )p + psp; \
\
	if ( PASTEMAC(ch,eq1)( *beta_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2jris)( *beta_r, *beta_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *beta_r, *beta_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *beta_r, *beta_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *beta_r, *beta_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2ris)( *beta_r, *beta_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *beta_r, *beta_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *beta_r, *beta_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *beta_r, *beta_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_ref_4xk_ri )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conja, \
                           dim_t   n, \
                           void*   beta, \
                           void*   a, inc_t inca, inc_t lda, \
                           void*   p, inc_t psp,  inc_t ldp  \
                         ) \
{ \
	const inc_t       inca2     = 2 * inca; \
	const inc_t       lda2      = 2 * lda; \
\
	ctype*            beta_cast =             beta; \
	ctype_r* restrict beta_r    = ( ctype_r* )beta; \
	ctype_r* restrict beta_i    = ( ctype_r* )beta + 1; \
	ctype_r* restrict alpha1_r  = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i  = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r     = ( ctype_r* )p; \
	ctype_r* restrict pi1_i     = ( ctype_r* )p +   psp; \
	ctype_r* restrict pi1_ri    = ( ctype_r* )p + 2*psp; \
\
	if ( PASTEMAC(ch,eq1)( *beta_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjri3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0), *(pi1_ri + 0) ); \
				PASTEMAC(ch,copyjri3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1), *(pi1_ri + 1) ); \
				PASTEMAC(ch,copyjri3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2), *(pi1_ri + 2) ); \
				PASTEMAC(ch,copyjri3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3), *(pi1_ri + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
				pi1_ri   += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyri3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0), *(pi1_ri + 0) ); \
				PASTEMAC(ch,copyri3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1), *(pi1_ri + 1) ); \
				PASTEMAC(ch,copyri3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2), *(pi1_ri + 2) ); \
				PASTEMAC(ch,copyri3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3), *(pi1_ri + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
				pi1_ri   += ldp; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2jri3s)( *beta_r, *beta_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0), *(pi1_ri + 0) ); \
				PASTEMAC(ch,scal2jri3s)( *beta_r, *beta_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1), *(pi1_ri + 1) ); \
				PASTEMAC(ch,scal2jri3s)( *beta_r, *beta_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2), *(pi1_ri + 2) ); \
				PASTEMAC(ch,scal2jri3s)( *beta_r, *beta_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3), *(pi1_ri + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
				pi1_ri   += ldp; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2ri3s)( *beta_r, *beta_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0), *(pi1_ri + 0) ); \
				PASTEMAC(ch,scal2ri3s)( *beta_r, *beta_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1), *(pi1_ri + 1) ); \
				PASTEMAC(ch,scal2ri3s)( *beta_r, *beta_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2), *(pi1_ri + 2) ); \
				PASTEMAC(ch,scal2ri3s)( *beta_r, *beta_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3), *(pi1_ri + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
				pi1_ri   += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_ref_4xk_ri3 )

