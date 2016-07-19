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

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_2xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_4xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_6xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_8xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_10xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_12xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_14xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_16xk_4mi_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast =             kappa; \
	ctype_r* restrict kappa_r    = ( ctype_r* )kappa; \
	ctype_r* restrict kappa_i    = ( ctype_r* )kappa + 1; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
	ctype_r* restrict pi1_i      = ( ctype_r* )p + is_p; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +16*inca2), *(alpha1_i +16*inca2), *(pi1_r +16), *(pi1_i +16) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +17*inca2), *(alpha1_i +17*inca2), *(pi1_r +17), *(pi1_i +17) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +18*inca2), *(alpha1_i +18*inca2), *(pi1_r +18), *(pi1_i +18) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +19*inca2), *(alpha1_i +19*inca2), *(pi1_r +19), *(pi1_i +19) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +20*inca2), *(alpha1_i +20*inca2), *(pi1_r +20), *(pi1_i +20) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +21*inca2), *(alpha1_i +21*inca2), *(pi1_r +21), *(pi1_i +21) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +22*inca2), *(alpha1_i +22*inca2), *(pi1_r +22), *(pi1_i +22) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +23*inca2), *(alpha1_i +23*inca2), *(pi1_r +23), *(pi1_i +23) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +24*inca2), *(alpha1_i +24*inca2), *(pi1_r +24), *(pi1_i +24) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +25*inca2), *(alpha1_i +25*inca2), *(pi1_r +25), *(pi1_i +25) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +26*inca2), *(alpha1_i +26*inca2), *(pi1_r +26), *(pi1_i +26) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +27*inca2), *(alpha1_i +27*inca2), *(pi1_r +27), *(pi1_i +27) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +28*inca2), *(alpha1_i +28*inca2), *(pi1_r +28), *(pi1_i +28) ); \
				PASTEMAC(ch,copyjris)( *(alpha1_r +29*inca2), *(alpha1_i +29*inca2), *(pi1_r +29), *(pi1_i +29) ); \
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
				PASTEMAC(ch,copyris)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +16*inca2), *(alpha1_i +16*inca2), *(pi1_r +16), *(pi1_i +16) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +17*inca2), *(alpha1_i +17*inca2), *(pi1_r +17), *(pi1_i +17) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +18*inca2), *(alpha1_i +18*inca2), *(pi1_r +18), *(pi1_i +18) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +19*inca2), *(alpha1_i +19*inca2), *(pi1_r +19), *(pi1_i +19) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +20*inca2), *(alpha1_i +20*inca2), *(pi1_r +20), *(pi1_i +20) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +21*inca2), *(alpha1_i +21*inca2), *(pi1_r +21), *(pi1_i +21) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +22*inca2), *(alpha1_i +22*inca2), *(pi1_r +22), *(pi1_i +22) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +23*inca2), *(alpha1_i +23*inca2), *(pi1_r +23), *(pi1_i +23) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +24*inca2), *(alpha1_i +24*inca2), *(pi1_r +24), *(pi1_i +24) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +25*inca2), *(alpha1_i +25*inca2), *(pi1_r +25), *(pi1_i +25) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +26*inca2), *(alpha1_i +26*inca2), *(pi1_r +26), *(pi1_i +26) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +27*inca2), *(alpha1_i +27*inca2), *(pi1_r +27), *(pi1_i +27) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +28*inca2), *(alpha1_i +28*inca2), *(pi1_r +28), *(pi1_i +28) ); \
				PASTEMAC(ch,copyris)( *(alpha1_r +29*inca2), *(alpha1_i +29*inca2), *(pi1_r +29), *(pi1_i +29) ); \
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
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +16*inca2), *(alpha1_i +16*inca2), *(pi1_r +16), *(pi1_i +16) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +17*inca2), *(alpha1_i +17*inca2), *(pi1_r +17), *(pi1_i +17) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +18*inca2), *(alpha1_i +18*inca2), *(pi1_r +18), *(pi1_i +18) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +19*inca2), *(alpha1_i +19*inca2), *(pi1_r +19), *(pi1_i +19) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +20*inca2), *(alpha1_i +20*inca2), *(pi1_r +20), *(pi1_i +20) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +21*inca2), *(alpha1_i +21*inca2), *(pi1_r +21), *(pi1_i +21) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +22*inca2), *(alpha1_i +22*inca2), *(pi1_r +22), *(pi1_i +22) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +23*inca2), *(alpha1_i +23*inca2), *(pi1_r +23), *(pi1_i +23) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +24*inca2), *(alpha1_i +24*inca2), *(pi1_r +24), *(pi1_i +24) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +25*inca2), *(alpha1_i +25*inca2), *(pi1_r +25), *(pi1_i +25) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +26*inca2), *(alpha1_i +26*inca2), *(pi1_r +26), *(pi1_i +26) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +27*inca2), *(alpha1_i +27*inca2), *(pi1_r +27), *(pi1_i +27) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +28*inca2), *(alpha1_i +28*inca2), *(pi1_r +28), *(pi1_i +28) ); \
				PASTEMAC(ch,scal2jris)( *kappa_r, *kappa_i, *(alpha1_r +29*inca2), *(alpha1_i +29*inca2), *(pi1_r +29), *(pi1_i +29) ); \
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
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0), *(pi1_i + 0) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1), *(pi1_i + 1) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2), *(pi1_i + 2) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3), *(pi1_i + 3) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4), *(pi1_i + 4) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5), *(pi1_i + 5) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6), *(pi1_i + 6) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7), *(pi1_i + 7) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8), *(pi1_i + 8) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9), *(pi1_i + 9) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10), *(pi1_i +10) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11), *(pi1_i +11) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12), *(pi1_i +12) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13), *(pi1_i +13) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14), *(pi1_i +14) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15), *(pi1_i +15) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +16*inca2), *(alpha1_i +16*inca2), *(pi1_r +16), *(pi1_i +16) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +17*inca2), *(alpha1_i +17*inca2), *(pi1_r +17), *(pi1_i +17) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +18*inca2), *(alpha1_i +18*inca2), *(pi1_r +18), *(pi1_i +18) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +19*inca2), *(alpha1_i +19*inca2), *(pi1_r +19), *(pi1_i +19) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +20*inca2), *(alpha1_i +20*inca2), *(pi1_r +20), *(pi1_i +20) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +21*inca2), *(alpha1_i +21*inca2), *(pi1_r +21), *(pi1_i +21) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +22*inca2), *(alpha1_i +22*inca2), *(pi1_r +22), *(pi1_i +22) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +23*inca2), *(alpha1_i +23*inca2), *(pi1_r +23), *(pi1_i +23) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +24*inca2), *(alpha1_i +24*inca2), *(pi1_r +24), *(pi1_i +24) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +25*inca2), *(alpha1_i +25*inca2), *(pi1_r +25), *(pi1_i +25) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +26*inca2), *(alpha1_i +26*inca2), *(pi1_r +26), *(pi1_i +26) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +27*inca2), *(alpha1_i +27*inca2), *(pi1_r +27), *(pi1_i +27) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +28*inca2), *(alpha1_i +28*inca2), *(pi1_r +28), *(pi1_i +28) ); \
				PASTEMAC(ch,scal2ris)( *kappa_r, *kappa_i, *(alpha1_r +29*inca2), *(alpha1_i +29*inca2), *(pi1_r +29), *(pi1_i +29) ); \
\
				alpha1_r += lda2; \
				alpha1_i += lda2; \
				pi1_r    += ldp; \
				pi1_i    += ldp; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_30xk_4mi_ref )

