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
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_2xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_4xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_6xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_8xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_10xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_12xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_14xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_16xk_1e_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca1      = inca; \
	const inc_t       lda1       = lda; \
	const inc_t       ldp1       = ldp; \
\
	ctype*   restrict kappa_cast = ( ctype* )kappa; \
	ctype*   restrict alpha1_ri  = ( ctype* )a; \
	ctype*   restrict pi1_ri     = ( ctype* )p; \
	ctype*   restrict pi1_ir     = ( ctype* )p + ldp1/2; \
\
	if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +16*inca1), *(pi1_ri +16), *(pi1_ir +16) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +17*inca1), *(pi1_ri +17), *(pi1_ir +17) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +18*inca1), *(pi1_ri +18), *(pi1_ir +18) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +19*inca1), *(pi1_ri +19), *(pi1_ir +19) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +20*inca1), *(pi1_ri +20), *(pi1_ir +20) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +21*inca1), *(pi1_ri +21), *(pi1_ir +21) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +22*inca1), *(pi1_ri +22), *(pi1_ir +22) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +23*inca1), *(pi1_ri +23), *(pi1_ir +23) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +24*inca1), *(pi1_ri +24), *(pi1_ir +24) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +25*inca1), *(pi1_ri +25), *(pi1_ir +25) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +26*inca1), *(pi1_ri +26), *(pi1_ir +26) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +27*inca1), *(pi1_ri +27), *(pi1_ir +27) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +28*inca1), *(pi1_ri +28), *(pi1_ir +28) ); \
				PASTEMAC(ch,copyj1es)( *(alpha1_ri +29*inca1), *(pi1_ri +29), *(pi1_ir +29) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +16*inca1), *(pi1_ri +16), *(pi1_ir +16) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +17*inca1), *(pi1_ri +17), *(pi1_ir +17) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +18*inca1), *(pi1_ri +18), *(pi1_ir +18) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +19*inca1), *(pi1_ri +19), *(pi1_ir +19) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +20*inca1), *(pi1_ri +20), *(pi1_ir +20) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +21*inca1), *(pi1_ri +21), *(pi1_ir +21) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +22*inca1), *(pi1_ri +22), *(pi1_ir +22) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +23*inca1), *(pi1_ri +23), *(pi1_ir +23) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +24*inca1), *(pi1_ri +24), *(pi1_ir +24) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +25*inca1), *(pi1_ri +25), *(pi1_ir +25) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +26*inca1), *(pi1_ri +26), *(pi1_ir +26) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +27*inca1), *(pi1_ri +27), *(pi1_ir +27) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +28*inca1), *(pi1_ri +28), *(pi1_ir +28) ); \
				PASTEMAC(ch,copy1es)( *(alpha1_ri +29*inca1), *(pi1_ri +29), *(pi1_ir +29) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +16*inca1), *(pi1_ri +16), *(pi1_ir +16) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +17*inca1), *(pi1_ri +17), *(pi1_ir +17) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +18*inca1), *(pi1_ri +18), *(pi1_ir +18) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +19*inca1), *(pi1_ri +19), *(pi1_ir +19) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +20*inca1), *(pi1_ri +20), *(pi1_ir +20) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +21*inca1), *(pi1_ri +21), *(pi1_ir +21) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +22*inca1), *(pi1_ri +22), *(pi1_ir +22) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +23*inca1), *(pi1_ri +23), *(pi1_ir +23) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +24*inca1), *(pi1_ri +24), *(pi1_ir +24) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +25*inca1), *(pi1_ri +25), *(pi1_ir +25) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +26*inca1), *(pi1_ri +26), *(pi1_ir +26) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +27*inca1), *(pi1_ri +27), *(pi1_ir +27) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +28*inca1), *(pi1_ri +28), *(pi1_ir +28) ); \
				PASTEMAC(ch,scal2j1es)( *kappa_cast, *(alpha1_ri +29*inca1), *(pi1_ri +29), *(pi1_ir +29) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
		else \
		{ \
			for ( ; n != 0; --n ) \
			{ \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 0*inca1), *(pi1_ri + 0), *(pi1_ir + 0) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 1*inca1), *(pi1_ri + 1), *(pi1_ir + 1) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 2*inca1), *(pi1_ri + 2), *(pi1_ir + 2) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 3*inca1), *(pi1_ri + 3), *(pi1_ir + 3) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 4*inca1), *(pi1_ri + 4), *(pi1_ir + 4) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 5*inca1), *(pi1_ri + 5), *(pi1_ir + 5) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 6*inca1), *(pi1_ri + 6), *(pi1_ir + 6) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 7*inca1), *(pi1_ri + 7), *(pi1_ir + 7) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 8*inca1), *(pi1_ri + 8), *(pi1_ir + 8) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri + 9*inca1), *(pi1_ri + 9), *(pi1_ir + 9) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +10*inca1), *(pi1_ri +10), *(pi1_ir +10) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +11*inca1), *(pi1_ri +11), *(pi1_ir +11) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +12*inca1), *(pi1_ri +12), *(pi1_ir +12) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +13*inca1), *(pi1_ri +13), *(pi1_ir +13) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +14*inca1), *(pi1_ri +14), *(pi1_ir +14) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +15*inca1), *(pi1_ri +15), *(pi1_ir +15) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +16*inca1), *(pi1_ri +16), *(pi1_ir +16) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +17*inca1), *(pi1_ri +17), *(pi1_ir +17) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +18*inca1), *(pi1_ri +18), *(pi1_ir +18) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +19*inca1), *(pi1_ri +19), *(pi1_ir +19) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +20*inca1), *(pi1_ri +20), *(pi1_ir +20) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +21*inca1), *(pi1_ri +21), *(pi1_ir +21) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +22*inca1), *(pi1_ri +22), *(pi1_ir +22) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +23*inca1), *(pi1_ri +23), *(pi1_ir +23) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +24*inca1), *(pi1_ri +24), *(pi1_ir +24) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +25*inca1), *(pi1_ri +25), *(pi1_ir +25) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +26*inca1), *(pi1_ri +26), *(pi1_ir +26) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +27*inca1), *(pi1_ri +27), *(pi1_ir +27) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +28*inca1), *(pi1_ri +28), *(pi1_ir +28) ); \
				PASTEMAC(ch,scal21es)( *kappa_cast, *(alpha1_ri +29*inca1), *(pi1_ri +29), *(pi1_ir +29) ); \
\
				alpha1_ri += lda1; \
				pi1_ri    += ldp1; \
				pi1_ir    += ldp1; \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_30xk_1e_ref )

