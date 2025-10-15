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

#include "blis.h"


#define PACKM_BODY( ctypea, ctypep, cha, chp, pragma, cdim, dfac, inca, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		ctypep delta_cast, kappa_delta; \
		PASTEMAC(cha,chp,copys)( *delta1, delta_cast ); \
		PASTEMAC(chp,scal2s)( kappa_cast, delta_cast, kappa_delta ); \
\
		pragma \
		for ( dim_t mn = 0; mn < cdim; mn++ ) \
		{ \
			ctypep alpha_cast, kappa_alpha; \
			PASTEMAC(cha,chp,copys)( *(alpha1 + mn*inca), alpha_cast ); \
			PASTEMAC(chp,op)( kappa_delta, alpha_cast, kappa_alpha ); \
			for ( dim_t d = 0; d < dfac; d++ ) \
				PASTEMAC(chp,copys)( kappa_alpha, *(pi1 + mn*dfac + d) ); \
		} \
\
		alpha1 += lda; \
		delta1 += incd; \
		pi1    += ldp; \
	} \
} while(0)


#undef  GENTFUNC2
#define GENTFUNC2( ctypea, ctypep, cha, chp, opname, arch, suf ) \
\
void PASTEMAC(cha,chp,opname,arch,suf) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   cdim_max, \
             dim_t   cdim_bcast, \
             dim_t   n, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
       const void*   d, inc_t incd, \
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx  \
     ) \
{ \
	const dim_t mr  = PASTECH(BLIS_MR_, chp); \
	const dim_t nr  = PASTECH(BLIS_NR_, chp); \
	const dim_t bbm = PASTECH(BLIS_BBM_, chp); \
	const dim_t bbn = PASTECH(BLIS_BBN_, chp); \
\
	      ctypep           kappa_cast = *( ctypep* )kappa; \
	const ctypea* restrict alpha1     = a; \
	const ctypea* restrict delta1     = d; \
	      ctypep* restrict pi1        = p; \
\
	if ( cdim == mr && cdim_bcast == bbm && mr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, mr, bbm, 1, scal2js ); \
			else                        PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, mr, bbm, 1, scal2s ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, mr, bbm, inca, scal2js ); \
			else                        PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, mr, bbm, inca, scal2s ); \
		} \
	} \
	else if ( cdim == nr && cdim_bcast == bbn && nr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, nr, bbn, 1, scal2js ); \
			else                        PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, nr, bbn, 1, scal2s ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, nr, bbn, inca, scal2js ); \
			else                        PACKM_BODY( ctypea, ctypep, cha, chp, PRAGMA_SIMD, nr, bbn, inca, scal2s ); \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) PACKM_BODY( ctypea, ctypep, cha, chp, , cdim, cdim_bcast, inca, scal2js ); \
		else                        PACKM_BODY( ctypea, ctypep, cha, chp, , cdim, cdim_bcast, inca, scal2s ); \
	} \
\
	PASTEMAC(chp,set0s_edge) \
	( \
	  cdim*cdim_bcast, cdim_max*cdim_bcast, \
	  n, n_max, \
	  p, ldp  \
	); \
}

INSERT_GENTFUNC2_BASIC( packmd, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC2_MIX_P( packmd, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

