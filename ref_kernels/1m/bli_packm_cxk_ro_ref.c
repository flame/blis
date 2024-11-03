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


#define PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, pragma, cdim, dfac, inca2, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; ++mn ) \
		{ \
			ctypep_r ka_r, ka_i; \
			( void )ka_i; \
			PASTEMAC(t,op)( chp,cha,chp,chp, \
			                kappa_r, kappa_i, \
			                *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
			                ka_r, ka_i ); \
			for ( dim_t d = 0; d < dfac; ++d ) \
				bli_tcopys( chp_r,chp_r, ka_r, *(pi1_r + mn*dfac + d) ); \
		} \
\
		alpha1 += lda2; \
		pi1_r  += ldp; \
	} \
} while(0)


#undef  GENTFUNC2R
#define GENTFUNC2R( ctypea, ctypea_r, cha, cha_r, ctypep, ctypep_r, chp, chp_r, opname, arch, suf ) \
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
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx  \
     ) \
{ \
	const dim_t mr  = PASTECH(BLIS_MR_, chp_r); \
	const dim_t nr  = PASTECH(BLIS_NR_, chp_r); \
	const dim_t bbm = PASTECH(BLIS_BBM_, chp_r); \
	const dim_t bbn = PASTECH(BLIS_BBN_, chp_r); \
\
	const inc_t inca2 = 2 * inca; \
	const inc_t lda2  = 2 * lda; \
\
	      ctypep_r           kappa_r = ( ( ctypep_r* )kappa )[0]; \
	      ctypep_r           kappa_i = ( ( ctypep_r* )kappa )[1]; \
	const ctypea_r* restrict alpha1  = ( ctypea_r* )a; \
	      ctypep_r* restrict pi1_r   = ( ctypep_r* )p; \
\
	if ( cdim == mr && cdim_bcast == bbm && mr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, mr, bbm, 2, scal2jris ); \
			else                        PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, mr, bbm, 2, scal2ris ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, mr, bbm, inca2, scal2jris ); \
			else                        PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, mr, bbm, inca2, scal2ris ); \
		} \
	} \
	else if ( cdim == nr && cdim_bcast == bbn && nr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, nr, bbn, 2, scal2jris ); \
			else                        PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, nr, bbn, 2, scal2ris ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, nr, bbn, inca2, scal2jris ); \
			else                        PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, PRAGMA_SIMD, nr, bbn, inca2, scal2ris ); \
		} \
	} \
	else \
	{ \
		if ( bli_is_conj( conja ) ) PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, , cdim, cdim_bcast, inca2, scal2jris ); \
		else                        PACKM_RO_BODY( ctypep_r, cha, chp, chp_r, , cdim, cdim_bcast, inca2, scal2ris ); \
	} \
\
	bli_tset0s_edge \
	( \
	  chp_r, \
	  cdim*cdim_bcast, cdim_max*cdim_bcast, \
	  n, n_max, \
	  ( ctypep_r* )p, ldp  \
	); \
}

GENTFUNC2R( scomplex, float,  c, s, scomplex, float,  c, s, packm_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( scomplex, float,  c, s, dcomplex, double, z, d, packm_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, scomplex, float,  c, s, packm_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, dcomplex, double, z, d, packm_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

