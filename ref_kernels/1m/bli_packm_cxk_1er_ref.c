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


#define PACKM_1E_BODY( ctype, ch, pragma, cdim, dfac, inca2, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; ++mn ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
		{ \
			PASTEMAC(ch,op)(  kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
			                                    *(pi1_ri + (mn*2 + 0)*dfac + d), *(pi1_ri + (mn*2 + 1)*dfac + d) ); \
			PASTEMAC(ch,op)( -kappa_i, kappa_r, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
			                                    *(pi1_ir + (mn*2 + 0)*dfac + d), *(pi1_ir + (mn*2 + 1)*dfac + d) ); \
		} \
\
		alpha1 += lda2; \
		pi1_ri += ldp2; \
		pi1_ir += ldp2; \
	} \
} while(0)


#define PACKM_1R_BODY( ctype, ch, pragma, cdim, dfac, inca2, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; ++mn ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PASTEMAC(ch,op)( kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
			                                   *(pi1_r + mn*dfac + d), *(pi1_i + mn*dfac + d) ); \
\
		alpha1 += lda2; \
		pi1_r  += ldp2; \
		pi1_i  += ldp2; \
	} \
} while(0)


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
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
	const dim_t mr  = PASTECH(BLIS_MR_, ch); \
	const dim_t nr  = PASTECH(BLIS_NR_, ch); \
	const dim_t bbm = PASTECH(BLIS_BBM_, ch); \
	const dim_t bbn = PASTECH(BLIS_BBN_, ch); \
\
	if ( bli_is_1e_packed( schema ) ) \
	{ \
		const dim_t cdim2 = 2 * cdim; \
		const inc_t inca2 = 2 * inca; \
		const inc_t lda2  = 2 * lda; \
		const inc_t ldp2  = 2 * ldp; \
\
		      ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
		      ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
		const ctype_r* restrict alpha1  = ( ctype_r* )a; \
		      ctype_r* restrict pi1_ri  = ( ctype_r* )p; \
		      ctype_r* restrict pi1_ir  = ( ctype_r* )p + ldp; \
\
		if ( cdim2 == mr && cdim_bcast == bbm && mr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mr/2, bbm, 2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mr/2, bbm, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mr/2, bbm, inca2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mr/2, bbm, inca2, scal2ris ); \
			} \
		} \
		else if ( cdim2 == nr && cdim_bcast == bbn && nr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, nr/2, bbn, 2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, nr/2, bbn, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, nr/2, bbn, inca2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, nr/2, bbn, inca2, scal2ris ); \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, , cdim, cdim_bcast, inca2, scal2jris ); \
			else                        PACKM_1E_BODY( ctype, ch, , cdim, cdim_bcast, inca2, scal2ris ); \
		} \
\
		PASTEMAC(chr,set0s_edge) \
		( \
		  cdim2*cdim_bcast, 2*cdim_max*cdim_bcast, \
		  2*n, 2*n_max, \
		  ( ctype_r* )p, ldp  \
		); \
	} \
	else /* ( bli_is_1r_packed( schema ) ) */ \
	{ \
		const inc_t inca2 = 2 * inca; \
		const inc_t lda2  = 2 * lda; \
		const inc_t ldp2  = 2 * ldp; \
\
		      ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
		      ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
		const ctype_r* restrict alpha1  = ( ctype_r* )a; \
		      ctype_r* restrict pi1_r   = ( ctype_r* )p; \
		      ctype_r* restrict pi1_i   = ( ctype_r* )p + ldp; \
\
		if ( cdim == mr && cdim_bcast == bbm && mr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca2, scal2ris ); \
			} \
		} \
		else if ( cdim == nr && cdim_bcast == bbn && nr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca2, scal2ris ); \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, , cdim, cdim_bcast, inca2, scal2jris ); \
			else                        PACKM_1R_BODY( ctype, ch, , cdim, cdim_bcast, inca2, scal2ris ); \
		} \
\
		PASTEMAC(chr,set0s_edge) \
		( \
		  cdim*cdim_bcast, cdim_max*cdim_bcast, \
		  2*n, 2*n_max, \
		  ( ctype_r* )p, ldp  \
		); \
	} \
}

INSERT_GENTFUNCCO( packm_1er, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

