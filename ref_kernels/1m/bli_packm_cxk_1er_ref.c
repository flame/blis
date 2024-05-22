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


#define PACKM_1E_BODY( ctype, ch, pragma, cdim, inca2, op ) \
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


#define PACKM_1R_BODY( ctype, ch, pragma, cdim, inca2, op ) \
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
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, mnr0, bb0, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   n, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const cntx_t* cntx  \
     ) \
{ \
	const dim_t dfac = PASTECH2(bb0, _, chr); \
	const num_t dt_r = PASTEMAC(chr,type); \
\
	if ( bli_is_1e_packed( schema ) ) \
	{ \
		/* cdim and mnr are in units of complex values */ \
		const dim_t mnr      = PASTECH2(mnr0, _, chr) == -1 ? -1 : PASTECH2(mnr0, _, chr) / 2; \
		const dim_t cdim_max = bli_cntx_get_blksz_def_dt( dt_r, mnr0, cntx ) / 2; \
\
		const inc_t       inca2   = 2 * inca; \
		const inc_t       lda2    = 2 * lda; \
		const inc_t       ldp2    = 2 * ldp; \
\
		      ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
		      ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
		const ctype_r* restrict alpha1  = ( ctype_r* )a; \
		      ctype_r* restrict pi1_ri  = ( ctype_r* )p; \
		      ctype_r* restrict pi1_ir  = ( ctype_r* )p + ldp; \
\
		if ( cdim == mnr && mnr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mnr, 2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mnr, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca2, scal2jris ); \
				else                        PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca2, scal2ris ); \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_1E_BODY( ctype, ch, , cdim, inca2, scal2jris ); \
			else                        PACKM_1E_BODY( ctype, ch, , cdim, inca2, scal2ris ); \
		} \
\
		PASTEMAC(chr,set0s_edge) \
		( \
		  2*cdim*dfac, 2*cdim_max*dfac, \
		  2*n, 2*n_max, \
		  ( ctype_r* )p, ldp  \
		); \
	} \
	else /* ( bli_is_1r_packed( schema ) ) */ \
	{ \
		const dim_t mnr      = PASTECH2(mnr0, _, chr); \
		const dim_t cdim_max = bli_cntx_get_blksz_def_dt( dt_r, mnr0, cntx ); \
\
		const inc_t       inca2   = 2 * inca; \
		const inc_t       lda2    = 2 * lda; \
		const inc_t       ldp2    = 2 * ldp; \
\
		      ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
		      ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
		const ctype_r* restrict alpha1  = ( ctype_r* )a; \
		      ctype_r* restrict pi1_r   = ( ctype_r* )p; \
		      ctype_r* restrict pi1_i   = ( ctype_r* )p + ldp; \
\
		if ( cdim == mnr && mnr != -1 ) \
		{ \
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mnr, 2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mnr, 2, scal2ris ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca2, scal2jris ); \
				else                        PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca2, scal2ris ); \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_1R_BODY( ctype, ch, , cdim, inca2, scal2jris ); \
			else                        PACKM_1R_BODY( ctype, ch, , cdim, inca2, scal2ris ); \
		} \
\
		PASTEMAC(chr,set0s_edge) \
		( \
		  cdim*dfac, cdim_max*dfac, \
		  2*n, 2*n_max, \
		  ( ctype_r* )p, ldp  \
		); \
	} \
}

INSERT_GENTFUNCCO( packm_mrxk_1er, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNCCO( packm_nrxk_1er, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

