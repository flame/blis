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

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, mnr0, bb0, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t           conja, \
       pack_t           schema, \
       dim_t            cdim, \
       dim_t            n, \
       dim_t            n_max, \
       ctype*  restrict kappa, \
       ctype*  restrict a, inc_t inca, inc_t lda, \
       ctype*  restrict p,             inc_t ldp, \
       cntx_t* restrict cntx  \
     ) \
{ \
	const dim_t mnr  = PASTECH2(mnr0, _, ch); \
	const dim_t dfac = PASTECH2(bb0, _, ch); \
\
	if ( cdim == mnr && mnr != -1 ) \
	{ \
		if ( bli_is_1e_packed( schema ) ) \
		{ \
			const inc_t       inca2      = 2 * inca; \
			const inc_t       lda2       = 2 * lda; \
			const inc_t       ldp2       = 2 * ldp; \
\
			ctype_r           kappa_r    = ( ( ctype_r* )kappa )[0]; \
			ctype_r           kappa_i    = ( ( ctype_r* )kappa )[1]; \
			ctype_r* restrict alpha1     = ( ctype_r* )a; \
			ctype_r* restrict pi1_ri     = ( ctype_r* )p; \
			ctype_r* restrict pi1_ir     = ( ctype_r* )p + ldp; \
\
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
						{ \
							PASTEMAC(ch,scal2jris)(  kappa_r, kappa_i, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                           *(pi1_ri + (mn*2 + 0)*dfac + d), *(pi1_ri + (mn*2 + 1)*dfac + d) ); \
							PASTEMAC(ch,scal2jris)( -kappa_i, kappa_r, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                           *(pi1_ir + (mn*2 + 0)*dfac + d), *(pi1_ir + (mn*2 + 1)*dfac + d) ); \
						} \
\
						alpha1 += lda2; \
						pi1_ri += ldp2; \
						pi1_ir += ldp2; \
					} \
				} \
				else \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
						{ \
							PASTEMAC(ch,scal2ris)(  kappa_r, kappa_i, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                          *(pi1_ri + (mn*2 + 0)*dfac + d), *(pi1_ri + (mn*2 + 1)*dfac + d) ); \
							PASTEMAC(ch,scal2ris)( -kappa_i, kappa_r, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                          *(pi1_ir + (mn*2 + 0)*dfac + d), *(pi1_ir + (mn*2 + 1)*dfac + d) ); \
						} \
\
						alpha1 += lda2; \
						pi1_ri += ldp2; \
						pi1_ir += ldp2; \
					} \
				} \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
						{ \
							PASTEMAC(ch,scal2jris)(  kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                           *(pi1_ri + (mn*2 + 0)*dfac + d), *(pi1_ri + (mn*2 + 1)*dfac + d) ); \
							PASTEMAC(ch,scal2jris)( -kappa_i, kappa_r, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                           *(pi1_ir + (mn*2 + 0)*dfac + d), *(pi1_ir + (mn*2 + 1)*dfac + d) ); \
						} \
\
						alpha1 += lda2; \
						pi1_ri += ldp2; \
						pi1_ir += ldp2; \
					} \
				} \
				else \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
						{ \
							PASTEMAC(ch,scal2ris)(  kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                          *(pi1_ri + (mn*2 + 0)*dfac + d), *(pi1_ri + (mn*2 + 1)*dfac + d) ); \
							PASTEMAC(ch,scal2ris)( -kappa_i, kappa_r, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                          *(pi1_ir + (mn*2 + 0)*dfac + d), *(pi1_ir + (mn*2 + 1)*dfac + d) ); \
						} \
\
						alpha1 += lda2; \
						pi1_ri += ldp2; \
						pi1_ir += ldp2; \
					} \
				} \
			} \
		} \
		else \
		{ \
			const inc_t       inca2      = 2 * inca; \
			const inc_t       lda2       = 2 * lda; \
			const inc_t       ldp2       = 2 * ldp; \
\
			ctype_r           kappa_r    = ( ( ctype_r* )kappa )[0]; \
			ctype_r           kappa_i    = ( ( ctype_r* )kappa )[1]; \
			ctype_r* restrict alpha1     = ( ctype_r* )a; \
			ctype_r* restrict pi1_r      = ( ctype_r* )p; \
			ctype_r* restrict pi1_i      = ( ctype_r* )p + ldp; \
\
			if ( inca == 1 ) \
			{ \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
							PASTEMAC(ch,scal2jris)( kappa_r, kappa_i, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                          *(pi1_r + mn*dfac + d), *(pi1_i + mn*dfac + d) ); \
\
						alpha1 += lda2; \
						pi1_r  += ldp2; \
						pi1_i  += ldp2; \
					} \
				} \
				else \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
							PASTEMAC(ch,scal2ris)( kappa_r, kappa_i, *(alpha1 + mn*2 + 0), *(alpha1 + mn*2 + 1), \
							                                         *(pi1_r + mn*dfac + d), *(pi1_i + mn*dfac + d) ); \
\
						alpha1 += lda2; \
						pi1_r  += ldp2; \
						pi1_i  += ldp2; \
					} \
				} \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
							PASTEMAC(ch,scal2jris)( kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                          *(pi1_r + mn*dfac + d), *(pi1_i + mn*dfac + d) ); \
\
						alpha1 += lda2; \
						pi1_r  += ldp2; \
						pi1_i  += ldp2; \
					} \
				} \
				else \
				{ \
					for ( dim_t k = n; k != 0; --k ) \
					{ \
						PRAGMA_SIMD \
						for ( dim_t mn = 0; mn < mnr; ++mn ) \
						for ( dim_t d = 0; d < dfac; ++d ) \
							PASTEMAC(ch,scal2ris)( kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0), *(alpha1 + mn*inca2 + 1), \
							                                         *(pi1_r + mn*dfac + d), *(pi1_i + mn*dfac + d) ); \
\
						alpha1 += lda2; \
						pi1_r  += ldp2; \
						pi1_i  += ldp2; \
					} \
				} \
			} \
		} \
	} \
	else /* if ( cdim < mnr ) */ \
	{ \
		PASTEMAC(ch,scal21ms_mxn) \
		( \
		  schema, \
		  conja, \
		  cdim, \
		  n, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, ldp  \
		); \
\
		const dim_t       i      = cdim; \
		const dim_t       erfac  = bli_is_1e_packed( schema ) ? 2 : 1; \
		/* use ldp instead of mnr, in case the latter is -1 \
		   this may write extra zeros, but not too many \
		   this also automatically accounts for dfac when \
		   using set0s_mxn instead of set0bbs_mxn */ \
		const dim_t       m_edge = ldp - cdim*dfac*erfac; \
		const dim_t       n_edge = 2*n_max; \
		ctype_r* restrict p_cast = ( ctype_r* )p; \
		ctype_r* restrict p_edge = p_cast + (i  )*dfac*erfac; \
\
		PASTEMAC(chr,set0s_mxn) \
		( \
		  m_edge, \
		  n_edge, \
		  p_edge, 1, ldp  \
		); \
	} \
\
	const dim_t       i      = n; \
	/* use ldp instead of mnr, in case the latter is -1 \
	   this may write extra zeros, but not too many \
	   this also automatically accounts for dfac when \
	   using set0s_mxn instead of set0bbs_mxn */ \
	const dim_t       m_edge = ldp; \
	const dim_t       n_edge = 2*(n_max-i); \
	ctype_r* restrict p_cast = ( ctype_r* )p; \
	ctype_r* restrict p_edge = p_cast + (i  )*ldp*2; \
\
	PASTEMAC(chr,set0s_mxn) \
	( \
	  m_edge, \
	  n_edge, \
	  p_edge, 1, ldp  \
	); \
}

INSERT_GENTFUNCCO_BASIC4( packm_mrxk_1er, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNCCO_BASIC4( packm_nrxk_1er, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

