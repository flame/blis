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


#define PACKM_SET1_1E( chr, mnk ) \
do { \
	PASTEMAC(chr,set1s)( *(pi1_ri + (mnk*2 + 0)*dfac + d + mnk*ldp2) ); \
	PASTEMAC(chr,set0s)( *(pi1_ri + (mnk*2 + 1)*dfac + d + mnk*ldp2) ); \
	PASTEMAC(chr,set0s)( *(pi1_ir + (mnk*2 + 0)*dfac + d + mnk*ldp2) ); \
	PASTEMAC(chr,set1s)( *(pi1_ir + (mnk*2 + 1)*dfac + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SET1_1R( chr, mnk ) \
do { \
	PASTEMAC(chr,set1s)( *(pi1_r + mnk*dfac + d + mnk*ldp2) ); \
	PASTEMAC(chr,set0s)( *(pi1_i + mnk*dfac + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SCAL_1E( ch, mn, k, op ) \
do { \
	PASTEMAC(ch,op)(  kappa_r, kappa_i, *(alpha1 +  mn       *inca2 + 0 + k*lda2), \
	                                    *(alpha1 +  mn       *inca2 + 1 + k*lda2), \
	                                    *(pi1_ri + (mn*2 + 0)*dfac  + d + k*ldp2), \
	                                    *(pi1_ri + (mn*2 + 1)*dfac  + d + k*ldp2) ); \
	PASTEMAC(ch,op)( -kappa_i, kappa_r, *(alpha1 +  mn       *inca2 + 0 + k*lda2), \
	                                    *(alpha1 +  mn       *inca2 + 1 + k*lda2), \
	                                    *(pi1_ir + (mn*2 + 0)*dfac  + d + k*ldp2), \
	                                    *(pi1_ir + (mn*2 + 1)*dfac  + d + k*ldp2) ); \
} while (0)


#define PACKM_SCAL_1R( ch, mn, k, op ) \
do { \
	PASTEMAC(ch,op)( kappa_r, kappa_i, *(alpha1 + mn*inca2 + 0 + k*lda2), \
	                                   *(alpha1 + mn*inca2 + 1 + k*lda2), \
	                                   *(pi1_r  + mn*dfac  + d + k*ldp2), \
	                                   *(pi1_i  + mn*dfac  + d + k*ldp2) ); \
} while (0)


#define PACKM_DIAG_1E_BODY( ch, mn_min, mn_max, inca2_lu, lda2_lu, op ) \
\
do \
{ \
	/* PACKM_SCAL_1E assumes inca2 and lda2 are the strides to use. */ \
	dim_t inca2 = inca2_lu; \
	dim_t lda2 = lda2_lu; \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < dfac; d++ ) \
		PACKM_SCAL_1E( ch, mn, k, op ); \
} while(0)


#define PACKM_DIAG_BODY_1E_L( ch, op ) \
	PACKM_DIAG_1E_BODY( ch, k+1, cdim, inca_l2, lda_l2, op )

#define PACKM_DIAG_BODY_1E_U( ch, op ) \
	PACKM_DIAG_1E_BODY( ch, 0, k, inca_u2, lda_u2, op )


#define PACKM_DIAG_1R_BODY( ch, mn_min, mn_max, inca2_lu, lda2_lu, op ) \
\
do \
{ \
	/* PACKM_SCAL_1R assumes inca2 and lda2 are the strides to use. */ \
	dim_t inca2 = inca2_lu; \
	dim_t lda2 = lda2_lu; \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < dfac; d++ ) \
		PACKM_SCAL_1R( ch, mn, k, op ); \
} while(0)


#define PACKM_DIAG_BODY_1R_L( ch, op ) \
	PACKM_DIAG_1R_BODY( ch, k+1, cdim, inca_l2, lda_l2, op )

#define PACKM_DIAG_BODY_1R_U( ch, op ) \
	PACKM_DIAG_1R_BODY( ch, 0, k, inca_u2, lda_u2, op )


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, mnr0, bb0, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             struc_t struca, \
             diag_t  diaga, \
             uplo_t  uploa, \
             conj_t  conja, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   cdim, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const cntx_t* cntx \
     ) \
{ \
	const num_t dt_r      = PASTEMAC(chr,type); \
	const dim_t cdim_pack = bli_cntx_get_blksz_max_dt( dt_r, mnr0, cntx ); \
	const dim_t dfac      = bli_cntx_get_blksz_def_dt( dt_r, bb0, cntx ); \
\
	/* start by zeroing out the whole block */ \
	PASTEMAC(chr,set0s_mxn) \
	( \
	  cdim_pack, \
	  2*n_max, \
	  ( ctype_r* )p, 1, ldp  \
	); \
\
	const inc_t       inca2   = 2 * inca; \
	const inc_t       lda2    = 2 * lda; \
	const inc_t       ldp2    = 2 * ldp; \
\
	      ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
	      ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
	const ctype_r* restrict alpha1  = ( const ctype_r* )a; \
\
	if ( bli_is_1e_packed( schema ) ) \
	{ \
		const dim_t       cdim_max = bli_cntx_get_blksz_def_dt( dt_r, mnr0, cntx ) / 2; \
\
		ctype_r* restrict pi1_ri   = ( ctype_r* )p; \
		ctype_r* restrict pi1_ir   = ( ctype_r* )p + ldp; \
\
		/* write the strictly lower part if it exists */ \
		if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) ) \
		{ \
			dim_t  inca_l2 = inca2; \
			dim_t  lda_l2  = lda2; \
			conj_t conja_l = conja; \
\
			if ( bli_is_upper( uploa ) ) \
			{ \
				bli_swap_incs( &inca_l2, &lda_l2 ); \
				if ( bli_is_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_l ); \
			} \
\
			if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_1E_L( ch, scal2jris ); \
			else                          PACKM_DIAG_BODY_1E_L( ch, scal2ris ); \
		} \
\
		/* write the strictly upper part if it exists */ \
		/* assume either symmetric, hermitian, or triangular */ \
		if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) ) \
		{ \
			dim_t  inca_u2 = inca2; \
			dim_t  lda_u2  = lda2; \
			conj_t conja_u = conja; \
\
			if ( bli_is_lower( uploa ) ) \
			{ \
				bli_swap_incs( &inca_u2, &lda_u2 ); \
				if ( bli_is_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_u ); \
			} \
\
			if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_1E_U( ch, scal2jris ); \
			else                          PACKM_DIAG_BODY_1E_U( ch, scal2ris ); \
		} \
\
		/* write the diagonal */ \
		if ( bli_is_unit_diag( diaga ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SET1_1E( chr, mnk ); \
		} \
		else if ( bli_is_hermitian( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
			{ \
				ctype_r mu_r = *(alpha1 + mnk*(inca2 + lda2)); \
				PASTEMAC(chr,scal2s)(  kappa_r, mu_r, *(pi1_ri + (mnk*2 + 0)*dfac + d + mnk*ldp2) ); \
				PASTEMAC(chr,scal2s)(  kappa_i, mu_r, *(pi1_ri + (mnk*2 + 1)*dfac + d + mnk*ldp2) ); \
				PASTEMAC(chr,scal2s)( -kappa_i, mu_r, *(pi1_ir + (mnk*2 + 0)*dfac + d + mnk*ldp2) ); \
				PASTEMAC(chr,scal2s)(  kappa_r, mu_r, *(pi1_ir + (mnk*2 + 1)*dfac + d + mnk*ldp2) ); \
			} \
		} \
		else if ( bli_is_conj( conja )) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SCAL_1E( ch, mnk, mnk, scal2jris ); \
		} \
		else \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SCAL_1E( ch, mnk, mnk, scal2ris ); \
		} \
\
		/* invert the diagonal if requested */ \
		if ( invdiag ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
			{ \
				PASTEMAC(ch,invertris)( *(pi1_ri + (mnk*2 + 0)*dfac + d + mnk*ldp2), \
				                        *(pi1_ri + (mnk*2 + 1)*dfac + d + mnk*ldp2) ); \
				PASTEMAC(ch,copyjris)( *(pi1_ri + (mnk*2 + 0)*dfac + d + mnk*ldp2), \
				                       *(pi1_ri + (mnk*2 + 1)*dfac + d + mnk*ldp2), \
				                       *(pi1_ir + (mnk*2 + 1)*dfac + d + mnk*ldp2), \
				                       *(pi1_ir + (mnk*2 + 0)*dfac + d + mnk*ldp2) ); \
			} \
		} \
\
		/* if this an edge case in both directions, extend the diagonal with ones */ \
		for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PACKM_SET1_1E( chr, mnk ); \
	} \
	else /* bli_is_1r_packed( schema ) */ \
	{ \
		const dim_t       cdim_max = bli_cntx_get_blksz_def_dt( dt_r, mnr0, cntx ); \
\
		ctype_r* restrict pi1_r    = ( ctype_r* )p; \
		ctype_r* restrict pi1_i    = ( ctype_r* )p + ldp; \
\
		/* write the strictly lower part if it exists */ \
		if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) ) \
		{ \
			dim_t  inca_l2 = inca2; \
			dim_t  lda_l2  = lda2; \
			conj_t conja_l = conja; \
\
			if ( bli_is_upper( uploa ) ) \
			{ \
				bli_swap_incs( &inca_l2, &lda_l2 ); \
				if ( bli_is_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_l ); \
			} \
\
			if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_1R_L( ch, scal2jris ); \
			else                          PACKM_DIAG_BODY_1R_L( ch, scal2ris ); \
		} \
\
		/* write the strictly upper part if it exists */ \
		/* assume either symmetric, hermitian, or triangular */ \
		if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) ) \
		{ \
			dim_t  inca_u2 = inca2; \
			dim_t  lda_u2  = lda2; \
			conj_t conja_u = conja; \
\
			if ( bli_is_lower( uploa ) ) \
			{ \
				bli_swap_incs( &inca_u2, &lda_u2 ); \
				if ( bli_is_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_u ); \
			} \
\
			if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_1R_U( ch, scal2jris ); \
			else                          PACKM_DIAG_BODY_1R_U( ch, scal2ris ); \
		} \
\
		/* write the diagonal */ \
		if ( bli_is_unit_diag( diaga ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SET1_1R( chr, mnk ); \
		} \
		else if ( bli_is_hermitian( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
			{ \
				ctype_r mu_r = *(alpha1 + mnk*(inca2 + lda2)); \
				PASTEMAC(chr,scal2s)( kappa_r, mu_r, *(pi1_r + mnk*(dfac + ldp2) + d) ); \
				PASTEMAC(chr,scal2s)( kappa_i, mu_r, *(pi1_i + mnk*(dfac + ldp2) + d) ); \
			} \
		} \
		else if ( bli_is_conj( conja ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SCAL_1R( ch, mnk, mnk, scal2jris ); \
		} \
		else \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PACKM_SCAL_1R( ch, mnk, mnk, scal2ris ); \
		} \
\
		/* invert the diagonal if requested */ \
		if ( invdiag ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < dfac; ++d ) \
				PASTEMAC(ch,invertris)( *(pi1_r + mnk*(dfac + ldp2) + d), \
				                        *(pi1_i + mnk*(dfac + ldp2) + d) ); \
		} \
\
		/* if this an edge case in both directions, extend the diagonal with ones */ \
		for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PACKM_SET1_1R( chr, mnk ); \
	} \
}

INSERT_GENTFUNCCO_BASIC4( packm_mrxmr_diag_1er, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNCCO_BASIC4( packm_nrxnr_diag_1er, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

