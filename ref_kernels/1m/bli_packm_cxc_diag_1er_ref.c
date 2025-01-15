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


#define PACKM_SET_1E( chp_r, val_r, val_i, mnk ) \
do { \
	PASTEMAC(chp_r,copys)(  val_r, *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,copys)(  val_i, *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,copys)( -val_i, *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,copys)(  val_r, *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SET_1R( chp_r, val_r, val_i, mnk ) \
do { \
	PASTEMAC(chp_r,copys)( val_r, *(pi1_r + mnk*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,copys)( val_i, *(pi1_i + mnk*cdim_bcast + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SET0_1E( chp_r, mnk ) \
do { \
	PASTEMAC(chp_r,set0s)( *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,set0s)( *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,set0s)( *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,set0s)( *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SET0_1R( chp_r, mnk ) \
do { \
	PASTEMAC(chp_r,set0s)( *(pi1_r + mnk*cdim_bcast + d + mnk*ldp2) ); \
	PASTEMAC(chp_r,set0s)( *(pi1_i + mnk*cdim_bcast + d + mnk*ldp2) ); \
} while (0)


#define PACKM_SCAL_1E( ctypep_r, cha, chp, mn, k, op ) \
do { \
	ctypep_r alpha_r, alpha_i, ka_r, ka_i; \
	PASTEMAC(cha,chp,copyris)( *(alpha1 +  mn       *inca2       + 0 + k*lda2), \
	                           *(alpha1 +  mn       *inca2       + 1 + k*lda2), \
	                           alpha_r, alpha_i ); \
	PASTEMAC(chp,op)( kappa_r, kappa_i, alpha_r, alpha_i, ka_r, ka_i ); \
	PASTEMAC(chp,copyris)(  ka_r, ka_i, *(pi1_ri + (mn*2 + 0)*cdim_bcast  + d + k*ldp2), \
	                                    *(pi1_ri + (mn*2 + 1)*cdim_bcast  + d + k*ldp2) ); \
	PASTEMAC(chp,copyris)( -ka_i, ka_r, *(pi1_ir + (mn*2 + 0)*cdim_bcast  + d + k*ldp2), \
	                                    *(pi1_ir + (mn*2 + 1)*cdim_bcast  + d + k*ldp2) ); \
} while (0)


#define PACKM_SCAL_1R( ctypep_r, cha, chp, mn, k, op ) \
do { \
	ctypep_r alpha_r, alpha_i, ka_r, ka_i; \
	PASTEMAC(cha,chp,copyris)( *(alpha1 +  mn       *inca2       + 0 + k*lda2), \
	                           *(alpha1 +  mn       *inca2       + 1 + k*lda2), \
	                           alpha_r, alpha_i ); \
	PASTEMAC(chp,op)( kappa_r, kappa_i, alpha_r, alpha_i, ka_r, ka_i ); \
	PASTEMAC(chp,copyris)( ka_r, ka_i, *(pi1_r  + mn*cdim_bcast  + d + k*ldp2), \
	                                   *(pi1_i  + mn*cdim_bcast  + d + k*ldp2) ); \
} while (0)


#define PACKM_DIAG_1E_BODY( ctypep_r, cha, chp, mn_min, mn_max, inca2_lu, lda2_lu, op ) \
\
do \
{ \
	/* PACKM_SCAL_1E assumes inca2 and lda2 are the strides to use. */ \
	dim_t inca2 = inca2_lu; \
	dim_t lda2 = lda2_lu; \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < cdim_bcast; d++ ) \
		PACKM_SCAL_1E( ctypep_r, cha, chp, mn, k, op ); \
} while(0)


#define PACKM_DIAG_BODY_1E_L( ctypep_r, cha, chp, op ) \
	PACKM_DIAG_1E_BODY( ctypep_r, cha, chp, k+1, cdim, inca_l2, lda_l2, op )

#define PACKM_DIAG_BODY_1E_U( ctypep_r, cha, chp, op ) \
	PACKM_DIAG_1E_BODY( ctypep_r, cha, chp, 0, k, inca_u2, lda_u2, op )


#define PACKM_DIAG_1R_BODY( ctypep_r, cha, chp, mn_min, mn_max, inca2_lu, lda2_lu, op ) \
\
do \
{ \
	/* PACKM_SCAL_1R assumes inca2 and lda2 are the strides to use. */ \
	dim_t inca2 = inca2_lu; \
	dim_t lda2 = lda2_lu; \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < cdim_bcast; d++ ) \
		PACKM_SCAL_1R( ctypep_r, cha, chp, mn, k, op ); \
} while(0)


#define PACKM_DIAG_BODY_1R_L( ctypep_r, cha, chp, op ) \
	PACKM_DIAG_1R_BODY( ctypep_r, cha, chp, k+1, cdim, inca_l2, lda_l2, op )

#define PACKM_DIAG_BODY_1R_U( ctypep_r, cha, chp, op ) \
	PACKM_DIAG_1R_BODY( ctypep_r, cha, chp, 0, k, inca_u2, lda_u2, op )


#undef  GENTFUNC2R
#define GENTFUNC2R( ctypea, ctypea_r, cha, cha_r, ctypep, ctypep_r, chp, chp_r, opname, arch, suf ) \
\
void PASTEMAC(cha,chp,opname,arch,suf) \
     ( \
             struc_t struca, \
             diag_t  diaga, \
             uplo_t  uploa, \
             conj_t  conja, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   cdim, \
             dim_t   cdim_max, \
             dim_t   cdim_bcast, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx \
     ) \
{ \
	const inc_t inca2 = 2 * inca; \
	const inc_t lda2  = 2 * lda; \
	const inc_t ldp2  = 2 * ldp; \
\
	      ctypep_r           one     = *PASTEMAC(chp_r,1); \
	      ctypep_r           zero    = *PASTEMAC(chp_r,0); \
	const ctypea_r* restrict alpha1  = ( const ctypea_r* )a; \
\
	if ( bli_is_1e_packed( schema ) ) \
	{ \
		/* start by zeroing out the whole block */ \
		PASTEMAC(chp_r,set0s_mxn) \
		( \
		  2*cdim_max, \
		  2*n_max, \
		  ( ctypep_r* )p, 1, ldp  \
		); \
\
		ctypep_r* restrict pi1_ri   = ( ctypep_r* )p; \
		ctypep_r* restrict pi1_ir   = ( ctypep_r* )p + ldp; \
\
		/* write the strictly lower part if it exists */ \
		if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
		{ \
			dim_t    inca_l2 = inca2; \
			dim_t    lda_l2  = lda2; \
			conj_t   conja_l = conja; \
			ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
			ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
			if ( bli_is_upper( uploa ) ) \
			{ \
				bli_swap_incs( &inca_l2, &lda_l2 ); \
\
				if ( bli_is_hermitian( struca ) || bli_is_skew_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_l ); \
\
				if ( bli_is_skew_symmetric( struca ) || bli_is_skew_hermitian( struca ) ) \
				    PASTEMAC(chp,negris)( kappa_r, kappa_i ); \
			} \
\
			if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_1E_L( ctypep_r, cha, chp, scal2jris ); \
			else                          PACKM_DIAG_BODY_1E_L( ctypep_r, cha, chp, scal2ris ); \
		} \
\
		/* write the strictly upper part if it exists */ \
		/* assume either symmetric, hermitian, or triangular */ \
		if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
		{ \
			dim_t   inca_u2 = inca2; \
			dim_t   lda_u2  = lda2; \
			conj_t  conja_u = conja; \
			ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
			ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
			if ( bli_is_lower( uploa ) ) \
			{ \
				bli_swap_incs( &inca_u2, &lda_u2 ); \
\
				if ( bli_is_hermitian( struca ) || bli_is_skew_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_u ); \
\
				if ( bli_is_skew_symmetric( struca ) || bli_is_skew_hermitian( struca ) ) \
				    PASTEMAC(chp,negris)( kappa_r, kappa_i ); \
			} \
\
			if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_1E_U( ctypep_r, cha, chp, scal2jris ); \
			else                          PACKM_DIAG_BODY_1E_U( ctypep_r, cha, chp, scal2ris ); \
		} \
\
		ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
		ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
		/* write the diagonal */ \
		if ( bli_is_unit_diag( diaga ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SET_1E( chp_r, kappa_r, kappa_i, mnk ); \
		} \
		else if ( bli_is_hermitian( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			{ \
				ctypep_r alpha_r; \
				PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2)), alpha_r ); \
				PASTEMAC(chp_r,scal2s)(  kappa_r, alpha_r, *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
				PASTEMAC(chp_r,scal2s)(  kappa_i, alpha_r, *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
				PASTEMAC(chp_r,scal2s)( -kappa_i, alpha_r, *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
				PASTEMAC(chp_r,scal2s)(  kappa_r, alpha_r, *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
			} \
		} \
		else if ( bli_is_skew_hermitian( struca ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				{ \
					ctypep_r alpha_i; \
					PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2) + 1), alpha_i ); \
					PASTEMAC(chp_r,scal2s)(  kappa_i, alpha_i, *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)( -kappa_r, alpha_i, *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)(  kappa_r, alpha_i, *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)(  kappa_i, alpha_i, *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
				} \
			} \
			else \
			{ \
				for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				{ \
					ctypep_r alpha_i; \
					PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2) + 1), alpha_i ); \
					PASTEMAC(chp_r,scal2s)( -kappa_i, alpha_i, *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)(  kappa_r, alpha_i, *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)( -kappa_r, alpha_i, *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
					PASTEMAC(chp_r,scal2s)( -kappa_i, alpha_i, *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
				} \
			} \
		} \
		else if ( bli_is_skew_symmetric( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SET0_1E( chp_r, mnk ); \
		} \
		else if ( bli_is_conj( conja )) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SCAL_1E( ctypep_r, cha, chp, mnk, mnk, scal2jris ); \
		} \
		else \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SCAL_1E( ctypep_r, cha, chp, mnk, mnk, scal2ris ); \
		} \
\
		/* invert the diagonal if requested */ \
		if ( invdiag ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			{ \
				PASTEMAC(chp,invertris)( *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2), \
				                         *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2) ); \
				PASTEMAC(chp,copyjris)( *(pi1_ri + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2), \
				                        *(pi1_ri + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2), \
				                        *(pi1_ir + (mnk*2 + 1)*cdim_bcast + d + mnk*ldp2), \
				                        *(pi1_ir + (mnk*2 + 0)*cdim_bcast + d + mnk*ldp2) ); \
			} \
		} \
\
		/* if this an edge case in both directions, extend the diagonal with ones */ \
		for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PACKM_SET_1E( chp_r, one, zero, mnk ); \
	} \
	else /* bli_is_1r_packed( schema ) */ \
	{ \
		/* start by zeroing out the whole block */ \
		PASTEMAC(chp_r,set0s_mxn) \
		( \
		  cdim_max, \
		  2*n_max, \
		  ( ctypep_r* )p, 1, ldp  \
		); \
\
		ctypep_r* restrict pi1_r    = ( ctypep_r* )p; \
		ctypep_r* restrict pi1_i    = ( ctypep_r* )p + ldp; \
\
		/* write the strictly lower part if it exists */ \
		if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
		{ \
			dim_t    inca_l2 = inca2; \
			dim_t    lda_l2  = lda2; \
			conj_t   conja_l = conja; \
			ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
			ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
			if ( bli_is_upper( uploa ) ) \
			{ \
				bli_swap_incs( &inca_l2, &lda_l2 ); \
\
				if ( bli_is_hermitian( struca ) || bli_is_skew_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_l ); \
\
				if ( bli_is_skew_symmetric( struca ) || bli_is_skew_hermitian( struca ) ) \
				    PASTEMAC(chp,negris)( kappa_r, kappa_i ); \
			} \
\
			if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_1R_L( ctypep_r, cha, chp, scal2jris ); \
			else                          PACKM_DIAG_BODY_1R_L( ctypep_r, cha, chp, scal2ris ); \
		} \
\
		/* write the strictly upper part if it exists */ \
		/* assume either symmetric, hermitian, or triangular */ \
		if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
		{ \
			dim_t    inca_u2 = inca2; \
			dim_t    lda_u2  = lda2; \
			conj_t   conja_u = conja; \
			ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
			ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
			if ( bli_is_lower( uploa ) ) \
			{ \
				bli_swap_incs( &inca_u2, &lda_u2 ); \
\
				if ( bli_is_hermitian( struca ) || bli_is_skew_hermitian( struca ) ) \
				    bli_toggle_conj( &conja_u ); \
\
				if ( bli_is_skew_symmetric( struca ) || bli_is_skew_hermitian( struca ) ) \
				    PASTEMAC(chp,negris)( kappa_r, kappa_i ); \
			} \
\
			if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_1R_U( ctypep_r, cha, chp, scal2jris ); \
			else                          PACKM_DIAG_BODY_1R_U( ctypep_r, cha, chp, scal2ris ); \
		} \
\
		ctypep_r kappa_r = ( ( ctypep_r* )kappa )[0]; \
		ctypep_r kappa_i = ( ( ctypep_r* )kappa )[1]; \
\
		/* write the diagonal */ \
		if ( bli_is_unit_diag( diaga ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SET_1R( chp_r, kappa_r, kappa_i, mnk ); \
		} \
		else if ( bli_is_hermitian( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			{ \
				ctypep_r alpha_r; \
				PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2)), alpha_r ); \
				PASTEMAC(chp_r,scal2s)( kappa_r, alpha_r, *(pi1_r + mnk*(cdim_bcast + ldp2) + d) ); \
				PASTEMAC(chp_r,scal2s)( kappa_i, alpha_r, *(pi1_i + mnk*(cdim_bcast + ldp2) + d) ); \
			} \
		} \
		else if ( bli_is_skew_hermitian( struca ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				{ \
					ctypep_r alpha_i; \
					PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2) + 1), alpha_i ); \
					PASTEMAC(chp_r,scal2s)(  kappa_i, alpha_i, *(pi1_r + mnk*(cdim_bcast + ldp2) + d) ); \
					PASTEMAC(chp_r,scal2s)( -kappa_r, alpha_i, *(pi1_i + mnk*(cdim_bcast + ldp2) + d) ); \
				} \
			} \
			else \
			{ \
				for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				{ \
					ctypep_r alpha_i; \
					PASTEMAC(cha_r,chp_r,copys)( *(alpha1 + mnk*(inca2 + lda2) + 1), alpha_i ); \
					PASTEMAC(chp_r,scal2s)( -kappa_i, alpha_i, *(pi1_r + mnk*(cdim_bcast + ldp2) + d) ); \
					PASTEMAC(chp_r,scal2s)(  kappa_r, alpha_i, *(pi1_i + mnk*(cdim_bcast + ldp2) + d) ); \
				} \
			} \
		} \
		else if ( bli_is_skew_symmetric( struca ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SET0_1R( chp_r, mnk ); \
		} \
		else if ( bli_is_conj( conja ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SCAL_1R( ctypep_r, cha, chp, mnk, mnk, scal2jris ); \
		} \
		else \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PACKM_SCAL_1R( ctypep_r, cha, chp, mnk, mnk, scal2ris ); \
		} \
\
		/* invert the diagonal if requested */ \
		if ( invdiag ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PASTEMAC(chp,invertris)( *(pi1_r + mnk*(cdim_bcast + ldp2) + d), \
				                         *(pi1_i + mnk*(cdim_bcast + ldp2) + d) ); \
		} \
\
		/* if this an edge case in both directions, extend the diagonal with ones */ \
		for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PACKM_SET_1R( chp_r, one, zero, mnk ); \
	} \
}

GENTFUNC2R( scomplex, float,  c, s, scomplex, float,  c, s, packm_diag_1er, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( scomplex, float,  c, s, dcomplex, double, z, d, packm_diag_1er, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, scomplex, float,  c, s, packm_diag_1er, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, dcomplex, double, z, d, packm_diag_1er, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
