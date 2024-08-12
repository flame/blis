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


#define PACKM_DIAG_BODY( ctypea, ctypep, cha, chp, mn_min, mn_max, cdim_bcast, inca, lda, op ) \
\
do \
{ \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	{ \
		ctypep alpha_cast, kappa_alpha; \
		PASTEMAC(cha,chp,copys)( *(alpha1 + mn*inca + k*lda), alpha_cast ); \
		PASTEMAC(chp,op)( kappa_use, alpha_cast, kappa_alpha ); \
		for ( dim_t d = 0; d < cdim_bcast; d++ ) \
			PASTEMAC(chp,copys)( kappa_alpha, *(pi1 + mn*cdim_bcast + d + k*ldp) ); \
	} \
} while(0)


#define PACKM_DIAG_BODY_L( ctypea, ctypep, cha, chp, op ) \
	PACKM_DIAG_BODY( ctypea, ctypep, cha, chp, k+1, cdim, cdim_bcast, inca_l, lda_l, op )

#define PACKM_DIAG_BODY_U( ctypea, ctypep, cha, chp, op ) \
	PACKM_DIAG_BODY( ctypea, ctypep, cha, chp, 0, k, cdim_bcast, inca_u, lda_u, op )


#undef  GENTFUNC2
#define GENTFUNC2( ctypea, ctypep, cha, chp, opname, arch, suf ) \
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
       const cntx_t* cntx  \
     ) \
{ \
	/* start by zeroing out the whole block */ \
	PASTEMAC(chp,set0s_mxn) \
	( \
	  cdim_max, \
	  n_max, \
	  p, 1, ldp  \
	); \
\
	      ctypep           kappa_cast = *( ctypep* )kappa; \
	const ctypea* restrict alpha1     = a; \
	      ctypep* restrict pi1        = p; \
\
	/* write the strictly lower part if it exists */ \
	if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_l  = inca; \
		dim_t  lda_l   = lda; \
		conj_t conja_l = conja; \
		ctypep kappa_use; \
\
		PASTEMAC(chp,copys)( kappa_cast, kappa_use ); \
\
		if ( bli_is_upper( uploa ) ) \
		{ \
			bli_swap_incs( &inca_l, &lda_l ); \
\
			if ( bli_is_hermitian( struca ) || bli_is_skew_hermitian( struca ) ) \
				bli_toggle_conj( &conja_l ); \
\
			if ( bli_is_skew_symmetric( struca ) || bli_is_skew_hermitian( struca ) ) \
				PASTEMAC(chp,neg2s)( kappa_cast, kappa_use ); \
		} \
\
		if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_L( ctypea, ctypep, cha, chp, scal2js ); \
		else                          PACKM_DIAG_BODY_L( ctypea, ctypep, cha, chp, scal2s ); \
	} \
\
	/* write the strictly upper part if it exists */ \
	/* assume either symmetric, hermitian, or triangular */ \
	if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) || bli_is_skew_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_u    = inca; \
		dim_t  lda_u     = lda; \
		conj_t conja_u   = conja; \
		ctypep kappa_use; \
\
		PASTEMAC(chp,copys)( kappa_cast, kappa_use ); \
\
		if ( bli_is_lower( uploa ) ) \
		{ \
			bli_swap_incs( &inca_u, &lda_u ); \
\
			if ( bli_is_hermitian( struca ) || \
			     bli_is_skew_hermitian( struca ) ) \
				bli_toggle_conj( &conja_u ); \
\
			if ( bli_is_skew_symmetric( struca ) || \
			     bli_is_skew_hermitian( struca ) ) \
				PASTEMAC(chp,neg2s)( kappa_cast, kappa_use ); \
		} \
\
		if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_U( ctypea, ctypep, cha, chp, scal2js ); \
		else                          PACKM_DIAG_BODY_U( ctypea, ctypep, cha, chp, scal2s ); \
	} \
\
	/* write the diagonal */ \
	if ( bli_is_unit_diag( diaga ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PASTEMAC(chp,copys)( kappa_cast, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
	} \
	else if ( bli_is_hermitian( struca ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		{ \
			ctypep alpha_cast, kappa_alpha; \
			PASTEMAC(cha,chp,copys)( *(alpha1 + mnk*(inca + lda)), alpha_cast ); \
			PASTEMAC(chp,seti0s)( alpha_cast ); \
			PASTEMAC(chp,scal2s)( kappa_cast, alpha_cast, kappa_alpha ); \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PASTEMAC(chp,copys)( kappa_alpha, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
		} \
	} \
	else if ( bli_is_skew_hermitian( struca ) ) \
	{ \
		if ( bli_is_conj( conja ) ) \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			{ \
				ctypep alpha_cast; \
				PASTEMAC(cha,chp,copys)( *(alpha1 + mnk*(inca + lda)), alpha_cast ); \
				PASTEMAC(chp,setr0s)( alpha_cast ); \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
					PASTEMAC(chp,scal2js)( kappa_cast, alpha_cast, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
			} \
		} \
		else  \
		{ \
			for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
			{ \
				ctypep alpha_cast; \
				PASTEMAC(cha,chp,copys)( *(alpha1 + mnk*(inca + lda)), alpha_cast ); \
				PASTEMAC(chp,setr0s)( alpha_cast ); \
				for ( dim_t d = 0; d < cdim_bcast; ++d ) \
					PASTEMAC(chp,scal2s)( kappa_cast, alpha_cast, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
			} \
		} \
	} \
	else if ( bli_is_skew_symmetric( struca ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PASTEMAC(chp,set0s)( *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
	} \
	else if ( bli_is_conj( conja ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		{ \
			ctypep alpha_cast, kappa_alpha; \
			PASTEMAC(cha,chp,copys)( *(alpha1 + mnk*(inca + lda)), alpha_cast ); \
			PASTEMAC(chp,scal2js)( kappa_cast, alpha_cast, kappa_alpha ); \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PASTEMAC(chp,copys)( kappa_alpha, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
		} \
	} \
	else \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		{ \
			ctypep alpha_cast, kappa_alpha; \
			PASTEMAC(cha,chp,copys)( *(alpha1 + mnk*(inca + lda)), alpha_cast ); \
			PASTEMAC(chp,scal2s)( kappa_cast, alpha_cast, kappa_alpha ); \
			for ( dim_t d = 0; d < cdim_bcast; ++d ) \
				PASTEMAC(chp,copys)( kappa_alpha, *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
		} \
	} \
\
	/* invert the diagonal if requested */ \
	if ( invdiag ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PASTEMAC(chp,inverts)( *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
	} \
\
	/* if this an edge case in both directions, extend the diagonal with ones */ \
	for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
	for ( dim_t d = 0; d < cdim_bcast; ++d ) \
		PASTEMAC(chp,set1s)( *(pi1 + mnk*(cdim_bcast + ldp) + d) ); \
}

INSERT_GENTFUNC2_BASIC( packm_diag, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC2_MIX_P( packm_diag, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

