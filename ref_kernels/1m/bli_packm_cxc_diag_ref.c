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


#define PACKM_DIAG_BODY( ctype, ch, mn_min, mn_max, inca, lda, op ) \
\
do \
{ \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < dfac; d++ ) \
		PASTEMAC(ch,op)( kappa_cast, *(alpha1 + mn*inca + k*lda), *(pi1 + mn*dfac + d + k*ldp) ); \
} while(0)


#define PACKM_DIAG_BODY_L( ctype, ch, op ) \
	PACKM_DIAG_BODY( ctype, ch, k+1, cdim, inca_l, lda_l, op )

#define PACKM_DIAG_BODY_U( ctype, ch, op ) \
	PACKM_DIAG_BODY( ctype, ch, 0, k, inca_u, lda_u, op )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr0, bb0, arch, suf ) \
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
       const cntx_t* cntx  \
     ) \
{ \
	const num_t dt        = PASTEMAC(ch,type); \
	const dim_t cdim_max  = bli_cntx_get_blksz_def_dt( dt, mnr0, cntx ); \
	const dim_t cdim_pack = bli_cntx_get_blksz_max_dt( dt, mnr0, cntx ); \
	const dim_t dfac      = bli_cntx_get_blksz_def_dt( dt, bb0, cntx ); \
\
	/* start by zeroing out the whole block */ \
	PASTEMAC(ch,set0s_mxn) \
	( \
	  cdim_pack, \
	  n_max, \
	  p, 1, ldp  \
	); \
\
	      ctype           kappa_cast = *( ctype* )kappa; \
	const ctype* restrict alpha1     = a; \
	      ctype* restrict pi1        = p; \
\
	/* write the strictly lower part if it exists */ \
	if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_l  = inca; \
		dim_t  lda_l   = lda; \
		conj_t conja_l = conja; \
\
		if ( bli_is_upper( uploa ) ) \
		{ \
			bli_swap_incs( &inca_l, &lda_l ); \
			if ( bli_is_hermitian( struca ) ) \
				bli_toggle_conj( &conja_l ); \
		} \
\
		if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_L( ctype, ch, scal2js ); \
		else                          PACKM_DIAG_BODY_L( ctype, ch, scal2s ); \
	} \
\
	/* write the strictly upper part if it exists */ \
	/* assume either symmetric, hermitian, or triangular */ \
	if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_u  = inca; \
		dim_t  lda_u   = lda; \
		conj_t conja_u = conja; \
\
		if ( bli_is_lower( uploa ) ) \
		{ \
			bli_swap_incs( &inca_u, &lda_u ); \
			if ( bli_is_hermitian( struca ) ) \
				bli_toggle_conj( &conja_u ); \
		} \
\
		if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_U( ctype, ch, scal2js ); \
		else                          PACKM_DIAG_BODY_U( ctype, ch, scal2s ); \
	} \
\
	/* write the diagonal */ \
	if ( bli_is_unit_diag( diaga ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PASTEMAC(ch,set1s)( *(pi1 + mnk*(dfac + ldp) + d) ); \
	} \
	else if ( bli_is_hermitian( struca ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
		{ \
			ctype mu; \
			PASTEMAC(ch,copys)( *(alpha1 + mnk*(inca + lda)), mu ); \
			PASTEMAC(ch,seti0s)( mu ); \
			PASTEMAC(ch,scal2s)( kappa_cast, mu, *(pi1 + mnk*(dfac + ldp) + d) ); \
		} \
	} \
	else if ( bli_is_conj( conja )) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PASTEMAC(ch,scal2js)( kappa_cast, *(alpha1 + mnk*(inca + lda)), *(pi1 + mnk*(dfac + ldp) + d) ); \
	} \
	else \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PASTEMAC(ch,scal2s)( kappa_cast, *(alpha1 + mnk*(inca + lda)), *(pi1 + mnk*(dfac + ldp) + d) ); \
	} \
\
	/* invert the diagonal if requested */ \
	if ( invdiag ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < dfac; ++d ) \
			PASTEMAC(ch,inverts)( *(pi1 + mnk*(dfac + ldp) + d) ); \
	} \
\
	/* if this an edge case in both directions, extend the diagonal with ones */ \
	for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
	for ( dim_t d = 0; d < dfac; ++d ) \
		PASTEMAC(ch,set1s)( *(pi1 + mnk*(dfac + ldp) + d) ); \
}

INSERT_GENTFUNC_BASIC4( packm_mrxmr_diag, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC_BASIC4( packm_nrxnr_diag, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

