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

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, cxk_kername, cxc_kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t strucc, \
       diag_t  diagc, \
       uplo_t  uploc, \
       conj_t  conjc, \
       pack_t  schema, \
       bool    invdiag, \
       dim_t   panel_dim, \
       dim_t   panel_len, \
       dim_t   panel_dim_max, \
       dim_t   panel_len_max, \
       dim_t   panel_dim_off, \
       dim_t   panel_len_off, \
       ctype*  kappa, \
       ctype*  c, inc_t incc, inc_t ldc, \
       ctype*  p,             inc_t ldp, \
                  inc_t is_p, \
       void*   params, \
       cntx_t* cntx  \
     ) \
{ \
	num_t   dt            = PASTEMAC(ch,type); \
	num_t   dt_r          = PASTEMAC(chr,type); \
	dim_t   panel_len_pad = panel_len_max - panel_len; \
\
	bszid_t bsz_id        = bli_is_col_packed( schema ) ? BLIS_NR : BLIS_MR; \
	dim_t   packmrnr      = bli_cntx_get_blksz_max_dt( dt, bsz_id, cntx ); \
	dim_t   packmrnr_r    = bli_cntx_get_blksz_max_dt( dt_r, bsz_id, cntx ); \
\
	ukr_t   cxk_ker_id    = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXK_KER \
	                                                    : BLIS_PACKM_MRXK_KER; \
	ukr_t   cxc_ker_id    = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXNR_DIAG_KER \
	                                                    : BLIS_PACKM_MRXMR_DIAG_KER; \
\
	if ( bli_is_1m_packed( schema ) ) \
	{ \
		cxk_ker_id = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXK_1ER_KER \
		                                         : BLIS_PACKM_MRXK_1ER_KER; \
		cxc_ker_id = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXNR_DIAG_1ER_KER \
		                                         : BLIS_PACKM_MRXMR_DIAG_1ER_KER; \
	} \
\
	PASTECH(cxk_kername,_ker_ft) f_cxk = bli_cntx_get_ukr_dt( dt, cxk_ker_id, cntx ); \
	PASTECH(cxc_kername,_ker_ft) f_cxc = bli_cntx_get_ukr_dt( dt, cxc_ker_id, cntx ); \
\
	/* For general matrices, pack and return early */ \
	if ( bli_is_general( strucc ) ) \
	{ \
		f_cxk \
		( \
		  conjc, \
		  schema, \
		  panel_dim, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		  cntx  \
		); \
		return; \
	} \
\
	/* Sanity check. Diagonals should not intersect the short end of
	   a micro-panel. If they do, then somehow the constraints on
	   cache blocksizes being a whole multiple of the register
	   blocksizes was somehow violated. */ \
	doff_t diagoffc = panel_dim_off - panel_len_off; \
	if ( (          -panel_dim < diagoffc && diagoffc <         0 ) || \
		 ( panel_len-panel_dim < diagoffc && diagoffc < panel_len ) ) \
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
	/* For triangular, symmetric, and hermitian matrices we need to consider
	   three parts. */ \
\
	/* Pack to p10. */ \
	if ( 0 < diagoffc ) \
	{ \
		dim_t  p10_dim     = panel_dim; \
		dim_t  p10_len     = bli_min( diagoffc, panel_len ); \
		dim_t  p10_len_max = p10_len == panel_len ? panel_len_max : p10_len; \
		ctype* p10         = p; \
		conj_t conjc10     = conjc; \
		ctype* c10         = c; \
		inc_t  incc10      = incc; \
		inc_t  ldc10       = ldc; \
\
		if ( bli_is_upper( uploc ) ) \
		{ \
			bli_reflect_to_stored_part( diagoffc, c10, incc10, ldc10 ); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc10 ); \
		} \
\
		/* If we are referencing the unstored part of a triangular matrix,
		   explicitly store zeros */ \
		if ( bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) \
		{ \
			if ( bli_is_1m_packed( schema ) ) \
			{ \
				ctype_r* restrict zero = PASTEMAC(chr,0); \
\
				PASTEMAC2(chr,setm,BLIS_TAPI_EX_SUF) \
				( \
				  BLIS_NO_CONJUGATE, \
				  0, \
				  BLIS_NONUNIT_DIAG, \
				  BLIS_DENSE, \
				  packmrnr_r, \
				  p10_len_max * 2, \
				  zero, \
				  ( ctype_r* )p10, 1, ldp, \
				  cntx, \
				  NULL  \
				); \
			} \
			else \
			{ \
				ctype* restrict zero = PASTEMAC(ch,0); \
\
				PASTEMAC2(ch,setm,BLIS_TAPI_EX_SUF) \
				( \
				  BLIS_NO_CONJUGATE, \
				  0, \
				  BLIS_NONUNIT_DIAG, \
				  BLIS_DENSE, \
				  packmrnr, \
				  p10_len_max, \
				  zero, \
				  p10, 1, ldp, \
				  cntx, \
				  NULL  \
				); \
			} \
		} \
		else \
		{ \
			f_cxk \
			( \
			  conjc10, \
			  schema, \
			  p10_dim, \
			  p10_len, \
			  p10_len_max, \
			  kappa, \
			  c10, incc10, ldc10, \
			  p10,         ldp, \
			  cntx  \
			); \
		} \
	} \
\
	/* Pack to p11. */ \
	if ( 0 <= diagoffc && diagoffc + panel_dim <= panel_len ) \
	{ \
		dim_t  i           = diagoffc; \
		dim_t  p11_dim     = panel_dim; \
		dim_t  p11_len_max = panel_dim + ( diagoffc + panel_dim == panel_len \
		                                   ? panel_len_pad : 0 ); \
		ctype* p11         = p + i * ldp; \
		conj_t conjc11     = conjc; \
		ctype* c11         = c + i * ldc; \
		inc_t  incc11      = incc; \
		inc_t  ldc11       = ldc; \
\
		f_cxc \
		( \
		  strucc, \
		  diagc, \
		  uploc, \
		  conjc11, \
		  schema, \
		  invdiag, \
		  p11_dim, \
		  p11_len_max, \
		  kappa, \
		  c11, incc11, ldc11, \
		  p11,         ldp, \
		  cntx  \
		); \
	} \
\
	/* Pack to p12. */ \
	if ( diagoffc + panel_dim < panel_len ) \
	{ \
		dim_t  i           = bli_max( 0, diagoffc + panel_dim ); \
		dim_t  p12_dim     = panel_dim; \
		dim_t  p12_len     = panel_len - i; \
		/* If we are packing p12, then it is always the last partial block \
		   and so we should make sure to pad with zeros if necessary. */ \
		dim_t  p12_len_max = p12_len + panel_len_pad; \
		ctype* p12         = p + i * ldp; \
		conj_t conjc12     = conjc; \
		ctype* c12         = c + i * ldc; \
		inc_t  incc12      = incc; \
		inc_t  ldc12       = ldc; \
\
		if ( bli_is_lower( uploc ) ) \
		{ \
			bli_reflect_to_stored_part( diagoffc - i, c12, incc12, ldc12 ); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc12 ); \
		} \
\
		/* If we are referencing the unstored part of a triangular matrix,
		   explicitly store zeros */ \
		if ( bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) \
		{ \
			if ( bli_is_1m_packed( schema ) ) \
			{ \
			    ctype_r* restrict zero = PASTEMAC(chr,0); \
\
				PASTEMAC2(chr,setm,BLIS_TAPI_EX_SUF) \
				( \
				  BLIS_NO_CONJUGATE, \
				  0, \
				  BLIS_NONUNIT_DIAG, \
				  BLIS_DENSE, \
				  packmrnr_r, \
				  p12_len_max * 2, \
				  zero, \
				  ( ctype_r* )p12, 1, ldp, \
				  cntx, \
				  NULL  \
				); \
			} \
			else \
			{ \
				ctype* restrict zero = PASTEMAC(ch,0); \
\
				PASTEMAC2(ch,setm,BLIS_TAPI_EX_SUF) \
				( \
				  BLIS_NO_CONJUGATE, \
				  0, \
				  BLIS_NONUNIT_DIAG, \
				  BLIS_DENSE, \
				  packmrnr, \
				  p12_len_max, \
				  zero, \
				  p12, 1, ldp, \
				  cntx, \
				  NULL  \
				); \
			} \
		} \
		else \
		{ \
			f_cxk \
			( \
			  conjc12, \
			  schema, \
			  p12_dim, \
			  p12_len, \
			  p12_len_max, \
			  kappa, \
			  c12, incc12, ldc12, \
			  p12,         ldp, \
			  cntx  \
			); \
		} \
	} \
}

INSERT_GENTFUNCR_BASIC2( packm_struc_cxk, packm_cxk, packm_cxc_diag )

