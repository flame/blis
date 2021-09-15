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
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	/* Handle micro-panel packing based on the structure of the matrix
	   being packed. */ \
	if      ( bli_is_general( strucc ) ) \
	{ \
		/* For micro-panels of general matrices, we can call the pack
		   kernel front-end directly. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p, is_p, ldp, \
		  cntx  \
		); \
	} \
	else if ( bli_is_herm_or_symm( strucc ) ) \
	{ \
		/* Call a helper function for micro-panels of Hermitian/symmetric
		   matrices. */ \
		PASTEMAC(ch,packm_herm_cxk_4mi) \
		( \
          strucc, \
          diagc, \
          uploc, \
          conjc, \
          schema, \
          invdiag, \
          panel_dim, \
          panel_len, \
          panel_dim_max, \
          panel_len_max, \
          panel_dim_off, \
          panel_len_off, \
          kappa, \
          c, incc, ldc, \
          p,       ldp, \
             is_p, \
		  cntx, \
          params \
		); \
	} \
	else /* ( bli_is_triangular( strucc ) ) */ \
	{ \
		/* Call a helper function for micro-panels of triangular
		   matrices. */ \
		PASTEMAC(ch,packm_tri_cxk_4mi) \
		( \
          strucc, \
          diagc, \
          uploc, \
          conjc, \
          schema, \
          invdiag, \
          panel_dim, \
          panel_len, \
          panel_dim_max, \
          panel_len_max, \
          panel_dim_off, \
          panel_len_off, \
          kappa, \
          c, incc, ldc, \
          p,       ldp, \
             is_p, \
		  cntx, \
          params \
		); \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_struc_cxk_4mi, packm_cxk_4mi )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	doff_t diagoffc = panel_dim_off - panel_len_off; \
	doff_t diagoffc_abs; \
	dim_t  i, j; \
\
\
	/* Handle the case where the micro-panel does NOT intersect the
	   diagonal separately from the case where it does intersect. */ \
	if ( !bli_intersects_diag_n( diagoffc, panel_dim, panel_len ) ) \
	{ \
		/* If the current panel is unstored, we need to make a few
		   adjustments so we refer to the data where it is actually
		   stored, also taking conjugation into account. (Note this
		   implicitly assumes we are operating on a dense panel
		   within a larger symmetric or Hermitian matrix, since a
		   general matrix would not contain any unstored region.) */ \
		if ( bli_is_unstored_subpart_n( diagoffc, uploc, panel_dim, panel_len ) ) \
		{ \
			c = c + diagoffc * ( doff_t )ldc + \
			       -diagoffc * ( doff_t )incc;  \
			bli_swap_incs( &incc, &ldc ); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc ); \
		} \
\
		/* Pack the full panel. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p, is_p, ldp, \
		  cntx  \
		); \
	} \
	else /* if ( bli_intersects_diag_n( diagoffc, panel_dim, panel_len ) ) */ \
	{ \
		ctype_r* restrict p_r         = ( ctype_r* )p; \
\
		ctype_r* restrict one_r       = PASTEMAC(chr,1); \
		ctype_r* restrict minus_one_r = PASTEMAC(chr,m1); \
\
		ctype*   restrict c10; \
		ctype_r* restrict p10; \
		dim_t             p10_dim, p10_len; \
		inc_t             incc10, ldc10; \
		doff_t            diagoffc10; \
		conj_t            conjc10; \
\
		ctype*   restrict c12; \
		ctype_r* restrict p12; \
		dim_t             p12_dim, p12_len; \
		inc_t             incc12, ldc12; \
		doff_t            diagoffc12; \
		conj_t            conjc12; \
\
		/* Sanity check. Diagonals should not intersect the short end of
		   a micro-panel. If they do, then somehow the constraints on
		   cache blocksizes being a whole multiple of the register
		   blocksizes was somehow violated. */ \
		if ( diagoffc < 0 ) \
			bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
		diagoffc_abs = bli_abs( diagoffc ); \
\
		if      ( bli_is_lower( uploc ) ) \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs; \
			p10        = p_r; \
			c10        = c; \
			incc10     = incc; \
			ldc10      = ldc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			diagoffc12 = diagoffc_abs - j; \
			p12        = p_r + (j  )*ldp; \
			c12        = c   + (j  )*ldc; \
			c12        = c12 + diagoffc12 * ( doff_t )ldc + \
			                  -diagoffc12 * ( doff_t )incc;  \
			incc12     = ldc; \
			ldc12      = incc; \
			conjc12    = conjc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc12 ); \
		} \
		else /* if ( bli_is_upper( uploc ) ) */ \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs + panel_dim; \
			diagoffc10 = diagoffc; \
			p10        = p_r; \
			c10        = c; \
			c10        = c10 + diagoffc10 * ( doff_t )ldc + \
			                  -diagoffc10 * ( doff_t )incc;  \
			incc10     = ldc; \
			ldc10      = incc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			p12        = p_r + (j  )*ldp; \
			c12        = c   + (j  )*ldc; \
			incc12     = incc; \
			ldc12      = ldc; \
			conjc12    = conjc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc10 ); \
		} \
\
		/* Pack to p10. For upper storage, this includes the unstored
		   triangle of c11. */ \
		/* NOTE: Since we're only packing partial panels here, we pass in
		   p1x_len as panel_len_max; otherwise, the packm kernel will zero-
		   fill the columns up to panel_len_max, which is not what we need
		   or want to happen. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc10, \
		  p10_dim, \
		  panel_dim_max, \
		  p10_len, \
		  p10_len, \
		  kappa, \
		            c10, incc10, ldc10, \
		  ( ctype* )p10, is_p,   ldp, \
		  cntx  \
		); \
\
		/* Pack to p12. For lower storage, this includes the unstored
		   triangle of c11. */ \
		/* NOTE: Since we're only packing partial panels here, we pass in
		   p1x_len as panel_len_max; otherwise, the packm kernel will zero-
		   fill the columns up to panel_len_max, which is not what we need
		   or want to happen. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc12, \
		  p12_dim, \
		  panel_dim_max, \
		  p12_len, \
		  p12_len, \
		  kappa, \
		            c12, incc12, ldc12, \
		  ( ctype* )p12, is_p,   ldp, \
		  cntx  \
		); \
\
		/* Pack the stored triangle of c11 to p11. */ \
		{ \
			dim_t    p11_m   = panel_dim; \
			dim_t    p11_n   = panel_dim; \
			inc_t    incc11  = 2*incc; \
			inc_t    ldc11  = 2*ldc; \
			dim_t    j2      = diagoffc_abs; \
			ctype*   c11     = ( ctype*   )c   + (j2 )*ldc; \
			ctype_r* p11     = ( ctype_r* )p_r + (j2 )*ldp; \
			ctype_r* c11_r   = ( ctype_r* )c11; \
			ctype_r* c11_i   = ( ctype_r* )c11 + 1; \
			ctype_r* p11_r   = ( ctype_r* )p11; \
			ctype_r* p11_i   = ( ctype_r* )p11 + is_p; \
			ctype_r* alpha_r = one_r; \
			ctype_r* alpha_i = ( bli_is_conj( conjc ) ? minus_one_r : one_r ); \
			ctype_r  kappa_r = PASTEMAC(ch,real)( *kappa ); \
			ctype_r  kappa_i = PASTEMAC(ch,imag)( *kappa ); \
\
			/* Copy the real part of the stored triangle of c11 to p11_r. */ \
			PASTEMAC2(chr,scal2m,BLIS_TAPI_EX_SUF) \
			( \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  uploc, \
			  BLIS_NO_TRANSPOSE, \
			  p11_m, \
			  p11_n, \
			  alpha_r, \
			  c11_r, incc11, ldc11, \
			  p11_r, 1,      ldp, \
			  cntx, \
			  NULL  \
			); \
\
			/* Copy the imaginary part of the stored triangle of c11 to p11_i,
			   scaling by -1 if conjugation on c was requested. */ \
			PASTEMAC2(chr,scal2m,BLIS_TAPI_EX_SUF) \
			( \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  uploc, \
			  BLIS_NO_TRANSPOSE, \
			  p11_m, \
			  p11_n, \
			  alpha_i, \
			  c11_i, incc11, ldc11, \
			  p11_i, 1,      ldp, \
			  cntx, \
			  NULL  \
			); \
\
			/* If source matrix c is Hermitian, we have to zero out the
			   imaginary components of the diagonal of p11 in case the
			   corresponding elements in c11 were not already zero. */ \
			if ( bli_is_hermitian( strucc ) ) \
			{ \
				for ( i = 0; i < p11_m; ++i ) \
				{ \
					ctype_r* pi11_i = p11_i + (i  ) + (i  )*ldp; \
\
					PASTEMAC(chr,set0s)( *pi11_i ); \
				} \
			} \
\
			/* Apply kappa to the part of p11 that corresponds to the stored
			   part of c11 that was copied above. */ \
			if ( bli_is_upper( uploc ) ) \
			{ \
				PASTEMAC(ch,scalris_mxn_u) \
				( \
				  0, \
				  p11_m, \
				  p11_n, \
				  &kappa_r, \
				  &kappa_i, \
				  p11_r,  \
				  p11_i, 1, ldp  \
				); \
			} \
			else \
			{ \
				PASTEMAC(ch,scalris_mxn_l) \
				( \
				  0, \
				  p11_m, \
				  p11_n, \
				  &kappa_r, \
				  &kappa_i, \
				  p11_r,  \
				  p11_i, 1, ldp  \
				); \
			} \
/*
	PASTEMAC(chr,fprintm)( stdout, "packm_herm_cxk: ap_r copied", panel_dim_max, panel_len_max, \
	                       p_r + 0*is_p, rs_p, ldp, "%4.1f", "" ); \
	PASTEMAC(chr,fprintm)( stdout, "packm_herm_cxk: ap_i copied", panel_dim_max, panel_len_max, \
	                       p_r + 1*is_p, rs_p, ldp, "%4.1f", "" ); \
*/ \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_herm_cxk_4mi, packm_cxk_4mi )





#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	doff_t diagoffc = panel_dim_off - panel_len_off; \
\
	/* Pack the panel. */ \
	PASTEMAC(ch,kername) \
	( \
	  conjc, \
	  panel_dim, \
	  panel_dim_max, \
	  panel_len, \
	  panel_len_max, \
	  kappa, \
	  c, incc, ldc, \
	  p, is_p, ldp, \
	  cntx  \
	); \
\
\
	/* Tweak the panel according to its triangular structure */ \
	{ \
		ctype_r* p_r   = ( ctype_r* )p; \
		ctype_r* p_i   = ( ctype_r* )p + is_p; \
\
		dim_t    j     = bli_abs( diagoffc ); \
		ctype_r* p11_r = p_r + (j  )*ldp; \
		ctype_r* p11_i = p_i + (j  )*ldp; \
\
		/* If the diagonal of c is implicitly unit, explicitly set the
		   the diagonal of the packed panel to kappa. */ \
		if ( bli_is_unit_diag( diagc ) ) \
		{ \
			ctype_r kappa_r = PASTEMAC(ch,real)( *kappa ); \
			ctype_r kappa_i = PASTEMAC(ch,imag)( *kappa ); \
\
			PASTEMAC2(chr,setd,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  diagoffc, \
			  panel_dim, \
			  panel_len, \
			  &kappa_r, \
			  p_r, 1, ldp, \
			  cntx, \
			  NULL  \
			); \
			PASTEMAC2(chr,setd,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  diagoffc, \
			  panel_dim, \
			  panel_len, \
			  &kappa_i, \
			  p_i, 1, ldp, \
			  cntx, \
			  NULL  \
			); \
		} \
\
\
		/* If requested, invert the diagonal of the packed panel. */ \
		if ( invdiag == TRUE ) \
		{ \
			dim_t i; \
\
			for ( i = 0; i < panel_dim; ++i ) \
			{ \
				ctype_r* pi11_r = p11_r + (i  ) + (i  )*ldp; \
				ctype_r* pi11_i = p11_i + (i  ) + (i  )*ldp; \
\
				PASTEMAC(ch,invertris)( *pi11_r, *pi11_i ); \
			} \
		} \
\
\
		/* Set the region opposite the diagonal of p to zero. To do this,
		   we need to reference the "unstored" region on the other side of
		   the diagonal. This amounts to toggling uploc and then shifting
		   the diagonal offset to shrink the newly referenced region (by
		   one diagonal). Note that this zero-filling is not needed for
		   trsm, since the unstored region is not referenced by the trsm
		   micro-kernel; however, zero-filling is needed for trmm, which
		   uses the gemm micro-kernel.*/ \
		{ \
			ctype_r* restrict zero_r = PASTEMAC(chr,0); \
			uplo_t            uplop  = uploc; \
\
			bli_toggle_uplo( &uplop ); \
			bli_shift_diag_offset_to_shrink_uplo( uplop, &diagoffc ); \
\
			PASTEMAC2(chr,setm,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  diagoffc, \
			  BLIS_NONUNIT_DIAG, \
			  uplop, \
			  panel_dim, \
			  panel_len, \
			  zero_r, \
			  p_r, 1, ldp, \
			  cntx, \
			  NULL  \
			); \
			PASTEMAC2(chr,setm,BLIS_TAPI_EX_SUF) \
			( \
			  BLIS_NO_CONJUGATE, \
			  diagoffc, \
			  BLIS_NONUNIT_DIAG, \
			  uplop, \
			  panel_dim, \
			  panel_len, \
			  zero_r, \
			  p_i, 1, ldp, \
			  cntx, \
			  NULL  \
			); \
		} \
	} \
\
	/* If this panel is an edge case in both panel dimension and length,
	   then it must be a bottom-right corner case. Set the part of the
	   diagonal that extends into the zero-padded region to identity.
	   NOTE: This is actually only necessary when packing for trsm, as
	   it helps prevent NaNs and Infs from creeping into the computation.
	   However, we set the region to identity for trmm as well. Those
	   1.0's end up getting muliplied by the 0.0's in the zero-padded
	   region of the other matrix, so there is no harm in this. */ \
	if ( panel_dim != panel_dim_max && \
	     panel_len != panel_len_max ) \
	{ \
		ctype_r* restrict one_r  = PASTEMAC(chr,1); \
		ctype_r* restrict zero_r = PASTEMAC(chr,0); \
		dim_t             i      = panel_dim; \
		dim_t             j      = panel_len; \
		dim_t             m_br   = panel_dim_max - i; \
		dim_t             n_br   = panel_len_max - j; \
		ctype_r*          p_br_r = ( ctype_r* )p +        (i  ) + (j  )*ldp; \
		ctype_r*          p_br_i = ( ctype_r* )p + is_p + (i  ) + (j  )*ldp; \
\
		PASTEMAC2(chr,setd,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  0, \
		  m_br, \
		  n_br, \
		  one_r, \
		  p_br_r, 1, ldp, \
		  cntx, \
		  NULL  \
		); \
		PASTEMAC2(chr,setd,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  0, \
		  m_br, \
		  n_br, \
		  zero_r, \
		  p_br_i, 1, ldp, \
		  cntx, \
		  NULL  \
		); \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_tri_cxk_4mi, packm_cxk_4mi )

