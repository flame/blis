/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           struc_t         strucc, \
                           doff_t          diagoffc, \
                           uplo_t          uploc, \
                           conj_t          conjc, \
                           dim_t           m_panel, \
                           dim_t           n_panel, \
                           dim_t           m_panel_max, \
                           dim_t           n_panel_max, \
                           ctype* restrict kappa, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           ctype* restrict p, inc_t rs_p, inc_t cs_p  \
                         ) \
{ \
	ctype* restrict zero = PASTEMAC(ch,0); \
\
	dim_t           i, j; \
	dim_t           panel_len; \
	doff_t          diagoffc_abs; \
	dim_t           panel_dim; \
	inc_t           incc, ldc; \
	inc_t           ldp; \
\
	ctype* restrict c10; \
	ctype* restrict p10; \
	dim_t           p10_dim, p10_len; \
	inc_t           incc10, ldc10; \
	doff_t          diagoffc10; \
	conj_t          conjc10; \
\
	ctype* restrict c12; \
	ctype* restrict p12; \
	dim_t           p12_dim, p12_len; \
	inc_t           incc12, ldc12; \
	doff_t          diagoffc12; \
	conj_t          conjc12; \
\
	ctype* restrict c11; \
	ctype* restrict p11; \
	dim_t           p11_m; \
	dim_t           p11_n; \
	inc_t           rs_p11, cs_p11; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_len = m_panel; \
		panel_dim = n_panel; \
		incc      = cs_c; \
		ldc       = rs_c; \
		ldp       = rs_p; \
		rs_p11    = rs_p; \
		cs_p11    = 1; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_len = n_panel; \
		panel_dim = m_panel; \
		incc      = rs_c; \
		ldc       = cs_c; \
		ldp       = cs_p; \
		rs_p11    = 1; \
		cs_p11    = cs_p; \
	} \
\
	if ( !bli_intersects_diag_n( diagoffc, m_panel, n_panel ) ) \
	{ \
		/* If the current panel is unstored, we need to make a few
		   adjustments so we refer to the data where it is actually
		   stored, also taking conjugation into account. (Note this
		   implicitly assumes we are operating on a dense panel
		   within a larger symmetric or Hermitian matrix, since a
		   general matrix would not contain any unstored region.) */ \
		if ( bli_is_unstored_subpart_n( diagoffc, uploc, m_panel, n_panel ) ) \
		{ \
			c = c + diagoffc * ( doff_t )cs_c + \
			       -diagoffc * ( doff_t )rs_c;  \
			bli_swap_incs( incc, ldc ); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( conjc ); \
		} \
\
		/* Pack the full panel. */ \
		PASTEMAC(ch,packm_cxk)( conjc, \
		                        panel_dim, \
		                        panel_len, \
		                        kappa, \
		                        c, incc, ldc, \
		                        p,       ldp ); \
	} \
	else /* if ( bli_intersects_diag_n( diagoffc, m_panel, n_panel ) ) */ \
	{ \
		/* Sanity check. Diagonals should not intersect the short end of
		   a micro-panel. If they do, then somehow the constraints on
		   cache blocksizes being a whole multiple of the register
		   blocksizes was somehow violated. */ \
		if ( ( bli_is_col_stored_f( rs_p, cs_p ) && diagoffc < 0 ) || \
		     ( bli_is_row_stored_f( rs_p, cs_p ) && diagoffc > 0 ) ) \
			bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
		diagoffc_abs = bli_abs( diagoffc ); \
\
		if      ( ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) ) || \
		          ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) ) \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs; \
			p10        = p; \
			c10        = c; \
			incc10     = incc; \
			ldc10      = ldc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			diagoffc12 = diagoffc_abs - j; \
			p12        = p + (j  )*ldp; \
			c12        = c + (j  )*ldc; \
			c12        = c12 + diagoffc12 * ( doff_t )cs_c + \
			                  -diagoffc12 * ( doff_t )rs_c;  \
			incc12     = ldc; \
			ldc12      = incc; \
			conjc12    = conjc; \
\
			p11_m      = panel_dim; \
			p11_n      = panel_dim; \
			j          = diagoffc_abs; \
			p11        = p + (j  )*ldp; \
			c11        = c + (j  )*ldc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( conjc12 ); \
		} \
		else /* if ( ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) || \
		             ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) ) ) */ \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs + panel_dim; \
			diagoffc10 = diagoffc; \
			p10        = p; \
			c10        = c; \
			c10        = c10 + diagoffc10 * ( doff_t )cs_c + \
			                  -diagoffc10 * ( doff_t )rs_c;  \
			incc10     = ldc; \
			ldc10      = incc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			p12        = p + (j  )*ldp; \
			c12        = c + (j  )*ldc; \
			incc12     = incc; \
			ldc12      = ldc; \
			conjc12    = conjc; \
\
			p11_m      = panel_dim; \
			p11_n      = panel_dim; \
			j          = diagoffc_abs; \
			p11        = p + (j  )*ldp; \
			c11        = c + (j  )*ldc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( conjc10 ); \
		} \
\
		/* Pack to P10. For upper storage, this includes the unstored
		   triangle of C11. */ \
		PASTEMAC(ch,packm_cxk)( conjc10, \
		                        p10_dim, \
		                        p10_len, \
		                        kappa, \
		                        c10, incc10, ldc10, \
		                        p10,         ldp ); \
\
		/* Pack to P12. For lower storage, this includes the unstored
		   triangle of C11. */ \
		PASTEMAC(ch,packm_cxk)( conjc12, \
		                        p12_dim, \
		                        p12_len, \
		                        kappa, \
		                        c12, incc12, ldc12, \
		                        p12,         ldp ); \
\
		/* Pack the stored triangule of C11 to P11. */ \
		PASTEMAC3(ch,ch,ch,scal2m_unb_var1)( 0, \
		                                     BLIS_NONUNIT_DIAG, \
		                                     uploc, \
		                                     conjc, \
		                                     p11_m, \
		                                     p11_n, \
		                                     kappa, \
		                                     c11, rs_c,   cs_c, \
		                                     p11, rs_p11, cs_p11 ); \
\
		/* If source matrix C is Hermitian, we have to zero out the
		   imaginary components of the diagonal of P11 in case the
		   corresponding elements in C11 were not already zero. */ \
		if ( bli_is_hermitian( strucc ) ) \
		{ \
			/* NOTE: We can directly increment p11 since we are done
			   using p11 for the remainder of the function. */ \
			for ( i = 0; i < p11_m; ++i ) \
			{ \
				PASTEMAC(ch,seti0s)( *p11 ); \
\
				p11 += rs_p11 + cs_p11; \
			} \
		} \
	} \
\
	/* The packed memory region was acquired/allocated with "aligned"
	   dimensions (ie: dimensions that were possibly inflated up to a
	   multiple). When these dimension are inflated, it creates empty
	   regions along the bottom and/or right edges of the matrix. If
	   either region exists, we set them to zero. This allows the
	   micro-kernel to remain simple since it does not need to support
	   different register blockings for the edge cases. */ \
	if ( m_panel != m_panel_max ) \
	{ \
		dim_t  i      = m_panel; \
		dim_t  m_edge = m_panel_max - i; \
		dim_t  n_edge = n_panel_max; \
		ctype* p_edge = p + (i  )*rs_p; \
\
		PASTEMAC2(ch,ch,setm_unb_var1)( 0, \
		                                BLIS_NONUNIT_DIAG, \
		                                BLIS_DENSE, \
		                                m_edge, \
		                                n_edge, \
		                                zero, \
		                                p_edge, rs_p, cs_p ); \
	} \
\
	if ( n_panel != n_panel_max ) \
	{ \
		dim_t  j      = n_panel; \
		dim_t  m_edge = m_panel_max; \
		dim_t  n_edge = n_panel_max - j; \
		ctype* p_edge = p + (j  )*cs_p; \
\
		PASTEMAC2(ch,ch,setm_unb_var1)( 0, \
		                                BLIS_NONUNIT_DIAG, \
		                                BLIS_DENSE, \
		                                m_edge, \
		                                n_edge, \
		                                zero, \
		                                p_edge, rs_p, cs_p ); \
	} \
\
}

INSERT_GENTFUNC_BASIC0( packm_herm_cxk )

