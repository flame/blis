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
                           struc_t strucc, \
                           doff_t  diagoffp, \
                           diag_t  diagc, \
                           uplo_t  uploc, \
                           conj_t  conjc, \
                           bool_t  invdiag, \
                           dim_t   m_panel, \
                           dim_t   n_panel, \
                           dim_t   m_panel_max, \
                           dim_t   n_panel_max, \
                           ctype*  kappa, \
                           ctype*  c, inc_t rs_c, inc_t cs_c, \
                           ctype*  p, inc_t rs_p, inc_t cs_p  \
                         ) \
{ \
	ctype* restrict c_begin    = c; \
	ctype* restrict p_begin    = p; \
	ctype* restrict kappa_cast = kappa; \
	ctype* restrict zero       = PASTEMAC(ch,0); \
\
	dim_t           panel_dim; \
	dim_t           panel_len; \
	inc_t           incc, ldc; \
	inc_t           ldp; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim = n_panel; \
		panel_len = m_panel; \
		incc      = cs_c; \
		ldc       = rs_c; \
		ldp       = rs_p; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_dim = m_panel; \
		panel_len = n_panel; \
		incc      = rs_c; \
		ldc       = cs_c; \
		ldp       = cs_p; \
	} \
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk)( conjc, \
	                        panel_dim, \
	                        panel_len, \
	                        kappa_cast, \
	                        c_begin, incc, ldc, \
	                        p_begin,       ldp ); \
\
	/* If the diagonal of C is implicitly unit, set the diagonal of
	   the packed panel to unit. */ \
	if ( bli_is_unit_diag( diagc ) ) \
	{ \
		PASTEMAC2(ch,ch,setd_unb_var1)( diagoffp, \
		                                m_panel, \
		                                n_panel, \
		                                kappa_cast, \
		                                p_begin, rs_p, cs_p ); \
	} \
\
	/* If requested, invert the diagonal of the packed panel. */ \
	if ( invdiag == TRUE ) \
	{ \
		PASTEMAC(ch,invertd_unb_var1)( diagoffp, \
		                               m_panel, \
		                               n_panel, \
		                               p_begin, rs_p, cs_p ); \
	} \
\
	/* Set the region opposite the diagonal of P to zero. To do this,
	   we need to reference the "unstored" region on the other side of
	   the diagonal. This amounts to toggling uploc and then shifting
	   the diagonal offset to shrink the newly referenced region (by
	   one diagonal). */ \
	{ \
		uplo_t uplop = uploc; \
\
		bli_toggle_uplo( uplop ); \
		bli_shift_diag_offset_to_shrink_uplo( uplop, diagoffp ); \
\
		PASTEMAC2(ch,ch,setm_unb_var1)( diagoffp, \
		                                BLIS_NONUNIT_DIAG, \
		                                uplop, \
		                                m_panel, \
		                                n_panel, \
		                                zero, \
		                                p_begin, rs_p, cs_p ); \
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
		ctype* p_edge = p_begin + (i  )*rs_p; \
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
		ctype* p_edge = p_begin + (j  )*cs_p; \
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
	/* If this panel is an edge case in both panel dimension and length,
	   then it must be a bottom-right corner case. Set the part of the
	   diagonal that extends into the zero-padded region to identity. */ \
	if ( m_panel != m_panel_max && \
	     n_panel != n_panel_max ) \
	{ \
		dim_t  i      = m_panel; \
		dim_t  j      = n_panel; \
		dim_t  m_br   = m_panel_max - i; \
		dim_t  n_br   = n_panel_max - j; \
		ctype* one    = PASTEMAC(ch,1); \
		ctype* p_edge = p_begin + (i  )*rs_p + (j  )*cs_p; \
\
		PASTEMAC2(ch,ch,setd_unb_var1)( 0, \
		                                m_br, \
		                                n_br, \
		                                one, \
		                                p_edge, rs_p, cs_p ); \
/*
		PASTEMAC(ch,fprintm)( stdout, "packm_var3: setting br unit diag", m_br, n_br, \
		                      p_edge, rs_p, cs_p, "%4.1f", "" ); \
*/ \
	} \
/*
	if ( rs_p == 1 ) \
	PASTEMAC(ch,fprintm)( stdout, "packm_var3: ap copied", m_panel_max, n_panel_max, \
	                      p_begin, rs_p, cs_p, "%4.1f", "" ); \
	if ( cs_p == 1 ) \
	PASTEMAC(ch,fprintm)( stdout, "packm_var3: bp copied", m_panel_max, n_panel_max, \
	                      p_begin, rs_p, cs_p, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( packm_tri_cxk )

