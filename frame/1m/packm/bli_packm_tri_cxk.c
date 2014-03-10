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
                           doff_t          diagoffp, \
                           diag_t          diagc, \
                           uplo_t          uploc, \
                           conj_t          conjc, \
                           bool_t          invdiag, \
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
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk)( conjc, \
	                        panel_dim, \
	                        panel_len, \
	                        kappa, \
	                        c, incc, ldc, \
	                        p,       ldp ); \
\
\
	/* If the diagonal of c is implicitly unit, explicitly set the
	   the diagonal of the packed panel to kappa. */ \
	if ( bli_is_unit_diag( diagc ) ) \
	{ \
		PASTEMAC(ch,setd)( diagoffp, \
		                   m_panel, \
		                   n_panel, \
		                   kappa, \
		                   p, rs_p, cs_p ); \
	} \
\
	/* If requested, invert the diagonal of the packed panel. */ \
	if ( invdiag == TRUE ) \
	{ \
		PASTEMAC(ch,invertd)( diagoffp, \
		                      m_panel, \
		                      n_panel, \
		                      p, rs_p, cs_p ); \
	} \
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
		uplo_t uplop = uploc; \
\
		bli_toggle_uplo( uplop ); \
		bli_shift_diag_offset_to_shrink_uplo( uplop, diagoffp ); \
\
		PASTEMAC(ch,setm)( diagoffp, \
		                   BLIS_NONUNIT_DIAG, \
		                   uplop, \
		                   m_panel, \
		                   n_panel, \
		                   zero, \
		                   p, rs_p, cs_p ); \
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
		PASTEMAC(ch,setm)( 0, \
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
		PASTEMAC(ch,setm)( 0, \
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
	   diagonal that extends into the zero-padded region to identity.
	   NOTE: This is actually only necessary when packing for trsm, as
	   it helps prevent NaNs and Infs from creeping into the computation.
	   However, we set the region to identity for trmm as well. Those
	   1.0's end up getting muliplied by the 0.0's in the zero-padded
	   region of the other matrix, so there is no harm in this. */ \
	if ( m_panel != m_panel_max && \
	     n_panel != n_panel_max ) \
	{ \
		dim_t  i      = m_panel; \
		dim_t  j      = n_panel; \
		dim_t  m_br   = m_panel_max - i; \
		dim_t  n_br   = n_panel_max - j; \
		ctype* one    = PASTEMAC(ch,1); \
		ctype* p_br   = p + (i  )*rs_p + (j  )*cs_p; \
\
		PASTEMAC(ch,setd)( 0, \
		                   m_br, \
		                   n_br, \
		                   one, \
		                   p_br, rs_p, cs_p ); \
	} \
/*
		PASTEMAC(ch,fprintm)( stdout, "packm_var1: setting br unit diag", m_br, n_br, \
		                      p_edge, rs_p, cs_p, "%4.1f", "" ); \
*/ \
/*
	if ( rs_p == 1 ) \
	PASTEMAC(ch,fprintm)( stdout, "packm_var1: ap copied", m_panel_max, n_panel_max, \
	                      p, rs_p, cs_p, "%4.1f", "" ); \
	if ( cs_p == 1 ) \
	PASTEMAC(ch,fprintm)( stdout, "packm_var1: bp copied", m_panel_max, n_panel_max, \
	                      p, rs_p, cs_p, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( packm_tri_cxk )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           struc_t         strucc, \
                           doff_t          diagoffp, \
                           diag_t          diagc, \
                           uplo_t          uploc, \
                           conj_t          conjc, \
                           bool_t          invdiag, \
                           dim_t           m_panel, \
                           dim_t           n_panel, \
                           dim_t           m_panel_max, \
                           dim_t           n_panel_max, \
                           ctype* restrict kappa, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           ctype* restrict p, inc_t rs_p, inc_t cs_p  \
                         ) \
{ \
	ctype_r* restrict zero_r = PASTEMAC(chr,0); \
	ctype_r* restrict one_r  = PASTEMAC(chr,1); \
\
	dim_t             i; \
	dim_t             panel_dim; \
	dim_t             panel_len; \
	dim_t             panel_len_max; \
	inc_t             incc, ldc; \
	inc_t             psp, ldp; \
\
	inc_t             rs_p11, cs_p11; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim     = n_panel; \
		panel_len     = m_panel; \
		panel_len_max = m_panel_max; \
		incc          = cs_c; \
		ldc           = rs_c; \
		ldp           = rs_p; \
		rs_p11        = rs_p; \
		cs_p11        = 1; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_dim     = m_panel; \
		panel_len     = n_panel; \
		panel_len_max = n_panel_max; \
		incc          = rs_c; \
		ldc           = cs_c; \
		ldp           = cs_p; \
		rs_p11        = 1; \
		cs_p11        = cs_p; \
	} \
\
	/* Compute the panel stride (ie: the element offset to the imaginary
	   panel). */ \
	psp = ldp * panel_len_max; \
\
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk_ri)( conjc, \
	                           panel_dim, \
	                           panel_len, \
	                           kappa, \
	                           c, incc, ldc, \
	                           p, psp,  ldp ); \
\
\
	/* Tweak the panel according to its triangular structure */ \
	{ \
		dim_t    j     = bli_abs( diagoffp ); \
		ctype_r* p11_r = ( ctype_r* )p +       (j  )*ldp; \
		ctype_r* p11_i = ( ctype_r* )p + psp + (j  )*ldp; \
\
		/* If the diagonal of c is implicitly unit, explicitly set the
		   the diagonal of the packed panel to kappa. */ \
		if ( bli_is_unit_diag( diagc ) ) \
		{ \
			ctype_r kappa_r = PASTEMAC(ch,real)( *kappa ); \
			ctype_r kappa_i = PASTEMAC(ch,imag)( *kappa ); \
\
			PASTEMAC(chr,setd)( 0, \
			                    m_panel, \
			                    n_panel, \
			                    &kappa_r, \
			                    p11_r, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setd)( 0, \
			                    m_panel, \
			                    n_panel, \
			                    &kappa_i, \
			                    p11_i, rs_p11, cs_p11 ); \
		} \
\
		/* If requested, invert the diagonal of the packed panel. */ \
		if ( invdiag == TRUE ) \
		{ \
			for ( i = 0; i < panel_dim; ++i ) \
			{ \
				ctype_r* pi11_r = p11_r + (i  )*rs_p + (i  )*cs_p; \
				ctype_r* pi11_i = p11_i + (i  )*rs_p + (i  )*cs_p; \
\
				PASTEMAC(ch,invertris)( *pi11_r, *pi11_i ); \
			} \
		} \
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
			uplo_t uplop11    = uploc; \
			doff_t diagoffp11 = 0; \
\
			bli_toggle_uplo( uplop11 ); \
			bli_shift_diag_offset_to_shrink_uplo( uplop11, diagoffp11 ); \
\
			PASTEMAC(chr,setm)( diagoffp11, \
			                    BLIS_NONUNIT_DIAG, \
			                    uplop11, \
			                    panel_dim, \
			                    panel_dim, \
			                    zero_r, \
			                    p11_r, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setm)( diagoffp11, \
			                    BLIS_NONUNIT_DIAG, \
			                    uplop11, \
			                    panel_dim, \
			                    panel_dim, \
			                    zero_r, \
			                    p11_i, rs_p11, cs_p11 ); \
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
		dim_t    i        = m_panel; \
		dim_t    m_edge   = m_panel_max - i; \
		dim_t    n_edge   = n_panel_max; \
		ctype_r* p_edge_r = ( ctype_r* )p +       (i  )*rs_p; \
		ctype_r* p_edge_i = ( ctype_r* )p + psp + (i  )*rs_p; \
\
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_r, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_i, rs_p, cs_p ); \
	} \
\
	if ( n_panel != n_panel_max ) \
	{ \
		dim_t    j        = n_panel; \
		dim_t    m_edge   = m_panel_max; \
		dim_t    n_edge   = n_panel_max - j; \
		ctype_r* p_edge_r = ( ctype_r* )p +       (j  )*cs_p; \
		ctype_r* p_edge_i = ( ctype_r* )p + psp + (j  )*cs_p; \
\
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_r, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_i, rs_p, cs_p ); \
\
	} \
\
\
	/* If this panel is an edge case in both panel dimension and length,
	   then it must be a bottom-right corner case. Set the part of the
	   diagonal that extends into the zero-padded region to identity.
	   NOTE: This is actually only necessary when packing for trsm, as
	   it helps prevent NaNs and Infs from creeping into the computation.
	   However, we set the region to identity for trmm as well. Those
	   1.0's end up getting muliplied by the 0.0's in the zero-padded
	   region of the other matrix, so there is no harm in this. */ \
	if ( m_panel != m_panel_max && \
	     n_panel != n_panel_max ) \
	{ \
		dim_t    i        = m_panel; \
		dim_t    j        = n_panel; \
		dim_t    m_br     = m_panel_max - i; \
		dim_t    n_br     = n_panel_max - j; \
		ctype_r* p_br_r   = ( ctype_r* )p +       (i  )*rs_p + (j  )*cs_p; \
		ctype_r* p_br_i   = ( ctype_r* )p + psp + (i  )*rs_p + (j  )*cs_p; \
\
		PASTEMAC(chr,setd)( 0, \
		                    m_br, \
		                    n_br, \
		                    one_r, \
		                    p_br_r, rs_p, cs_p ); \
		PASTEMAC(chr,setd)( 0, \
		                    m_br, \
		                    n_br, \
		                    zero_r, \
		                    p_br_i, rs_p, cs_p ); \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_tri_cxk_ri )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           struc_t         strucc, \
                           doff_t          diagoffp, \
                           diag_t          diagc, \
                           uplo_t          uploc, \
                           conj_t          conjc, \
                           bool_t          invdiag, \
                           dim_t           m_panel, \
                           dim_t           n_panel, \
                           dim_t           m_panel_max, \
                           dim_t           n_panel_max, \
                           ctype* restrict kappa, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           ctype* restrict p, inc_t rs_p, inc_t cs_p  \
                         ) \
{ \
	ctype_r* restrict zero_r = PASTEMAC(chr,0); \
	ctype_r* restrict one_r  = PASTEMAC(chr,1); \
\
	dim_t             i; \
	dim_t             panel_dim; \
	dim_t             panel_len; \
	dim_t             panel_len_max; \
	inc_t             incc, ldc; \
	inc_t             psp, ldp; \
\
	inc_t             rs_p11, cs_p11; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim     = n_panel; \
		panel_len     = m_panel; \
		panel_len_max = m_panel_max; \
		incc          = cs_c; \
		ldc           = rs_c; \
		ldp           = rs_p; \
		rs_p11        = rs_p; \
		cs_p11        = 1; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_dim     = m_panel; \
		panel_len     = n_panel; \
		panel_len_max = n_panel_max; \
		incc          = rs_c; \
		ldc           = cs_c; \
		ldp           = cs_p; \
		rs_p11        = 1; \
		cs_p11        = cs_p; \
	} \
\
	/* Compute the panel stride (ie: the element offset to the imaginary
	   panel). */ \
	psp = ldp * panel_len_max; \
\
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk_ri3)( conjc, \
	                            panel_dim, \
	                            panel_len, \
	                            kappa, \
	                            c, incc, ldc, \
	                            p, psp,  ldp ); \
\
\
	/* Tweak the panel according to its triangular structure */ \
	{ \
		dim_t    j     = bli_abs( diagoffp ); \
		ctype_r* p11_r  = ( ctype_r* )p +         (j  )*ldp; \
		ctype_r* p11_i  = ( ctype_r* )p +   psp + (j  )*ldp; \
		ctype_r* p11_ri = ( ctype_r* )p + 2*psp + (j  )*ldp; \
\
		/* If the diagonal of c is implicitly unit, explicitly set the
		   the diagonal of the packed panel to kappa. */ \
		if ( bli_is_unit_diag( diagc ) ) \
		{ \
			ctype_r kappa_r = PASTEMAC(ch,real)( *kappa ); \
			ctype_r kappa_i = PASTEMAC(ch,imag)( *kappa ); \
\
			PASTEMAC(chr,setd)( 0, \
			                    m_panel, \
			                    n_panel, \
			                    &kappa_r, \
			                    p11_r, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setd)( 0, \
			                    m_panel, \
			                    n_panel, \
			                    &kappa_i, \
			                    p11_i, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setd)( 0, \
			                    m_panel, \
			                    n_panel, \
			                    &kappa_r, \
			                    p11_ri, rs_p11, cs_p11 ); \
		} \
\
		/* If requested, invert the diagonal of the packed panel. Note
		   that we do not need to update the ri panel since inverted
		   diagonals are only needed by trsm, which does not use the
		   p11 section of the ri panel. */ \
		if ( invdiag == TRUE ) \
		{ \
			for ( i = 0; i < panel_dim; ++i ) \
			{ \
				ctype_r* pi11_r = p11_r + (i  )*rs_p + (i  )*cs_p; \
				ctype_r* pi11_i = p11_i + (i  )*rs_p + (i  )*cs_p; \
\
				PASTEMAC(ch,invertris)( *pi11_r, *pi11_i ); \
			} \
		} \
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
			uplo_t uplop11    = uploc; \
			doff_t diagoffp11 = 0; \
\
			bli_toggle_uplo( uplop11 ); \
			bli_shift_diag_offset_to_shrink_uplo( uplop11, diagoffp11 ); \
\
			PASTEMAC(chr,setm)( diagoffp11, \
			                    BLIS_NONUNIT_DIAG, \
			                    uplop11, \
			                    panel_dim, \
			                    panel_dim, \
			                    zero_r, \
			                    p11_r, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setm)( diagoffp11, \
			                    BLIS_NONUNIT_DIAG, \
			                    uplop11, \
			                    panel_dim, \
			                    panel_dim, \
			                    zero_r, \
			                    p11_i, rs_p11, cs_p11 ); \
			PASTEMAC(chr,setm)( diagoffp11, \
			                    BLIS_NONUNIT_DIAG, \
			                    uplop11, \
			                    panel_dim, \
			                    panel_dim, \
			                    zero_r, \
			                    p11_ri, rs_p11, cs_p11 ); \
		} \
	} \
\
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
		dim_t    i         = m_panel; \
		dim_t    m_edge    = m_panel_max - i; \
		dim_t    n_edge    = n_panel_max; \
		ctype_r* p_edge_r  = ( ctype_r* )p +         (i  )*rs_p; \
		ctype_r* p_edge_i  = ( ctype_r* )p +   psp + (i  )*rs_p; \
		ctype_r* p_edge_ri = ( ctype_r* )p + 2*psp + (i  )*rs_p; \
\
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_r, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_i, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_ri, rs_p, cs_p ); \
	} \
\
	if ( n_panel != n_panel_max ) \
	{ \
		dim_t    j        = n_panel; \
		dim_t    m_edge   = m_panel_max; \
		dim_t    n_edge   = n_panel_max - j; \
		ctype_r* p_edge_r  = ( ctype_r* )p +         (j  )*cs_p; \
		ctype_r* p_edge_i  = ( ctype_r* )p +   psp + (j  )*cs_p; \
		ctype_r* p_edge_ri = ( ctype_r* )p + 2*psp + (j  )*cs_p; \
\
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_r, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_i, rs_p, cs_p ); \
		PASTEMAC(chr,setm)( 0, \
		                    BLIS_NONUNIT_DIAG, \
		                    BLIS_DENSE, \
		                    m_edge, \
		                    n_edge, \
		                    zero_r, \
		                    p_edge_ri, rs_p, cs_p ); \
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
	if ( m_panel != m_panel_max && \
	     n_panel != n_panel_max ) \
	{ \
		dim_t    i        = m_panel; \
		dim_t    j        = n_panel; \
		dim_t    m_br     = m_panel_max - i; \
		dim_t    n_br     = n_panel_max - j; \
		ctype_r* p_br_r   = ( ctype_r* )p +       (i  )*rs_p + (j  )*cs_p; \
		ctype_r* p_br_i   = ( ctype_r* )p + psp + (i  )*rs_p + (j  )*cs_p; \
\
		PASTEMAC(chr,setd)( 0, \
		                    m_br, \
		                    n_br, \
		                    one_r, \
		                    p_br_r, rs_p, cs_p ); \
		PASTEMAC(chr,setd)( 0, \
		                    m_br, \
		                    n_br, \
		                    zero_r, \
		                    p_br_i, rs_p, cs_p ); \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_tri_cxk_ri3 )

