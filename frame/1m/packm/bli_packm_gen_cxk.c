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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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
	dim_t           panel_dim; \
	dim_t           panel_len; \
	inc_t           incc, ldc; \
	inc_t           ldp; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( m_panel, n_panel, rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim = n_panel; \
		panel_len = m_panel; \
		incc      = cs_c; \
		ldc       = rs_c; \
		ldp       = rs_p; \
	} \
	else /* if ( bli_is_col_stored_f( m_panel, n_panel, rs_p, cs_p ) ) */ \
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
}

INSERT_GENTFUNC_BASIC0( packm_gen_cxk )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
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
	ctype_r* restrict zero_r = PASTEMAC(chr,0); \
\
	dim_t             panel_dim; \
	dim_t             panel_len; \
	dim_t             panel_len_max; \
	inc_t             incc, ldc; \
	inc_t             psp, ldp; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( m_panel, n_panel, rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim     = n_panel; \
		panel_len     = m_panel; \
		panel_len_max = m_panel_max; \
		incc          = cs_c; \
		ldc           = rs_c; \
		ldp           = rs_p; \
	} \
	else /* if ( bli_is_col_stored_f( m_panel, n_panel, rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_dim     = m_panel; \
		panel_len     = n_panel; \
		panel_len_max = n_panel_max; \
		incc          = rs_c; \
		ldc           = cs_c; \
		ldp           = cs_p; \
	} \
\
	/* Compute the panel stride (ie: the element offset to the imaginary
	   panel). */ \
	psp = ldp * panel_len_max; \
\
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk_4m)( conjc, \
	                           panel_dim, \
	                           panel_len, \
	                           kappa, \
	                           c, incc, ldc, \
	                           p, psp,  ldp ); \
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
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_gen_cxk_4m )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
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
	ctype_r* restrict zero_r = PASTEMAC(chr,0); \
\
	dim_t             panel_dim; \
	dim_t             panel_len; \
	dim_t             panel_len_max; \
	inc_t             incc, ldc; \
	inc_t             psp, ldp; \
\
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( m_panel, n_panel, rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panel. */ \
		panel_dim     = n_panel; \
		panel_len     = m_panel; \
		panel_len_max = m_panel_max; \
		incc          = cs_c; \
		ldc           = rs_c; \
		ldp           = rs_p; \
	} \
	else /* if ( bli_is_col_stored_f( m_panel, n_panel, rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panel. */ \
		panel_dim     = m_panel; \
		panel_len     = n_panel; \
		panel_len_max = n_panel_max; \
		incc          = rs_c; \
		ldc           = cs_c; \
		ldp           = cs_p; \
	} \
\
	/* Compute the panel stride (ie: the element offset to the imaginary
	   panel). */ \
	psp = ldp * panel_len_max; \
\
\
	/* Pack the panel. */ \
	PASTEMAC(ch,packm_cxk_3m)( conjc, \
	                           panel_dim, \
	                           panel_len, \
	                           kappa, \
	                           c, incc, ldc, \
	                           p, psp,  ldp ); \
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
		dim_t    i          = m_panel; \
		dim_t    m_edge     = m_panel_max - i; \
		dim_t    n_edge     = n_panel_max; \
		ctype_r* p_edge_r   = ( ctype_r* )p +         (i  )*rs_p; \
		ctype_r* p_edge_i   = ( ctype_r* )p +   psp + (i  )*rs_p; \
		ctype_r* p_edge_rpi = ( ctype_r* )p + 2*psp + (i  )*rs_p; \
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
		                    p_edge_rpi, rs_p, cs_p ); \
	} \
\
	if ( n_panel != n_panel_max ) \
	{ \
		dim_t    j          = n_panel; \
		dim_t    m_edge     = m_panel_max; \
		dim_t    n_edge     = n_panel_max - j; \
		ctype_r* p_edge_r   = ( ctype_r* )p +         (j  )*cs_p; \
		ctype_r* p_edge_i   = ( ctype_r* )p +   psp + (j  )*cs_p; \
		ctype_r* p_edge_rpi = ( ctype_r* )p + 2*psp + (j  )*cs_p; \
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
		                    p_edge_rpi, rs_p, cs_p ); \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_gen_cxk_3m )

