/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

#define FUNCPTR_T packm_fp

typedef void (*FUNCPTR_T)(
                           struc_t strucc,
                           doff_t  diagoffc,
                           diag_t  diagc,
                           uplo_t  uploc,
                           trans_t transc,
                           bool_t  invdiag,
                           bool_t  revifup,
                           bool_t  reviflo,
                           dim_t   m,
                           dim_t   n,
                           dim_t   m_max,
                           dim_t   n_max,
                           void*   beta,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void*   p, inc_t rs_p, inc_t cs_p,
                                      dim_t pd_p, inc_t ps_p
                         );

static FUNCPTR_T GENARRAY(ftypes,packm_blk_var3);


void bli_packm_blk_var3( obj_t*   beta,
                         obj_t*   c,
                         obj_t*   p )
{
	num_t     dt_cp     = bli_obj_datatype( *c );

	struc_t   strucc    = bli_obj_struc( *c );
	doff_t    diagoffc  = bli_obj_diag_offset( *c );
	diag_t    diagc     = bli_obj_diag( *c );
	uplo_t    uploc     = bli_obj_uplo( *c );
	trans_t   transc    = bli_obj_conjtrans_status( *c );
	bool_t    invdiag   = bli_obj_has_inverted_diag( *p );
	bool_t    revifup   = bli_obj_is_pack_rev_if_upper( *p );
	bool_t    reviflo   = bli_obj_is_pack_rev_if_lower( *p );

	dim_t     m_p       = bli_obj_length( *p );
	dim_t     n_p       = bli_obj_width( *p );
	dim_t     m_max_p   = bli_obj_padded_length( *p );
	dim_t     n_max_p   = bli_obj_padded_width( *p );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	void*     buf_p     = bli_obj_buffer_at_off( *p );
	inc_t     rs_p      = bli_obj_row_stride( *p );
	inc_t     cs_p      = bli_obj_col_stride( *p );
	dim_t     pd_p      = bli_obj_panel_dim( *p );
	inc_t     ps_p      = bli_obj_panel_stride( *p );

	void*     buf_beta  = bli_obj_scalar_buffer( dt_cp, *beta );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_cp];

	// Invoke the function.
	f( strucc,
	   diagoffc,
	   diagc,
	   uploc,
	   transc,
	   invdiag,
	   revifup,
	   reviflo,
	   m_p,
	   n_p,
	   m_max_p,
	   n_max_p,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   buf_p, rs_p, cs_p,
	          pd_p, ps_p );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname )( \
                            struc_t strucc, \
                            doff_t  diagoffc, \
                            diag_t  diagc, \
                            uplo_t  uploc, \
                            trans_t transc, \
                            bool_t  invdiag, \
                            bool_t  revifup, \
                            bool_t  reviflo, \
                            dim_t   m, \
                            dim_t   n, \
                            dim_t   m_max, \
                            dim_t   n_max, \
                            void*   beta, \
                            void*   c, inc_t rs_c, inc_t cs_c, \
                            void*   p, inc_t rs_p, inc_t cs_p, \
                                       dim_t pd_p, inc_t ps_p  \
                          ) \
{ \
	ctype* restrict beta_cast = beta; \
	ctype* restrict c_cast    = c; \
	ctype* restrict p_cast    = p; \
	ctype* restrict zero      = PASTEMAC(ch,0); \
	ctype* restrict c_begin; \
	ctype* restrict p_begin; \
\
	dim_t           iter_dim; \
	dim_t           num_iter; \
	dim_t           it, ic, ip; \
	dim_t           ic0, ip0; \
	doff_t          ic_inc, ip_inc; \
	dim_t           panel_dim; \
	dim_t           panel_len; \
	dim_t           panel_len_max; \
	doff_t          diagoffc_i; \
	doff_t          diagoffc_inc; \
	dim_t           panel_dim_i; \
	dim_t           panel_len_i; \
	dim_t           panel_len_max_i; \
	dim_t           panel_off_i; \
	inc_t           vs_c; \
	inc_t           incc, ldc; \
	inc_t           ldp, p_inc; \
	dim_t*          m_panel; \
	dim_t*          n_panel; \
	conj_t          conjc; \
\
	ctype* restrict c_use; \
	ctype* restrict p_use; \
	dim_t*          m_panel_use; \
	dim_t*          n_panel_use; \
	doff_t          diagoffp; \
\
\
	/* If C is zeros, then we don't need to pack it. */ \
	if ( bli_is_zeros( uploc ) ) return; \
\
	/* Extract the conjugation bit from the transposition argument. */ \
	conjc = bli_extract_conj( transc ); \
\
	/* If c needs a transposition, induce it so that we can more simply
	   express the remaining parameters and code. */ \
	if ( bli_does_trans( transc ) ) \
	{ \
		bli_swap_incs( rs_c, cs_c ); \
		bli_negate_diag_offset( diagoffc ); \
		bli_toggle_uplo( uploc ); \
		bli_toggle_trans( transc ); \
	} \
\
	/* If the strides of P indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panels. */ \
		iter_dim      = n; \
		panel_len     = m; \
		panel_len_max = m_max; \
		panel_dim     = pd_p; \
		incc          = cs_c; \
		ldc           = rs_c; \
		vs_c          = cs_c; \
		diagoffc_inc  = -( doff_t)panel_dim; \
		ldp           = rs_p; \
		m_panel       = &m; \
		n_panel       = &panel_dim_i; \
		m_panel_use   = &panel_len_i; \
		n_panel_use   = &panel_dim_i; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panels. */ \
		iter_dim      = m; \
		panel_len     = n; \
		panel_len_max = n_max; \
		panel_dim     = pd_p; \
		incc          = rs_c; \
		ldc           = cs_c; \
		vs_c          = rs_c; \
		diagoffc_inc  = ( doff_t )panel_dim; \
		ldp           = cs_p; \
		m_panel       = &panel_dim_i; \
		n_panel       = &n; \
		m_panel_use   = &panel_dim_i; \
		n_panel_use   = &panel_len_i; \
	} \
\
	/* Compute the total number of iterations we'll need. */ \
	num_iter = iter_dim / panel_dim + ( iter_dim % panel_dim ? 1 : 0 ); \
\
	/* Set the initial values and increments for indices related to C and P
	   based on whether reverse iteration was requested. */ \
	if ( ( revifup && bli_is_upper( uploc ) ) || \
	     ( reviflo && bli_is_lower( uploc ) ) ) \
	{ \
		ic0    = (num_iter - 1) * panel_dim; \
		ic_inc = -panel_dim; \
		ip0    = num_iter - 1; \
		ip_inc = -1; \
	} \
	else \
	{ \
		ic0    = 0; \
		ic_inc = panel_dim; \
		ip0    = 0; \
		ip_inc = 1; \
	} \
\
	p_begin = p_cast; \
\
	for ( ic  = ic0,    ip  = ip0,    it  = 0; it < num_iter; \
	      ic += ic_inc, ip += ip_inc, it += 1 ) \
	{ \
		panel_dim_i = bli_min( panel_dim, iter_dim - ic ); \
\
		diagoffc_i  = diagoffc + (ip  )*diagoffc_inc; \
		c_begin     = c_cast   + (ic  )*vs_c; \
\
		/* If the current panel is unstored, do nothing. (Notice that we use
		   the continue statement, so we don't even increment p_begin.)
		   If the current panel intersects the diagonal (and the matrix is
		   triangular), pack only as much as we need (ie: skip over as much
		   as possible on the unstored side of the diagonal).
		   Otherwise, we assume the current panel is full-length. */ \
		if ( bli_is_unstored_subpart_n( diagoffc_i, uploc, *m_panel, *n_panel ) ) \
		{ \
			continue; \
		} \
		else if ( bli_intersects_diag_n( diagoffc_i, *m_panel, *n_panel ) && \
		          bli_is_triangular( strucc ) ) \
		{ \
			/* Sanity check. Diagonals should not intersect the short end of
			   a micro-panel. If they do, then somehow the constraints on
			   cache blocksizes being a whole multiple of the register
			   blocksizes was somehow violated. */ \
			if ( ( bli_is_col_stored_f( rs_p, cs_p ) && diagoffc_i < 0 ) || \
			     ( bli_is_row_stored_f( rs_p, cs_p ) && diagoffc_i > 0 ) ) \
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
			if      ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) )  \
			{ \
				panel_off_i     = 0; \
				panel_len_i     = bli_min( panel_len,     -diagoffc_i + panel_dim_i ); \
				panel_len_max_i = bli_min( panel_len_max, -diagoffc_i + panel_dim ); \
				diagoffp        = diagoffc_i; \
			} \
			else if ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) \
			{ \
				panel_off_i     = bli_abs( bli_min( diagoffc_i, 0 ) ); \
				panel_len_i     = panel_len     - panel_off_i; \
				panel_len_max_i = panel_len_max - panel_off_i; \
				diagoffp        = diagoffc_i + panel_off_i; \
			} \
			else if ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) )  \
			{ \
				panel_off_i     = bli_max( diagoffc_i, 0 ); \
				panel_len_i     = panel_len     - panel_off_i; \
				panel_len_max_i = panel_len_max - panel_off_i; \
				diagoffp        = diagoffc_i - panel_off_i; \
			} \
			else /* if ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) */ \
			{ \
				panel_off_i     = 0; \
				panel_len_i     = bli_min( panel_len,     diagoffc_i + panel_dim_i ); \
				panel_len_max_i = bli_min( panel_len_max, diagoffc_i + panel_dim ); \
				diagoffp        = diagoffc_i; \
			} \
\
			/* Adjust the pointer to the beginning of the panel in C based on
			   the offset determined above. */ \
			c_use = c_begin + (panel_off_i  )*ldc; \
			p_use = p_begin; \
\
			/* Pack the panel. */ \
			PASTEMAC(ch,packm_cxk)( conjc, \
			                        panel_dim_i, \
			                        panel_len_i, \
			                        beta_cast, \
			                        c_use, incc, ldc, \
			                        p_use,       ldp ); \
\
			/* If the diagonal of C is implicitly unit, set the diagonal of
			   the packed panel to unit. */ \
			if ( bli_is_unit_diag( diagc ) ) \
			{ \
				PASTEMAC2(ch,ch,setd_unb_var1)( diagoffp, \
				                                *m_panel_use, \
				                                *n_panel_use, \
				                                beta_cast, \
				                                p_use, rs_p, cs_p ); \
			} \
\
			/* If requested, invert the diagonal of the packed panel. */ \
			if ( invdiag == TRUE ) \
			{ \
				PASTEMAC(ch,invertd_unb_var1)( diagoffp, \
				                               *m_panel_use, \
				                               *n_panel_use, \
				                               p_use, rs_p, cs_p ); \
			} \
\
			/* Always densify the unstored part of the packed panel. */ \
			{ \
				uplo_t uplop = uploc; \
\
				/* For triangular matrices, we wish to reference the region
				   strictly opposite the diagonal of C. This amounts to 
				   toggling uploc and then shifting the diagonal offset to
				   shrink the stored region (by one diagonal). */ \
				bli_toggle_uplo( uplop ); \
				bli_shift_diag_offset_to_shrink_uplo( uplop, diagoffp ); \
\
				/* Set the region opposite the diagonal of P to zero. */ \
				PASTEMAC2(ch,ch,setm_unb_var1)( diagoffp, \
				                                BLIS_NONUNIT_DIAG, \
				                                uplop, \
				                                *m_panel_use, \
				                                *n_panel_use, \
				                                zero, \
				                                p_use, rs_p, cs_p ); \
			} \
\
			p_inc = ldp * panel_len_max_i; \
		} \
		else \
		{ \
			c_use = c_begin; \
			p_use = p_begin; \
\
			/* Pack a full-length panel. */ \
			panel_off_i     = 0; \
			panel_len_i     = panel_len; \
			panel_len_max_i = panel_len_max; \
\
			/* Pack the panel. */ \
			PASTEMAC(ch,packm_cxk)( conjc, \
			                        panel_dim_i, \
			                        panel_len_i, \
			                        beta_cast, \
			                        c_use, incc, ldc, \
			                        p_use,       ldp ); \
\
			p_inc = ldp * panel_len_max_i; \
		} \
\
		/* If necessary, zero-pad at the edge of the panel dimension (ie: along
		   the long dimension of the panel). */ \
		if ( panel_dim_i != panel_dim ) \
		{ \
			/* Note that this code does the right thing for both row and
			   column panels, since an m x n column-stored row panel and an
			   n x m row-stored column panel look the same in memory. */ \
			dim_t  i      = panel_dim_i; \
			dim_t  m_edge = panel_dim - i; \
			dim_t  n_edge = panel_len_max_i; \
			inc_t  rs_pe  = 1; \
			inc_t  cs_pe  = ldp; \
			ctype* p_edge = p_begin + (i  )*rs_pe; \
\
			PASTEMAC2(ch,ch,setm_unb_var1)( 0, \
			                                BLIS_NONUNIT_DIAG, \
			                                BLIS_DENSE, \
			                                m_edge, \
			                                n_edge, \
			                                zero, \
			                                p_edge, rs_pe, cs_pe ); \
		} \
\
		/* If necessary, zero-pad at the far end of the panel (ie: at the
		   other side of the long dimension of the panel). */ \
		if ( panel_len_i != panel_len_max_i ) \
		{ \
			/* Note that this code does the right thing for both row and
			   column panels, since an m x n column-stored row panel and an
			   n x m row-stored column panel look the same in memory. */ \
			/* Note that we set m_edge as panel_dim, and not panel_dim_i;
			   this is so that we can simultaneously zero out the corner
			   region (if it exists). */ \
			dim_t  j      = panel_len_i; \
			dim_t  m_edge = panel_dim; \
			dim_t  n_edge = panel_len_max_i - j; \
			inc_t  rs_pe  = 1; \
			inc_t  cs_pe  = ldp; \
			ctype* p_edge = p_begin + (j  )*cs_pe; \
\
			PASTEMAC2(ch,ch,setm_unb_var1)( 0, \
			                                BLIS_NONUNIT_DIAG, \
			                                BLIS_DENSE, \
			                                m_edge, \
			                                n_edge, \
			                                zero, \
			                                p_edge, rs_pe, cs_pe ); \
		} \
\
		/* If this panel is an edge case in both panel dimension and length,
		   then it must be a bottom-right corner case. Set the part of the
		   diagonal that extends into the zero-padded region to identity. */ \
		if ( panel_dim_i != panel_dim && \
		     panel_len_i != panel_len_max_i ) \
		{ \
			/* Note that this code does the right thing for both row and
			   column panels, since an m x n column-stored row panel and an
			   n x m row-stored column panel look the same in memory. */ \
			dim_t  i      = panel_dim_i; \
			dim_t  j      = panel_len_i; \
			dim_t  m_br   = panel_dim       - i; \
			dim_t  n_br   = panel_len_max_i - j; \
			inc_t  rs_pe  = 1; \
			inc_t  cs_pe  = ldp; \
			ctype* one    = PASTEMAC(ch,1); \
			ctype* p_edge = p_begin + (i  )*rs_pe + (j  )*cs_pe; \
\
			PASTEMAC2(ch,ch,setd_unb_var1)( 0, \
			                                m_br, \
			                                n_br, \
			                                one, \
			                                p_edge, rs_pe, cs_pe ); \
\
/*
			PASTEMAC(ch,fprintm)( stdout, "packm_var3: setting br unit diag", m_br, n_br, \
			                      p_edge, rs_pe, cs_pe, "%5.2f", "" ); \
*/ \
		} \
\
/*
		if ( rs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_var3: ap copied", panel_dim, panel_len_max_i, \
		                      p_begin, rs_p, cs_p, "%4.1f", "" ); \
		if ( cs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_var3: bp copied", panel_len_max_i, panel_dim, \
		                      p_begin, rs_p, cs_p, "%4.1f", "" ); \
*/ \
\
	 	p_begin += p_inc; \
	} \
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var3 )

