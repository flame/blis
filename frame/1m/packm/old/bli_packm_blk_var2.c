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

#define FUNCPTR_T packm_fp

typedef void (*FUNCPTR_T)(
                           struc_t strucc,
                           doff_t  diagoffc,
                           diag_t  diagc,
                           uplo_t  uploc,
                           trans_t transc,
                           dim_t   m,
                           dim_t   n,
                           dim_t   m_max,
                           dim_t   n_max,
                           void*   kappa,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void*   p, inc_t rs_p, inc_t cs_p,
                                      dim_t pd_p, inc_t ps_p
                         );

static FUNCPTR_T GENARRAY(ftypes,packm_blk_var2);


void bli_packm_blk_var2( obj_t*   c,
                         obj_t*   p )
{
	num_t     dt_cp     = bli_obj_datatype( *c );

	struc_t   strucc    = bli_obj_struc( *c );
	doff_t    diagoffc  = bli_obj_diag_offset( *c );
	diag_t    diagc     = bli_obj_diag( *c );
	uplo_t    uploc     = bli_obj_uplo( *c );
	trans_t   transc    = bli_obj_conjtrans_status( *c );

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

	void*     buf_kappa;

	FUNCPTR_T f;

	// This variant assumes that the micro-kernel will always apply the
	// alpha scalar of the higher-level operation. Thus, we use BLIS_ONE
	// for kappa so that the underlying packm implementation does not
	// scale during packing.
	buf_kappa = bli_obj_buffer_for_const( dt_cp, BLIS_ONE );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_cp];

	// Invoke the function.
	f( strucc,
	   diagoffc,
	   diagc,
	   uploc,
	   transc,
	   m_p,
	   n_p,
	   m_max_p,
	   n_max_p,
	   buf_kappa,
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
                            dim_t   m, \
                            dim_t   n, \
                            dim_t   m_max, \
                            dim_t   n_max, \
                            void*   kappa, \
                            void*   c, inc_t rs_c, inc_t cs_c, \
                            void*   p, inc_t rs_p, inc_t cs_p, \
                                       dim_t pd_p, inc_t ps_p  \
                          ) \
{ \
	ctype* restrict c_cast     = c; \
	ctype* restrict p_cast     = p; \
\
	ctype* restrict c_begin; \
	ctype* restrict p_begin; \
\
	dim_t           iter_dim; \
	dim_t           num_iter; \
	dim_t           it, ic, ip; \
\
	dim_t           ic0, ip0; \
	dim_t           ic_inc, ip_inc; \
	dim_t           panel_dim; \
	doff_t          diagoffc_i; \
	doff_t          diagoffc_inc; \
\
	dim_t           panel_dim_i; \
	inc_t           vs_c; \
	dim_t*          m_panel; \
	dim_t*          n_panel; \
	dim_t           m_panel_max; \
	dim_t           n_panel_max; \
	conj_t          conjc; \
\
\
	/* Extract the conjugation bit from the transposition argument. */ \
	conjc = bli_extract_conj( transc ); \
\
	/* If C needs a transposition, induce it so that we can more simply
	   express the remaining parameters and code. */ \
	if ( bli_does_trans( transc ) ) \
	{ \
		bli_swap_incs( rs_c, cs_c ); \
		bli_negate_diag_offset( diagoffc ); \
		bli_toggle_uplo( uploc ); \
		bli_toggle_trans( transc ); \
	} \
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panels. */ \
		iter_dim     = n; \
		panel_dim    = pd_p; \
		vs_c         = cs_c; \
		diagoffc_inc = -( doff_t)panel_dim; \
		m_panel      = &m; \
		n_panel      = &panel_dim_i; \
		m_panel_max  = m_max; \
		n_panel_max  = panel_dim; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panels. */ \
		iter_dim     = m; \
		panel_dim    = pd_p; \
		vs_c         = rs_c; \
		diagoffc_inc = ( doff_t )panel_dim; \
		m_panel      = &panel_dim_i; \
		n_panel      = &n; \
		m_panel_max  = panel_dim; \
		n_panel_max  = n_max; \
	} \
\
	/* Compute the total number of iterations we'll need. */ \
	num_iter = iter_dim / panel_dim + ( iter_dim % panel_dim ? 1 : 0 ); \
\
	/* Set the initial values and increments for indices related to C and P.
	   Currently we only support forwards iteration. */ \
	{ \
		ic0    = 0; \
		ic_inc = panel_dim; \
		ip0    = 0; \
		ip_inc = 1; \
	} \
\
	for ( ic  = ic0,    ip  = ip0,    it  = 0; it < num_iter; \
	      ic += ic_inc, ip += ip_inc, it += 1 ) \
	{ \
		panel_dim_i    = bli_min( panel_dim, iter_dim - ic ); \
\
		diagoffc_i     = diagoffc + (ip  )*diagoffc_inc; \
		c_begin        = c_cast   + (ic  )*vs_c; \
		p_begin        = p_cast   + (ip  )*ps_p; \
\
		/* Call a specialized packm kernel wrapper for Hermitian and
		   symmetric matrices. Otherwise, call the kernel wrapper for
		   general matrices. */ \
		if ( bli_is_herm_or_symm( strucc ) ) \
		{ \
			PASTEMAC(ch,packm_herm_cxk)( strucc, \
			                             diagoffc_i, \
			                             uploc, \
			                             conjc, \
			                             *m_panel, \
			                             *n_panel, \
			                             m_panel_max, \
			                             n_panel_max, \
			                             kappa, \
			                             c_begin, rs_c, cs_c, \
			                             p_begin, rs_p, cs_p ); \
		} \
		else /* if ( bli_is_general( strucc ) ) */ \
		{ \
			PASTEMAC(ch,packm_gen_cxk)( strucc, \
			                            diagoffc_i, \
			                            uploc, \
			                            conjc, \
			                            *m_panel, \
			                            *n_panel, \
			                            m_panel_max, \
			                            n_panel_max, \
			                            kappa, \
			                            c_begin, rs_c, cs_c, \
			                            p_begin, rs_p, cs_p ); \
		} \
\
/*
		if ( rs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: a copied", m_panel_max, n_panel_max, \
		                      p_begin, 1, cs_p, "%4.1f", "" ); \
		if ( cs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: b copied", m_panel_max, n_panel_max, \
		                      p_begin, rs_p, 1, "%4.1f", "" ); \
*/ \
	} \
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var2 )

