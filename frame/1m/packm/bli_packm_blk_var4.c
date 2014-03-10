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
                           void*   kappa,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void*   p, inc_t rs_p, inc_t cs_p,
                                      dim_t pd_p, inc_t ps_p
                         );

//static FUNCPTR_T GENARRAY(ftypes,packm_blk_var4);


void bli_packm_blk_var4( obj_t*   c,
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

	obj_t     kappa;
	obj_t*    kappa_p;
	void*     buf_kappa;

	FUNCPTR_T f;


	// We want this variant to behave identically to that of variant 1
	// in the real domain.
	if ( bli_is_real( dt_cp ) )
	{
		bli_packm_blk_var1( c, p, &BLIS_PACKM_SINGLE_THREADED );
		return;
	}

	// The value for kappa we use will depend on whether the scalar
	// attached to A has a nonzero imaginary component. If it does,
	// then we will apply the scalar during packing to facilitate
	// implementing complex domain micro-kernels in terms of their
	// real domain counterparts. (In the aforementioned situation,
	// applying a real scalar is easy, but applying a complex one is
	// harder, so we avoid the need altogether with the code below.)
	if ( bli_obj_scalar_has_nonzero_imag( p ) )
	{
		// Detach the scalar.
		bli_obj_scalar_detach( p, &kappa );

		// Reset the attached scalar (to 1.0).
		bli_obj_scalar_reset( p );

		kappa_p = &kappa;
	}
	else
	{
		// If the internal scalar of A has only a real component, then
		// we will apply it later (in the micro-kernel), and so we will
		// use BLIS_ONE to indicate no scaling during packing.
		kappa_p = &BLIS_ONE;
	}


	// Acquire the buffer to the kappa chosen above.
	buf_kappa = bli_obj_buffer_for_1x1( dt_cp, *kappa_p );


	// Index into the type combination array to extract the correct
	// function pointer.
	//f = ftypes[dt_cp];
	if ( bli_is_scomplex( dt_cp ) ) f = bli_cpackm_blk_var4;
	else                            f = bli_zpackm_blk_var4;

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
	   buf_kappa,
	   buf_c, rs_c, cs_c,
	   buf_p, rs_p, cs_p,
	          pd_p, ps_p );
}


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
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
                           void*   kappa, \
                           void*   c, inc_t rs_c, inc_t cs_c, \
                           void*   p, inc_t rs_p, inc_t cs_p, \
                                      dim_t pd_p, inc_t ps_p  \
                         ) \
{ \
	ctype* restrict kappa_cast = kappa; \
	ctype* restrict c_cast     = c; \
	ctype* restrict p_cast     = p; \
	ctype* restrict c_begin; \
	ctype* restrict p_begin; \
\
	dim_t           iter_dim; \
	dim_t           num_iter; \
	dim_t           it, ic, ip; \
	dim_t           ic0, ip0; \
	doff_t          ic_inc, ip_inc; \
	doff_t          diagoffc_i; \
	doff_t          diagoffc_inc; \
	dim_t           panel_len_full; \
	dim_t           panel_len_i; \
	dim_t           panel_len_max; \
	dim_t           panel_len_max_i; \
	dim_t           panel_dim_i; \
	dim_t           panel_dim_max; \
	dim_t           panel_off_i; \
	inc_t           vs_c; \
	inc_t           ldc; \
	inc_t           ldp, p_inc; \
	dim_t*          m_panel_full; \
	dim_t*          n_panel_full; \
	dim_t*          m_panel_use; \
	dim_t*          n_panel_use; \
	dim_t*          m_panel_max; \
	dim_t*          n_panel_max; \
	conj_t          conjc; \
\
	ctype* restrict c_use; \
	ctype* restrict p_use; \
	doff_t          diagoffp_i; \
\
\
	/* If C is zeros and part of a triangular matrix, then we don't need
	   to pack it. */ \
	if ( bli_is_zeros( uploc ) && \
	     bli_is_triangular( strucc ) ) return; \
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
		iter_dim       = n; \
		panel_len_full = m; \
		panel_len_max  = m_max; \
		panel_dim_max  = pd_p; \
		ldc            = rs_c; \
		vs_c           = cs_c; \
		diagoffc_inc   = -( doff_t)panel_dim_max; \
		ldp            = rs_p; \
		m_panel_full   = &m; \
		n_panel_full   = &panel_dim_i; \
		m_panel_use    = &panel_len_i; \
		n_panel_use    = &panel_dim_i; \
		m_panel_max    = &panel_len_max_i; \
		n_panel_max    = &panel_dim_max; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panels. */ \
		iter_dim       = m; \
		panel_len_full = n; \
		panel_len_max  = n_max; \
		panel_dim_max  = pd_p; \
		ldc            = cs_c; \
		vs_c           = rs_c; \
		diagoffc_inc   = ( doff_t )panel_dim_max; \
		ldp            = cs_p; \
		m_panel_full   = &panel_dim_i; \
		n_panel_full   = &n; \
		m_panel_use    = &panel_dim_i; \
		n_panel_use    = &panel_len_i; \
		m_panel_max    = &panel_dim_max; \
		n_panel_max    = &panel_len_max_i; \
	} \
\
	/* Compute the total number of iterations we'll need. */ \
	num_iter = iter_dim / panel_dim_max + ( iter_dim % panel_dim_max ? 1 : 0 ); \
\
	/* Set the initial values and increments for indices related to C and P
	   based on whether reverse iteration was requested. */ \
	if ( ( revifup && bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) || \
	     ( reviflo && bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) ) \
	{ \
		ic0    = (num_iter - 1) * panel_dim_max; \
		ic_inc = -panel_dim_max; \
		ip0    = num_iter - 1; \
		ip_inc = -1; \
	} \
	else \
	{ \
		ic0    = 0; \
		ic_inc = panel_dim_max; \
		ip0    = 0; \
		ip_inc = 1; \
	} \
\
	p_begin = p_cast; \
\
	for ( ic  = ic0,    ip  = ip0,    it  = 0; it < num_iter; \
	      ic += ic_inc, ip += ip_inc, it += 1 ) \
	{ \
		panel_dim_i = bli_min( panel_dim_max, iter_dim - ic ); \
\
		diagoffc_i  = diagoffc + (ip  )*diagoffc_inc; \
		c_begin     = c_cast   + (ic  )*vs_c; \
\
		if ( bli_is_triangular( strucc ) &&  \
		     bli_is_unstored_subpart_n( diagoffc_i, uploc, *m_panel_full, *n_panel_full ) ) \
		{ \
			/* This case executes if the panel belongs to a triangular
			   matrix AND is completely unstored (ie: zero). If the panel
			   is unstored, we do nothing. (Notice that we don't even
			   increment p_begin.) */ \
\
			continue; \
		} \
		else if ( bli_is_triangular( strucc ) &&  \
		          bli_intersects_diag_n( diagoffc_i, *m_panel_full, *n_panel_full ) ) \
		{ \
			/* This case executes if the panel belongs to a triangular
			   matrix AND is diagonal-intersecting. Notice that we
			   cannot bury the following conditional logic into
			   packm_tri_cxk() because we need to know the value of
			   panel_len_max_i so we can properly increment p_inc. */ \
\
			/* Sanity check. Diagonals should not intersect the short end of
			   a micro-panel. If they do, then somehow the constraints on
			   cache blocksizes being a whole multiple of the register
			   blocksizes was somehow violated. */ \
			if ( ( bli_is_col_stored_f( rs_p, cs_p ) && diagoffc_i < 0 ) || \
			     ( bli_is_row_stored_f( rs_p, cs_p ) && diagoffc_i > 0 ) ) \
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
			if      ( ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) ) || \
			          ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) )  \
			{ \
				panel_off_i     = 0; \
				panel_len_i     = bli_abs( diagoffc_i ) + panel_dim_i; \
				panel_len_max_i = bli_abs( diagoffc_i ) + panel_dim_max; \
				diagoffp_i      = diagoffc_i; \
			} \
			else /* if ( ( bli_is_row_stored_f( rs_p, cs_p ) && bli_is_lower( uploc ) ) || \
			             ( bli_is_col_stored_f( rs_p, cs_p ) && bli_is_upper( uploc ) ) )  */ \
			{ \
				panel_off_i     = bli_abs( diagoffc_i ); \
				panel_len_i     = panel_len_full - panel_off_i; \
				panel_len_max_i = panel_len_max  - panel_off_i; \
				diagoffp_i      = 0; \
			} \
\
			c_use = c_begin + (panel_off_i  )*ldc; \
			p_use = p_begin; \
\
			PASTEMAC(ch,packm_tri_cxk_ri)( strucc, \
			                               diagoffp_i, \
			                               diagc, \
			                               uploc, \
			                               conjc, \
			                               invdiag, \
			                               *m_panel_use, \
			                               *n_panel_use, \
			                               *m_panel_max, \
			                               *n_panel_max, \
			                               kappa_cast, \
			                               c_use, rs_c, cs_c, \
			                               p_use, rs_p, cs_p ); \
\
			p_inc = ldp * panel_len_max_i; \
\
/*
	if ( rs_p == 1 ) { \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: ap_r", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin,         rs_p, cs_p, "%4.1f", "" ); \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: ap_i", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin + p_inc, rs_p, cs_p, "%4.1f", "" ); \
	} \
*/ \
/*
	if ( cs_p == 1 ) { \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: bp_r", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin,         rs_p, cs_p, "%4.1f", "" ); \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: bp_i", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin + p_inc, rs_p, cs_p, "%4.1f", "" ); \
	} \
*/ \
\
		} \
		else if ( bli_is_herm_or_symm( strucc ) ) \
		{ \
			/* This case executes if the panel belongs to a Hermitian or
			   symmetric matrix, which includes stored, unstored, and
			   diagonal-intersecting panels. */ \
\
			panel_len_i     = panel_len_full; \
			panel_len_max_i = panel_len_max; \
\
			PASTEMAC(ch,packm_herm_cxk_ri)( strucc, \
			                                diagoffc_i, \
			                                uploc, \
			                                conjc, \
			                                *m_panel_use, \
			                                *n_panel_use, \
			                                *m_panel_max, \
			                                *n_panel_max, \
			                                kappa_cast, \
			                                c_begin, rs_c, cs_c, \
			                                p_begin, rs_p, cs_p ); \
\
			/* NOTE: This value is equivalent to ps_p. */ \
			p_inc = ldp * panel_len_max_i; \
		} \
		else \
		{ \
			/* This case executes if the panel is general, or, if the
			   panel is part of a triangular matrix and is neither unstored
			   (ie: zero) nor diagonal-intersecting. */ \
\
			panel_len_i     = panel_len_full; \
			panel_len_max_i = panel_len_max; \
\
			PASTEMAC(ch,packm_gen_cxk_ri)( BLIS_GENERAL, \
			                               0, \
			                               BLIS_DENSE, \
			                               conjc, \
			                               *m_panel_use, \
			                               *n_panel_use, \
			                               *m_panel_max, \
			                               *n_panel_max, \
			                               kappa_cast, \
			                               c_begin, rs_c, cs_c, \
			                               p_begin, rs_p, cs_p ); \
\
			/* NOTE: This value is equivalent to ps_p. */ \
			p_inc = ldp * panel_len_max_i; \
\
/*
	if ( cs_p == 1 ) { \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: bp_r", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin,         rs_p, cs_p, "%4.1f", "" ); \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: bp_i", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin + p_inc, rs_p, cs_p, "%4.1f", "" ); \
	} \
*/ \
/*
	if ( rs_p == 1 ) { \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: ap_r", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin,         rs_p, cs_p, "%4.1f", "" ); \
	PASTEMAC(chr,fprintm)( stdout, "packm_var4: ap_i", *m_panel_max, *n_panel_max, \
	                       ( ctype_r* )p_begin + p_inc, rs_p, cs_p, "%4.1f", "" ); \
	} \
*/ \
\
		} \
\
	 	p_begin += p_inc; \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_blk_var4 )

