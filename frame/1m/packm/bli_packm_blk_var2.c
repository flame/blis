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
                           pack_t  schema,
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
                                      inc_t is_p,
                                      dim_t pd_p, inc_t ps_p,
                           void*   packm_ker,
                           packm_thrinfo_t* thread
                         );

//static FUNCPTR_T GENARRAY(ftypes,packm_blk_var2);

extern func_t* packm_struc_cxk_kers;
extern func_t* packm_struc_cxk_4mi_kers;
extern func_t* packm_struc_cxk_3mis_kers;
extern func_t* packm_struc_cxk_rih_kers;


void bli_packm_blk_var2( obj_t*   c,
                         obj_t*   p,
                         packm_thrinfo_t* t )
{
	num_t     dt_cp      = bli_obj_datatype( *c );

	struc_t   strucc     = bli_obj_struc( *c );
	doff_t    diagoffc   = bli_obj_diag_offset( *c );
	diag_t    diagc      = bli_obj_diag( *c );
	uplo_t    uploc      = bli_obj_uplo( *c );
	trans_t   transc     = bli_obj_conjtrans_status( *c );
	pack_t    schema     = bli_obj_pack_schema( *p );
	bool_t    invdiag    = bli_obj_has_inverted_diag( *p );
	bool_t    revifup    = bli_obj_is_pack_rev_if_upper( *p );
	bool_t    reviflo    = bli_obj_is_pack_rev_if_lower( *p );

	dim_t     m_p        = bli_obj_length( *p );
	dim_t     n_p        = bli_obj_width( *p );
	dim_t     m_max_p    = bli_obj_padded_length( *p );
	dim_t     n_max_p    = bli_obj_padded_width( *p );

	void*     buf_c      = bli_obj_buffer_at_off( *c );
	inc_t     rs_c       = bli_obj_row_stride( *c );
	inc_t     cs_c       = bli_obj_col_stride( *c );

	void*     buf_p      = bli_obj_buffer_at_off( *p );
	inc_t     rs_p       = bli_obj_row_stride( *p );
	inc_t     cs_p       = bli_obj_col_stride( *p );
	inc_t     is_p       = bli_obj_imag_stride( *p );
	dim_t     pd_p       = bli_obj_panel_dim( *p );
	inc_t     ps_p       = bli_obj_panel_stride( *p );

	obj_t     kappa;
	obj_t*    kappa_p;
	void*     buf_kappa;

	func_t*   packm_kers;
	void*     packm_ker;

	FUNCPTR_T f;


/*
	// We want this variant to behave identically to that of variant 1
	// in the real domain.
	if ( bli_is_real( dt_cp ) )
	{
		bli_packm_blk_var1( c, p, t );
		return;
	}
*/

	// The value for kappa we use will depend on whether the scalar
	// attached to A has a nonzero imaginary component. If it does,
	// then we will apply the scalar during packing to facilitate
	// implementing induced complex domain algorithms in terms of
	// real domain micro-kernels. (In the aforementioned situation,
	// applying a real scalar is easy, but applying a complex one is
	// harder, so we avoid the need altogether with the code below.)
	if( thread_am_ochief( t ) )
	{
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
	}
	kappa_p = thread_obroadcast( t, kappa_p );


	// Acquire the buffer to the kappa chosen above.
	buf_kappa = bli_obj_buffer_for_1x1( dt_cp, *kappa_p );


	// Choose the correct func_t object based on the pack_t schema.
	if      ( bli_is_4mi_packed( schema ) ) packm_kers = packm_struc_cxk_4mi_kers;
	else if ( bli_is_3mi_packed( schema ) ||
	          bli_is_3ms_packed( schema ) ) packm_kers = packm_struc_cxk_3mis_kers;
	else if ( bli_is_ro_packed( schema ) ||
	          bli_is_io_packed( schema ) ||
	         bli_is_rpi_packed( schema ) )  packm_kers = packm_struc_cxk_rih_kers;
	else                                    packm_kers = packm_struc_cxk_kers;

	// Query the datatype-specific function pointer from the func_t object.
	packm_ker = bli_func_obj_query( dt_cp, packm_kers );


	// Index into the type combination array to extract the correct
	// function pointer.
	//f = ftypes[dt_cp];
	if ( bli_is_scomplex( dt_cp ) ) f = bli_cpackm_blk_var2;
	else                            f = bli_zpackm_blk_var2;

	// Invoke the function.
	f( strucc,
	   diagoffc,
	   diagc,
	   uploc,
	   transc,
	   schema,
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
	          is_p,
	          pd_p, ps_p,
	   packm_ker,
	   t );
}


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kertype ) \
\
void PASTEMAC(ch,varname)( \
                           struc_t strucc, \
                           doff_t  diagoffc, \
                           diag_t  diagc, \
                           uplo_t  uploc, \
                           trans_t transc, \
                           pack_t  schema, \
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
                                      inc_t is_p, \
                                      dim_t pd_p, inc_t ps_p, \
                           void*   packm_ker, \
                           packm_thrinfo_t* thread \
                         ) \
{ \
	PASTECH(ch,kertype) packm_ker_cast = packm_ker; \
\
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
	bool_t          row_stored; \
	bool_t          col_stored; \
	dim_t           ss_num; \
	dim_t           ss_den; \
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
	/* Create flags to incidate row or column storage. Note that the
	   schema bit that encodes row or column is describing the form of
	   micro-panel, not the storage in the micro-panel. Hence the
	   mismatch in "row" and "column" semantics. */ \
	row_stored = bli_is_col_packed( schema ); \
	col_stored = bli_is_row_packed( schema ); \
\
	/* If the row storage flag indicates row storage, then we are packing
	   to column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( row_stored ) \
	{ \
		/* Prepare to pack to row-stored column panels. */ \
		iter_dim       = n; \
		panel_len_full = m; \
		panel_len_max  = m_max; \
		panel_dim_max  = pd_p; \
		ldc            = rs_c; \
		vs_c           = cs_c; \
		diagoffc_inc   = -( doff_t )panel_dim_max; \
		ldp            = rs_p; \
		m_panel_full   = &m; \
		n_panel_full   = &panel_dim_i; \
		m_panel_use    = &panel_len_i; \
		n_panel_use    = &panel_dim_i; \
		m_panel_max    = &panel_len_max_i; \
		n_panel_max    = &panel_dim_max; \
	} \
	else /* if ( col_stored ) */ \
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
	/* Compute the storage stride scaling. Usually this is just 1. However,
	   in the case of interleaved 3m, we need to scale by 3/2, and in the
	   cases of real-only, imag-only, or summed-only, we need to scale by
	   1/2. In both cases, we are compensating for the fact that pointer
	   arithmetic occurs in terms of complex elements rather than real
	   elements. */ \
	if      ( bli_is_3mi_packed( schema ) ) { ss_num = 3; ss_den = 2; } \
	else if ( bli_is_3ms_packed( schema ) ) { ss_num = 1; ss_den = 2; } \
	else if ( bli_is_rih_packed( schema ) ) { ss_num = 1; ss_den = 2; } \
	else                                    { ss_num = 1; ss_den = 1; } \
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
/*
if ( row_stored ) \
PASTEMAC(ch,fprintm)( stdout, "packm_var2: b", m, n, \
                      c_cast,        rs_c, cs_c, "%4.1f", "" ); \
if ( col_stored ) \
PASTEMAC(ch,fprintm)( stdout, "packm_var2: a", m, n, \
                      c_cast,        rs_c, cs_c, "%4.1f", "" ); \
*/ \
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
			   packm_struc_cxk() because we need to know the value of
			   panel_len_max_i so we can properly increment p_inc. */ \
\
			/* Sanity check. Diagonals should not intersect the short end of
			   a micro-panel. If they do, then somehow the constraints on
			   cache blocksizes being a whole multiple of the register
			   blocksizes was somehow violated. */ \
			if ( ( col_stored && diagoffc_i < 0 ) || \
			     ( row_stored && diagoffc_i > 0 ) ) \
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
			if      ( ( row_stored && bli_is_upper( uploc ) ) || \
			          ( col_stored && bli_is_lower( uploc ) ) )  \
			{ \
				panel_off_i     = 0; \
				panel_len_i     = bli_abs( diagoffc_i ) + panel_dim_i; \
				panel_len_max_i = bli_min( bli_abs( diagoffc_i ) + panel_dim_max, \
				                           panel_len_max ); \
				diagoffp_i      = diagoffc_i; \
			} \
			else /* if ( ( row_stored && bli_is_lower( uploc ) ) || \
			             ( col_stored && bli_is_upper( uploc ) ) )  */ \
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
			if( packm_thread_my_iter( it, thread ) ) \
			{ \
				packm_ker_cast( strucc, \
				                diagoffp_i, \
				                diagc, \
				                uploc, \
				                conjc, \
				                schema, \
				                invdiag, \
				                *m_panel_use, \
				                *n_panel_use, \
				                *m_panel_max, \
				                *n_panel_max, \
				                kappa_cast, \
				                c_use, rs_c, cs_c, \
				                p_use, rs_p, cs_p, \
			                           is_p ); \
			} \
\
			/* NOTE: This value is usually LESS than ps_p because triangular
			   matrices usually have several micro-panels that are shorter
			   than a "full" micro-panel. */ \
			p_inc = ldp * panel_len_max_i; \
\
			/* We nudge the panel increment up by one if it is odd. */ \
			p_inc += ( bli_is_odd( p_inc ) ? 1 : 0 ); \
\
			p_inc = ( p_inc * ss_num ) / ss_den; \
		} \
		else if ( bli_is_herm_or_symm( strucc ) ) \
		{ \
			/* This case executes if the panel belongs to a Hermitian or
			   symmetric matrix, which includes stored, unstored, and
			   diagonal-intersecting panels. */ \
\
			c_use = c_begin; \
			p_use = p_begin; \
\
			panel_len_i     = panel_len_full; \
			panel_len_max_i = panel_len_max; \
\
			if( packm_thread_my_iter( it, thread ) ) \
			{ \
				packm_ker_cast( strucc, \
				                diagoffc_i, \
				                diagc, \
				                uploc, \
				                conjc, \
				                schema, \
				                invdiag, \
				                *m_panel_use, \
				                *n_panel_use, \
				                *m_panel_max, \
				                *n_panel_max, \
				                kappa_cast, \
				                c_use, rs_c, cs_c, \
				                p_use, rs_p, cs_p, \
			                           is_p ); \
			} \
\
			/* NOTE: This value is equivalent to ps_p. */ \
			p_inc = ps_p; \
		} \
		else \
		{ \
			/* This case executes if the panel is general, or, if the
			   panel is part of a triangular matrix and is neither unstored
			   (ie: zero) nor diagonal-intersecting. */ \
\
			c_use = c_begin; \
			p_use = p_begin; \
\
			panel_len_i     = panel_len_full; \
			panel_len_max_i = panel_len_max; \
\
			if( packm_thread_my_iter( it, thread ) ) \
			{ \
				packm_ker_cast( BLIS_GENERAL, \
				                0, \
				                diagc, \
				                BLIS_DENSE, \
				                conjc, \
				                schema, \
				                invdiag, \
				                *m_panel_use, \
				                *n_panel_use, \
				                *m_panel_max, \
				                *n_panel_max, \
				                kappa_cast, \
				                c_use, rs_c, cs_c, \
				                p_use, rs_p, cs_p, \
			                           is_p ); \
			} \
\
			/* NOTE: This value is equivalent to ps_p. */ \
			p_inc = ps_p; \
		} \
\
/*
		if ( bli_is_ro_packed( schema ) ) { \
		if ( col_stored ) { \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_r", *m_panel_use, *n_panel_use, \
		                       ( ctype_r* )c_use,         2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" ); \
		} \
		if ( row_stored && *n_panel_use == 3 ) { \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_r", *m_panel_use, *n_panel_use, \
		                       ( ctype_r* )c_use,         2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" ); \
		} \
		} \
*/ \
/*
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_rpi", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" ); \
*/ \
\
\
/*
		if ( row_stored ) { \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )c_use,        2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_i", *m_panel_max, *n_panel_max, \
		                       (( ctype_r* )c_use)+rs_c, 2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" ); \
		inc_t is_b = rs_p * *m_panel_max; \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_i", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use + is_b, rs_p, cs_p, "%4.1f", "" ); \
		} \
*/ \
\
\
/*
		if ( col_stored ) { \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )c_use,        2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_i", *m_panel_max, *n_panel_max, \
		                       (( ctype_r* )c_use)+rs_c, 2*rs_c, 2*cs_c, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_r", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" ); \
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_i", *m_panel_max, *n_panel_max, \
		                       ( ctype_r* )p_use + p_inc, rs_p, cs_p, "%4.1f", "" ); \
		} \
*/ \
\
		p_begin += p_inc; \
\
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_blk_var2, packm_ker_t )

