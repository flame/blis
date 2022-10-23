/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#if 0
#define FUNCPTR_T packm_fp

typedef void (*FUNCPTR_T)
     (
       struc_t strucc,
       doff_t  diagoffc,
       diag_t  diagc,
       uplo_t  uploc,
       trans_t transc,
       pack_t  schema,
       bool    invdiag,
       bool    revifup,
       bool    reviflo,
       dim_t   m,
       dim_t   n,
       dim_t   m_max,
       dim_t   n_max,
       void*   kappa,
       void*   c, inc_t rs_c, inc_t cs_c,
       void*   p, inc_t rs_p, inc_t cs_p,
                  inc_t is_p,
                  dim_t pd_p, inc_t ps_p,
       void_fp packm_ker,
       cntx_t* cntx,
       thrinfo_t* thread
     );

static FUNCPTR_T GENARRAY(ftypes,packm_blk_var1);
#endif


static func_t packm_struc_cxk_kers[BLIS_NUM_PACK_SCHEMA_TYPES] =
{
    /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
// 0000 row/col panels
    { { bli_spackm_struc_cxk,      bli_cpackm_struc_cxk,
        bli_dpackm_struc_cxk,      bli_zpackm_struc_cxk,      } },
// 0001 row/col panels: 1m-expanded (1e)
    { { NULL,                      bli_cpackm_struc_cxk_1er,
        NULL,                      bli_zpackm_struc_cxk_1er,  } },
// 0010 row/col panels: 1m-reordered (1r)
    { { NULL,                      bli_cpackm_struc_cxk_1er,
        NULL,                      bli_zpackm_struc_cxk_1er,  } },
};


void bli_packm_blk_var1
     (
       obj_t*   c,
       obj_t*   p,
       cntx_t*  cntx,
       rntm_t*  rntm,
       cntl_t*  cntl,
       thrinfo_t* thread
     )
{
#ifdef BLIS_ENABLE_GEMM_MD
	// Call a different packm implementation when the storage and target
	// datatypes differ.
	if ( bli_obj_dt( c ) != bli_obj_target_dt( c ) )
	{
		bli_packm_blk_var1_md( c, p, cntx, rntm, cntl, thread );
		return;
	}
#endif

	// Extract various fields from the control tree.
	pack_t schema  = bli_cntl_packm_params_pack_schema( cntl );
	bool   invdiag = bli_cntl_packm_params_does_invert_diag( cntl );
	bool   revifup = bli_cntl_packm_params_rev_iter_if_upper( cntl );
	bool   reviflo = bli_cntl_packm_params_rev_iter_if_lower( cntl );

	// Every thread initializes p and determines the size of memory
	// block needed (which gets embedded into the otherwise "blank" mem_t
	// entry in the control tree node). Return early if no packing is required.
	if ( !bli_packm_init( c, p, cntx, rntm, cntl, thread ) )
		return;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packm_int_check( c, p, cntx );

	num_t     dt_c       = bli_obj_dt( c );
	dim_t     dt_c_size  = bli_dt_size( dt_c );

	num_t     dt_p       = bli_obj_dt( p );
	dim_t     dt_p_size  = bli_dt_size( dt_p );

	struc_t   strucc     = bli_obj_struc( c );
	doff_t    diagoffc   = bli_obj_diag_offset( c );
	diag_t    diagc      = bli_obj_diag( c );
	uplo_t    uploc      = bli_obj_uplo( c );
	trans_t   transc     = bli_obj_conjtrans_status( c );

	dim_t     m          = bli_obj_length( p );
	dim_t     n          = bli_obj_width( p );
	//dim_t     m_max      = bli_obj_padded_length( p );
	dim_t     n_max      = bli_obj_padded_width( p );

	char*     c_cast     = bli_obj_buffer_at_off( c );
	inc_t     rs_c       = bli_obj_row_stride( c );
	inc_t     cs_c       = bli_obj_col_stride( c );

	char*     p_cast     = bli_obj_buffer_at_off( p );
	inc_t     rs_p       = bli_obj_row_stride( p );
	inc_t     cs_p       = bli_obj_col_stride( p );
	inc_t     is_p       = bli_obj_imag_stride( p );
	dim_t     pd_p       = bli_obj_panel_dim( p );
	inc_t     ps_p       = bli_obj_panel_stride( p );

	obj_t     kappa_local;
	char*     kappa_cast = bli_packm_scalar( &kappa_local, p );

	// We use the default lookup table to determine the right func_t
    // for the current schema.
	const dim_t i = bli_pack_schema_index( schema );
	func_t* packm_kers = &packm_struc_cxk_kers[ i ];

	// Query the datatype-specific function pointer from the func_t object.
	void_fp packm_ker = bli_func_get_dt( dt_p, packm_kers );

	packm_ker_vft   packm_ker_cast = packm_ker;

	// -------------------------------------------------------------------------

	/* If C is zeros and part of a triangular matrix, then we don't need
	   to pack it. */
	if ( bli_is_zeros( uploc ) &&
	     bli_is_triangular( strucc ) ) return;

	/* Extract the conjugation bit from the transposition argument. */
	conj_t conjc = bli_extract_conj( transc );

	/* If c needs a transposition, induce it so that we can more simply
	   express the remaining parameters and code. */
	if ( bli_does_trans( transc ) )
	{
		bli_swap_incs( &rs_c, &cs_c );
		bli_negate_diag_offset( &diagoffc );
		bli_toggle_uplo( &uploc );
		bli_toggle_trans( &transc );
	}

	/* Create flags to incidate row or column storage. Note that the
	   schema bit that encodes row or column is describing the form of
	   micro-panel, not the storage in the micro-panel. Hence the
	   mismatch in "row" and "column" semantics. */
/*
	row_stored = bli_is_col_packed( schema );
	col_stored = bli_is_row_packed( schema );
*/
	bool row_stored = FALSE;
	bool col_stored = TRUE;

	dim_t panel_dim_i;
	dim_t panel_len_i;
	dim_t panel_len_max_i;

	/* If the row storage flag indicates row storage, then we are packing
	   to column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */
#if 0
	if ( row_stored )
	{
		/* Prepare to pack to row-stored column panels. */
		iter_dim       = n;
		panel_len_full = m;
		panel_len_max  = m_max;
		panel_dim_max  = pd_p;
		ldc            = rs_c;
		vs_c           = cs_c;
		diagoffc_inc   = -( doff_t )panel_dim_max;
		ldp            = rs_p;
		m_panel_full   = &m;
		n_panel_full   = &panel_dim_i;
		m_panel_use    = &panel_len_i;
		n_panel_use    = &panel_dim_i;
		m_panel_max    = &panel_len_max_i;
		n_panel_max    = &panel_dim_max;
	}
	else
	{
#endif
		/* Prepare to pack to column-stored row panels. */
		dim_t  iter_dim       = m;
		dim_t  panel_len_full = n;
		dim_t  panel_len_max  = n_max;
		dim_t  panel_dim_max  = pd_p;
		inc_t  ldc            = cs_c;
		inc_t  vs_c           = rs_c;
		doff_t diagoffc_inc   = ( doff_t )panel_dim_max;
		inc_t  ldp            = cs_p;
		dim_t* m_panel_full   = &panel_dim_i;
		dim_t* n_panel_full   = &n;
		dim_t* m_panel_use    = &panel_dim_i;
		dim_t* n_panel_use    = &panel_len_i;
		dim_t* m_panel_max    = &panel_dim_max;
		dim_t* n_panel_max    = &panel_len_max_i;
#if 0
	}
#endif

	/* Compute the total number of iterations we'll need. */
	dim_t n_iter = iter_dim / panel_dim_max + ( iter_dim % panel_dim_max ? 1 : 0 );

	/* Set the initial values and increments for indices related to C and P
	   based on whether reverse iteration was requested. */
	dim_t  ic0,    ip0;
	doff_t ic_inc, ip_inc;
	if ( ( revifup && bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) ||
	     ( reviflo && bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) )
	{
		ic0    = (n_iter - 1) * panel_dim_max;
		ic_inc = -panel_dim_max;
		ip0    = n_iter - 1;
		ip_inc = -1;
	}
	else
	{
		ic0    = 0;
		ic_inc = panel_dim_max;
		ip0    = 0;
		ip_inc = 1;
	}

	char* p_begin = p_cast;

	/* Query the number of threads and thread ids from the current thread's
	   packm thrinfo_t node. */
	const dim_t nt  = bli_thread_n_way( thread );
	const dim_t tid = bli_thread_work_id( thread );

	dim_t it_start, it_end, it_inc;

	/* Determine the thread range and increment using the current thread's
	   packm thrinfo_t node. NOTE: The definition of bli_thread_range_jrir()
	   will depend on whether slab or round-robin partitioning was requested
	   at configure-time. */
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

	/* Iterate over every logical micropanel in the source matrix. */
	dim_t ic, ip, it;
	for ( ic  = ic0,    ip  = ip0,    it  = 0; it < n_iter;
	      ic += ic_inc, ip += ip_inc, it += 1 )
	{
		panel_dim_i = bli_min( panel_dim_max, iter_dim - ic );

		doff_t diagoffc_i = diagoffc + (ip  )*diagoffc_inc;
		char*  c_begin    = c_cast   + (ic  )*vs_c*dt_c_size;

		inc_t  p_inc      = ps_p;

		if ( bli_is_triangular( strucc ) && 
		     bli_is_unstored_subpart_n( diagoffc_i, uploc, *m_panel_full, *n_panel_full ) )
		{
			/* This case executes if the panel belongs to a triangular
			   matrix AND is completely unstored (ie: zero). If the panel
			   is unstored, we do nothing. (Notice that we don't even
			   increment p_begin.) */

			continue;
		}
		else if ( bli_is_triangular( strucc ) && 
		          bli_intersects_diag_n( diagoffc_i, *m_panel_full, *n_panel_full ) )
		{
			/* This case executes if the panel belongs to a triangular
			   matrix AND is diagonal-intersecting. Notice that we
			   cannot bury the following conditional logic into
			   packm_struc_cxk() because we need to know the value of
			   panel_len_max_i so we can properly increment p_inc. */

			/* Sanity check. Diagonals should not intersect the short end of
			   a micro-panel. If they do, then somehow the constraints on
			   cache blocksizes being a whole multiple of the register
			   blocksizes was somehow violated. */
			if ( ( col_stored && diagoffc_i < 0 ) ||
			     ( row_stored && diagoffc_i > 0 ) )
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

			dim_t  panel_off_i;
			doff_t diagoffp_i;

			if      ( ( row_stored && bli_is_upper( uploc ) ) ||
			          ( col_stored && bli_is_lower( uploc ) ) ) 
			{
				panel_off_i     = 0;
				panel_len_i     = bli_abs( diagoffc_i ) + panel_dim_i;
				panel_len_max_i = bli_min( bli_abs( diagoffc_i ) + panel_dim_max,
				                           panel_len_max );
				diagoffp_i      = diagoffc_i;
			}
			else /* if ( ( row_stored && bli_is_lower( uploc ) ) ||
			             ( col_stored && bli_is_upper( uploc ) ) )  */
			{
				panel_off_i     = bli_abs( diagoffc_i );
				panel_len_i     = panel_len_full - panel_off_i;
				panel_len_max_i = panel_len_max  - panel_off_i;
				diagoffp_i      = 0;
			}

			char* c_use = c_begin + (panel_off_i  )*ldc*dt_c_size;
			char* p_use = p_begin;

			/* We need to re-compute the imaginary stride as a function of
			   panel_len_max_i since triangular packed matrices have panels
			   of varying lengths. NOTE: This imaginary stride value is
			   only referenced by the packm kernels for induced methods. */
			inc_t is_p_use = ldp * panel_len_max_i;

			/* We nudge the imaginary stride up by one if it is odd. */
			is_p_use += ( bli_is_odd( is_p_use ) ? 1 : 0 );

			/* NOTE: We MUST use round-robin partitioning when packing
			   micropanels of a triangular matrix. Hermitian/symmetric
			   and general packing may use slab or round-robin, depending
			   on which was selected at configure-time. */
			if ( bli_packm_my_iter_rr( it, it_start, it_end, tid, nt ) )
			{
				packm_ker_cast( strucc,
				                diagoffp_i,
				                diagc,
				                uploc,
				                conjc,
				                schema,
				                invdiag,
				                *m_panel_use,
				                *n_panel_use,
				                *m_panel_max,
				                *n_panel_max,
				                kappa_cast,
				                c_use, rs_c, cs_c,
				                p_use, rs_p, cs_p,
			                           is_p_use,
				                cntx );
			}

			/* NOTE: This value is usually LESS than ps_p because triangular
			   matrices usually have several micro-panels that are shorter
			   than a "full" micro-panel. */
			p_inc = is_p_use;
		}
		else if ( bli_is_herm_or_symm( strucc ) )
		{
			/* This case executes if the panel belongs to a Hermitian or
			   symmetric matrix, which includes stored, unstored, and
			   diagonal-intersecting panels. */

			char* c_use = c_begin;
			char* p_use = p_begin;

			panel_len_i     = panel_len_full;
			panel_len_max_i = panel_len_max;

			inc_t is_p_use = is_p;

			/* The definition of bli_packm_my_iter() will depend on whether slab
			   or round-robin partitioning was requested at configure-time. */
			if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) )
			{
				packm_ker_cast( strucc,
				                diagoffc_i,
				                diagc,
				                uploc,
				                conjc,
				                schema,
				                invdiag,
				                *m_panel_use,
				                *n_panel_use,
				                *m_panel_max,
				                *n_panel_max,
				                kappa_cast,
				                c_use, rs_c, cs_c,
				                p_use, rs_p, cs_p,
			                           is_p_use,
				                cntx );
			}
		}
		else
		{
			/* This case executes if the panel is general, or, if the
			   panel is part of a triangular matrix and is neither unstored
			   (ie: zero) nor diagonal-intersecting. */

			char* c_use = c_begin;
			char* p_use = p_begin;

			panel_len_i     = panel_len_full;
			panel_len_max_i = panel_len_max;

			inc_t is_p_use = is_p;

			/* The definition of bli_packm_my_iter() will depend on whether slab
			   or round-robin partitioning was requested at configure-time. */
			if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) )
			{
				packm_ker_cast( BLIS_GENERAL,
				                0,
				                diagc,
				                BLIS_DENSE,
				                conjc,
				                schema,
				                invdiag,
				                *m_panel_use,
				                *n_panel_use,
				                *m_panel_max,
				                *n_panel_max,
				                kappa_cast,
				                c_use, rs_c, cs_c,
				                p_use, rs_p, cs_p,
			                           is_p_use,
				                cntx );
			}
		}

		p_begin += p_inc*dt_p_size;
	}
}




/*
if ( row_stored )
PASTEMAC(ch,fprintm)( stdout, "packm_var2: b", m, n,
                      c_cast,        rs_c, cs_c, "%4.1f", "" );
if ( col_stored )
PASTEMAC(ch,fprintm)( stdout, "packm_var2: a", m, n,
                      c_cast,        rs_c, cs_c, "%4.1f", "" );
*/
/*
if ( row_stored )
PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: b packed", *m_panel_max, *n_panel_max,
                               p_use, rs_p, cs_p, "%5.2f", "" );
else
PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: a packed", *m_panel_max, *n_panel_max,
                               p_use, rs_p, cs_p, "%5.2f", "" );
*/

/*
if ( col_stored ) {
	if ( bli_thread_work_id( thread ) == 0 )
	{
	printf( "packm_blk_var1: thread %lu  (a = %p, ap = %p)\n", bli_thread_work_id( thread ), c_use, p_use );
	fflush( stdout );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: a", *m_panel_use, *n_panel_use,
	                      ( ctype* )c_use,         rs_c, cs_c, "%4.1f", "" );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: ap", *m_panel_max, *n_panel_max,
	                      ( ctype* )p_use,         rs_p, cs_p, "%4.1f", "" );
	fflush( stdout );
	}
bli_thread_barrier( thread );
	if ( bli_thread_work_id( thread ) == 1 )
	{
	printf( "packm_blk_var1: thread %lu  (a = %p, ap = %p)\n", bli_thread_work_id( thread ), c_use, p_use );
	fflush( stdout );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: a", *m_panel_use, *n_panel_use,
	                      ( ctype* )c_use,         rs_c, cs_c, "%4.1f", "" );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: ap", *m_panel_max, *n_panel_max,
	                      ( ctype* )p_use,         rs_p, cs_p, "%4.1f", "" );
	fflush( stdout );
	}
bli_thread_barrier( thread );
}
else {
	if ( bli_thread_work_id( thread ) == 0 )
	{
	printf( "packm_blk_var1: thread %lu  (b = %p, bp = %p)\n", bli_thread_work_id( thread ), c_use, p_use );
	fflush( stdout );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: b", *m_panel_use, *n_panel_use,
	                      ( ctype* )c_use,         rs_c, cs_c, "%4.1f", "" );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: bp", *m_panel_max, *n_panel_max,
	                      ( ctype* )p_use,         rs_p, cs_p, "%4.1f", "" );
	fflush( stdout );
	}
bli_thread_barrier( thread );
	if ( bli_thread_work_id( thread ) == 1 )
	{
	printf( "packm_blk_var1: thread %lu  (b = %p, bp = %p)\n", bli_thread_work_id( thread ), c_use, p_use );
	fflush( stdout );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: b", *m_panel_use, *n_panel_use,
	                      ( ctype* )c_use,         rs_c, cs_c, "%4.1f", "" );
	PASTEMAC(ch,fprintm)( stdout, "packm_blk_var1: bp", *m_panel_max, *n_panel_max,
	                      ( ctype* )p_use,         rs_p, cs_p, "%4.1f", "" );
	fflush( stdout );
	}
bli_thread_barrier( thread );
}
*/
/*
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_rpi", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" );
*/
/*
		if ( row_stored ) {
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )c_use,        2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_i", *m_panel_max, *n_panel_max,
		                       (( ctype_r* )c_use)+rs_c, 2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" );
		inc_t is_b = rs_p * *m_panel_max;
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_i", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use + is_b, rs_p, cs_p, "%4.1f", "" );
		}
*/
/*
		if ( col_stored ) {
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )c_use,        2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_i", *m_panel_max, *n_panel_max,
		                       (( ctype_r* )c_use)+rs_c, 2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use,         rs_p, cs_p, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_i", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use + p_inc, rs_p, cs_p, "%4.1f", "" );
		}
*/
