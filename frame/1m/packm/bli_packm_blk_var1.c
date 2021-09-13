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


static func_t packm_struc_cxk_kers[BLIS_NUM_PACK_SCHEMA_TYPES] =
{
    /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
// 0000 row/col panels
    { { bli_spackm_struc_cxk,      bli_cpackm_struc_cxk,
        bli_dpackm_struc_cxk,      bli_zpackm_struc_cxk,      } },
// 0001 row/col panels: 4m interleaved
    { { NULL,                      bli_cpackm_struc_cxk_4mi,
        NULL,                      bli_zpackm_struc_cxk_4mi,  } },
// 0010 row/col panels: 3m interleaved
    { { NULL,                      bli_cpackm_struc_cxk_3mis,
        NULL,                      bli_zpackm_struc_cxk_3mis, } },
// 0011 row/col panels: 4m separated (NOT IMPLEMENTED)
    { { NULL,                      NULL,
        NULL,                      NULL,                      } },
// 0100 row/col panels: 3m separated
    { { NULL,                      bli_cpackm_struc_cxk_3mis,
        NULL,                      bli_zpackm_struc_cxk_3mis, } },
// 0101 row/col panels: real only
    { { NULL,                      bli_cpackm_struc_cxk_rih,
        NULL,                      bli_zpackm_struc_cxk_rih,  } },
// 0110 row/col panels: imaginary only
    { { NULL,                      bli_cpackm_struc_cxk_rih,
        NULL,                      bli_zpackm_struc_cxk_rih,  } },
// 0111 row/col panels: real+imaginary only
    { { NULL,                      bli_cpackm_struc_cxk_rih,
        NULL,                      bli_zpackm_struc_cxk_rih,  } },
// 1000 row/col panels: 1m-expanded (1e)
    { { NULL,                      bli_cpackm_struc_cxk_1er,
        NULL,                      bli_zpackm_struc_cxk_1er,  } },
// 1001 row/col panels: 1m-reordered (1r)
    { { NULL,                      bli_cpackm_struc_cxk_1er,
        NULL,                      bli_zpackm_struc_cxk_1er,  } },
};

static packm_ker_vft GENARRAY2_ALL(packm_struc_cxk_md,packm_struc_cxk_md);

void bli_packm_blk_var1
     (
       obj_t*   c,
       obj_t*   p,
       cntx_t*  cntx,
       cntl_t*  cntl,
       thrinfo_t* thread
     )
{
	num_t   dt_c           = bli_obj_dt( c );
    dim_t   dt_c_size      = bli_dt_size( dt_c );

	num_t   dt_p           = bli_obj_dt( p );
    dim_t   dt_p_size      = bli_dt_size( dt_p );

	struc_t strucc         = bli_obj_struc( c );
	doff_t  diagoffc       = bli_obj_diag_offset( c );
	diag_t  diagc          = bli_obj_diag( c );
	uplo_t  uploc          = bli_obj_uplo( c );
	trans_t transc         = bli_obj_conjtrans_status( c );
	pack_t  schema         = bli_obj_pack_schema( p );
	bool    invdiag        = bli_obj_has_inverted_diag( p );
	bool    revifup        = bli_obj_is_pack_rev_if_upper( p );
	bool    reviflo        = bli_obj_is_pack_rev_if_lower( p );

	dim_t   iter_dim       = bli_obj_length( p );
	dim_t   panel_len_full = bli_obj_width( p );
	dim_t   panel_len_max  = bli_obj_padded_width( p );

	char*   c_cast         = bli_obj_buffer_at_off( c );
	inc_t   incc           = bli_obj_row_stride( c );
	inc_t   ldc            = bli_obj_col_stride( c );
	dim_t   panel_dim_off  = bli_obj_row_off( c );
	dim_t   panel_len_off  = bli_obj_col_off( c );

	char*   p_cast         = bli_obj_buffer_at_off( p );
	inc_t   ldp            = bli_obj_col_stride( p );
	inc_t   is_p           = bli_obj_imag_stride( p );
	dim_t   panel_dim_max  = bli_obj_panel_dim( p );
	inc_t   ps_p           = bli_obj_panel_stride( p );

	doff_t  diagoffc_inc   = ( doff_t )panel_dim_max;

	/* If C is zeros and part of a triangular matrix, then we don't need
	   to pack it. */
	if ( bli_is_zeros( uploc ) &&
	     bli_is_triangular( strucc ) ) return;

    char* kappa_cast;

	// The value for kappa we use will depends on whether the scalar
	// attached to A has a nonzero imaginary component. If it does,
	// then we will apply the scalar during packing to facilitate
	// implementing induced complex domain algorithms in terms of
	// real domain micro-kernels. (In the aforementioned situation,
	// applying a real scalar is easy, but applying a complex one is
	// harder, so we avoid the need altogether with the code below.)
	if ( bli_obj_scalar_has_nonzero_imag( p ) &&
         !bli_is_nat_packed( schema ) )
	{
		//printf( "applying non-zero imag kappa\n_p" );
	    obj_t kappa;

		// Detach the scalar.
		bli_obj_scalar_detach( p, &kappa );

		// Reset the attached scalar (to 1.0).
		bli_obj_scalar_reset( p );

	    kappa_cast = bli_obj_buffer_for_1x1( dt_p, &kappa );
	}
	// This branch is also for native execution, where we assume that
	// the micro-kernel will always apply the alpha scalar of the
	// higher-level operation. Thus, we use BLIS_ONE for kappa so
	// that the underlying packm implementation does not perform
	// any scaling during packing.
	else
	{
		// If the internal scalar of A has only a real component, then
		// we will apply it later (in the micro-kernel), and so we will
		// use BLIS_ONE to indicate no scaling during packing.
	    kappa_cast = bli_obj_buffer_for_1x1( dt_p, &BLIS_ONE );
	}

	// If the packm structure-aware kernel func_t in the context is
	// NULL (which is the default value after the context is created),
	// we use the default lookup table to determine the right func_t
	// for the current schema.
	func_t* packm_kers = &packm_struc_cxk_kers[ bli_pack_schema_index( schema ) ];

	// Query the datatype-specific function pointer from the func_t object.
	packm_ker_vft packm_ker_cast = bli_func_get_dt( dt_p, packm_kers );

    // For mixed-precision gemm, select the proper kernel (only dense panels).
    if ( dt_c != dt_p )
    {
        packm_ker_cast = packm_struc_cxk_md[ dt_c ][ dt_p ];
    }

    // Query the user-provided packing kernel from the obj_t.
    obj_pack_ukr_fn_t pack_ker_user  = bli_obj_pack_ukr_fn( c );

	/* Extract the conjugation bit from the transposition argument. */
	conj_t conjc = bli_extract_conj( transc );

	/* Compute the storage stride scaling. Usually this is just 1. However,
	   in the case of interleaved 3m, we need to scale by 3/2, and in the
	   cases of real-only, imag-only, or summed-only, we need to scale by
	   1/2. In both cases, we are compensating for the fact that pointer
	   arithmetic occurs in terms of complex elements rather than real
	   elements. */
	dim_t ss_num;
	dim_t ss_den;

	if      ( bli_is_3mi_packed( schema ) ) { ss_num = 3; ss_den = 2; }
	else if ( bli_is_3ms_packed( schema ) ) { ss_num = 1; ss_den = 2; }
	else if ( bli_is_rih_packed( schema ) ) { ss_num = 1; ss_den = 2; }
	else                                    { ss_num = 1; ss_den = 1; }

	/* Compute the total number of iterations we'll need. */
	dim_t n_iter = iter_dim / panel_dim_max + ( iter_dim % panel_dim_max ? 1 : 0 );

	/* Set the initial values and increments for indices related to C and P
	   based on whether reverse iteration was requested. */
	dim_t  ic0, ip0;
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

	/* Query the number of threads and thread ids from the current thread's
	   packm thrinfo_t node. */
	const dim_t nt  = bli_thread_n_way( thread );
	const dim_t tid = bli_thread_work_id( thread );

	/* Determine the thread range and increment using the current thread's
	   packm thrinfo_t node. NOTE: The definition of bli_thread_range_jrir()
	   will depend on whether slab or round-robin partitioning was requested
	   at configure-time. */
	dim_t it_start, it_end, it_inc;
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

	char* p_begin = p_cast;

	/* Iterate over every logical micropanel in the source matrix. */
	for ( dim_t ic  = ic0,    ip  = ip0,    it  = 0; it < n_iter;
	            ic += ic_inc, ip += ip_inc, it += 1 )
	{
		dim_t  panel_dim_i = bli_min( panel_dim_max, iter_dim - ic );

		doff_t diagoffc_i  = diagoffc + (ip  )*diagoffc_inc;
		char*  c_begin     = c_cast   + (ic  )*incc*dt_c_size;

		inc_t  p_inc       = ps_p;

		if ( pack_ker_user )
		{
			/* This case executes if the user has specified a custom packing microkernel */

			dim_t panel_dim_off_i = panel_dim_off + ic;

			/* The definition of bli_packm_my_iter() will depend on whether slab
			   or round-robin partitioning was requested at configure-time. */
			if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) )
			{
				pack_ker_user( panel_dim_i,
				               panel_dim_max,
				               panel_dim_off_i,
				               panel_len_full,
				               panel_len_max,
				               panel_len_off,
				               kappa_cast,
				               c_begin, incc, ldc,
				               p_begin,       ldp,
			                   bli_obj_user_data( c ),
				               cntx );
			}
		}
		else if ( bli_is_triangular( strucc ) &&
		          bli_is_unstored_subpart_n( diagoffc_i, uploc, panel_dim_i, panel_len_full ) )
		{
			/* This case executes if the panel belongs to a triangular
			   matrix AND is completely unstored (ie: zero). If the panel
			   is unstored, we do nothing. (Notice that we don't even
			   increment p_begin.) */

			continue;
		}
		else if ( bli_is_triangular( strucc ) &&
		          bli_intersects_diag_n( diagoffc_i, panel_dim_i, panel_len_full ) )
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
			if ( diagoffc_i < 0 )
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

        	dim_t  panel_off_i;
        	dim_t  panel_len_i;
        	dim_t  panel_len_max_i;
        	doff_t diagoffp_i;

			if ( bli_is_lower( uploc ) )
			{
				panel_off_i     = 0;
				panel_len_i     = bli_abs( diagoffc_i ) + panel_dim_i;
				panel_len_max_i = bli_min( bli_abs( diagoffc_i ) + panel_dim_max,
				                           panel_len_max );
				diagoffp_i      = diagoffc_i;
			}
			else /* if ( bli_is_upper( uploc ) )  */
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
				                panel_dim_i,
				                panel_len_i,
				                panel_dim_max,
				                panel_len_max,
				                kappa_cast,
				                c_use, incc, ldc,
				                p_use,       ldp,
			                           is_p_use,
				                cntx );
			}

			/* NOTE: This value is usually LESS than ps_p because triangular
			   matrices usually have several micro-panels that are shorter
			   than a "full" micro-panel. */
			p_inc = ( is_p_use * ss_num ) / ss_den;
		}
		else if ( bli_is_herm_or_symm( strucc ) )
		{
			/* This case executes if the panel belongs to a Hermitian or
			   symmetric matrix, which includes stored, unstored, and
			   diagonal-intersecting panels. */

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
				                panel_dim_i,
				                panel_len_full,
				                panel_dim_max,
				                panel_len_max,
				                kappa_cast,
				                c_begin, incc, ldc,
				                p_begin,       ldp, is_p,
				                cntx );
			}
		}
		else
		{
			/* This case executes if the panel is general, or, if the
			   panel is part of a triangular matrix and is neither unstored
			   (ie: zero) nor diagonal-intersecting. */

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
				                panel_dim_i,
				                panel_len_full,
				                panel_dim_max,
				                panel_len_max,
				                kappa_cast,
				                c_begin, incc, ldc,
				                p_begin,       ldp, is_p,
				                cntx );
			}
		}

		p_begin += p_inc*dt_p_size;
	}
}



/*
if ( row_stored )
PASTEMAC(ch,fprintm)( stdout, "packm_var2: b", m_p, n_p,
                      c_cast,        rs_c, cs_c, "%4.1f", "" );
if ( col_stored )
PASTEMAC(ch,fprintm)( stdout, "packm_var2: a", m_p, n_p,
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
	printf( "packm_blk_var1: thread %lu  (a = %p, ap = %p)\n_p", bli_thread_work_id( thread ), c_use, p_use );
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
	printf( "packm_blk_var1: thread %lu  (a = %p, ap = %p)\n_p", bli_thread_work_id( thread ), c_use, p_use );
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
	printf( "packm_blk_var1: thread %lu  (b = %p, bp = %p)\n_p", bli_thread_work_id( thread ), c_use, p_use );
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
	printf( "packm_blk_var1: thread %lu  (b = %p, bp = %p)\n_p", bli_thread_work_id( thread ), c_use, p_use );
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
		if ( bli_is_4mi_packed( schema ) ) {
		printf( "packm_var2: is_p_use = %lu\n_p", is_p_use );
		if ( col_stored ) {
		if ( 0 )
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: a_r", *m_panel_use, *n_panel_use,
		                       ( ctype_r* )c_use,         2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use,            rs_p, cs_p, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: ap_i", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use + is_p_use, rs_p, cs_p, "%4.1f", "" );
		}
		if ( row_stored ) {
		if ( 0 )
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: b_r", *m_panel_use, *n_panel_use,
		                       ( ctype_r* )c_use,         2*rs_c, 2*cs_c, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_r", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use,            rs_p, cs_p, "%4.1f", "" );
		PASTEMAC(chr,fprintm)( stdout, "packm_var2: bp_i", *m_panel_max, *n_panel_max,
		                       ( ctype_r* )p_use + is_p_use, rs_p, cs_p, "%4.1f", "" );
		}
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
