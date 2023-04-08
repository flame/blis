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

void bli_trmm_rl_ker_var2b
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	const num_t     dt        = bli_obj_exec_dt( c );
	const dim_t     dt_size   = bli_dt_size( dt );

	      doff_t    diagoffb  = bli_obj_diag_offset( b );

	const pack_t    schema_a  = bli_obj_pack_schema( a );
	const pack_t    schema_b  = bli_obj_pack_schema( b );

	      dim_t     m         = bli_obj_length( c );
	      dim_t     n         = bli_obj_width( c );
	      dim_t     k         = bli_obj_width( a );

	const void*     buf_a     = bli_obj_buffer_at_off( a );
	const inc_t     cs_a      = bli_obj_col_stride( a );
	const dim_t     pd_a      = bli_obj_panel_dim( a );
	const inc_t     ps_a      = bli_obj_panel_stride( a );

	const void*     buf_b     = bli_obj_buffer_at_off( b );
	const inc_t     rs_b      = bli_obj_row_stride( b );
	const dim_t     pd_b      = bli_obj_panel_dim( b );
	const inc_t     ps_b      = bli_obj_panel_stride( b );

	      void*     buf_c     = bli_obj_buffer_at_off( c );
	const inc_t     rs_c      = bli_obj_row_stride( c );
	const inc_t     cs_c      = bli_obj_col_stride( c );

	// Detach and multiply the scalars attached to A and B.
	obj_t scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	const void* buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	const void* buf_beta  = bli_obj_internal_scalar_buffer( c );

	// Alias some constants to simpler names.
	const dim_t     MR         = pd_a;
	const dim_t     NR         = pd_b;
	const dim_t     PACKMR     = cs_a;
	const dim_t     PACKNR     = rs_b;

	// Query the context for the micro-kernel address and cast it to its
	// function pointer type.
	gemm_ukr_ft gemm_ukr = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx );

	const void* one        = bli_obj_buffer_for_const( dt, &BLIS_ONE );
	const char* a_cast     = buf_a;
	const char* b_cast     = buf_b;
	      char* c_cast     = buf_c;
	const char* alpha_cast = buf_alpha;
	const char* beta_cast  = buf_beta;

	/*
	   Assumptions/assertions:
	     rs_a == 1
	     cs_a == PACKMR
	     pd_a == MR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKNR
	     cs_b == 1
	     pd_b == NR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/

	// Safety trap: Certain indexing within this macro-kernel does not
	// work as intended if both MR and NR are odd.
	if ( ( bli_is_odd( PACKMR ) && bli_is_odd( NR ) ) ||
	     ( bli_is_odd( PACKNR ) && bli_is_odd( MR ) ) ) bli_abort();

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Safeguard: If the current panel of B is entirely above the diagonal,
	// it is implicitly zero. So we do nothing.
	if ( bli_is_strictly_above_diag_n( diagoffb, k, n ) ) return;

	// If there is a zero region above where the diagonal of B intersects
	// the left edge of the panel, adjust the pointer to A and treat this
	// case as if the diagonal offset were zero. Note that we don't need to
	// adjust the pointer to B since packm would have simply skipped over
	// the region that was not stored. (Note we assume the diagonal offset
	// is a multiple of NR; this assumption will hold as long as the cache
	// blocksizes KC and NC are each a multiple of NR.)
	if ( diagoffb < 0 )
	{
		k        += diagoffb;
		a_cast   -= diagoffb * PACKMR * dt_size;
		diagoffb  = 0;
	}

	// If there is a zero region to the right of where the diagonal
	// of B intersects the bottom of the panel, shrink it to prevent
	// "no-op" iterations from executing.
	if ( diagoffb + k < n )
	{
		n = diagoffb + k;
	}

	// Compute number of primary and leftover components of the m and n
	// dimensions.
	const dim_t n_iter = n / NR + ( n % NR ? 1 : 0 );
	const dim_t n_left = n % NR;

	const dim_t m_iter = m / MR + ( m % MR ? 1 : 0 );
	const dim_t m_left = m % MR;

	// Computing the number of NR x NR tiles in the k dimension is needed
	// when computing the thread ranges below.
	const dim_t k_iter = k / NR + ( k % NR ? 1 : 0 );

	// Determine some increments used to step through A, B, and C.
	const inc_t rstep_a = ps_a * dt_size;

	const inc_t cstep_b = ps_b * dt_size;

	const inc_t rstep_c = rs_c * MR * dt_size;
	const inc_t cstep_c = cs_c * NR * dt_size;

	auxinfo_t aux;

	// Save the pack schemas of A and B to the auxinfo_t object.
	bli_auxinfo_set_schema_a( schema_a, &aux );
	bli_auxinfo_set_schema_b( schema_b, &aux );

	// The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	// loop around the microkernel while the 'caucus' points to the thrinfo_t
	// node for the 1st loop (ir).
	thrinfo_t* thread = bli_thrinfo_sub_node( thread_par );
	//thrinfo_t* caucus = bli_thrinfo_sub_node( thread );

	// Query the number of threads and thread ids for each loop.
#if 0
{
	const dim_t jr_nt  = 17;
	const dim_t jr_tid = jr_nt - 1;

	const doff_t m_iter = 10;
	const doff_t k_iter = 10;
	const doff_t n_iter = 20;

	diagoffb = 30 * NR;
#else
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );
	//const dim_t ir_nt  = bli_thrinfo_n_way( caucus );
	//const dim_t ir_tid = bli_thrinfo_work_id( caucus );
#endif
	dim_t jr_st, ir_st;
	const dim_t n_ut_for_me
	=
	bli_thread_range_tlb_trmm_rl( jr_nt, jr_tid, diagoffb, m_iter, n_iter, k_iter,
	                              MR, NR, &jr_st, &ir_st );

#if 0
	printf( "tid %ld: final range: jr_st, ir_st: %ld %ld  (n_ut_for_me: %ld)\n",
	        jr_tid, jr_st, ir_st, n_ut_for_me );
	return;
}
const dim_t n_ut_for_me = -1; dim_t jr_st, ir_st;
#endif

	// It's possible that there are so few microtiles relative to the number
	// of threads that one or more threads gets no work. If that happens, those
	// threads can return early.
	if ( n_ut_for_me == 0 ) return;

	// Start the jr/ir loops with the current thread's microtile offsets computed
	// by bli_thread_range_tlb_trmm_r().
	dim_t i = ir_st;
	dim_t j = jr_st;

	// Initialize a counter to track the number of microtiles computed by the
	// current thread.
	dim_t ut = 0;

	const char* b1 = b_cast;

	// Get pointers into position by stepping through to the jth micropanel of
	// B and jth microtile of C (within the appropriate row of microtiles).
	for ( dim_t jj = 0; jj < jr_st; ++jj )
	{
		const doff_t diagoffb_jj = diagoffb - ( doff_t )jj*NR;

		if ( bli_intersects_diag_n( diagoffb_jj, k, NR ) )
		{
			// Determine the length of the panel that was packed.
			const dim_t off_b1121 = bli_max( -diagoffb_jj, 0 );
			const dim_t k_b1121   = k - off_b1121;

			// Compute the panel stride for the current diagonal-
			// intersecting micro-panel.
			inc_t ps_b_cur  = k_b1121 * PACKNR;
			      ps_b_cur += ( bli_is_odd( ps_b_cur ) ? 1 : 0 );
			      ps_b_cur *= dt_size;

			b1 += ps_b_cur;
		}
		else if ( bli_is_strictly_below_diag_n( diagoffb_jj, k, NR ) )
		{
			b1 += cstep_b;
		}
	}

	// Loop over the n dimension (NR columns at a time).
	for ( ; true; ++j )
	{
		char* c1 = c_cast + j * cstep_c;

		const doff_t diagoffb_j = diagoffb - ( doff_t )j*NR;

		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Determine the offset to and length of the panel that was packed
		// so we can index into the corresponding location in A.
		const dim_t off_b1121 = bli_max( -diagoffb_j, 0 );
		const dim_t k_b1121   = k - off_b1121;

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		bli_auxinfo_set_next_b( b2, &aux );

		// If the current panel of B intersects the diagonal, scale C
		// by beta. If it is strictly below the diagonal, scale by one.
		// This allows the current macro-kernel to work for both trmm
		// and trmm3.
		if ( bli_intersects_diag_n( diagoffb_j, k, NR ) )
		{
			// Compute the panel stride for the current diagonal-
			// intersecting micro-panel.
			inc_t ps_b_cur  = k_b1121 * PACKNR;
			      ps_b_cur += ( bli_is_odd( ps_b_cur ) ? 1 : 0 );
			      ps_b_cur *= dt_size;

			// Loop over the m dimension (MR rows at a time).
			for ( ; i < m_iter; ++i )
			{
				const char* a1  = a_cast + i * rstep_a;
				      char* c11 = c1     + i * rstep_c;

				const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
				                      ? MR : m_left );

				const char* a1_i = a1 + off_b1121 * PACKMR * dt_size;

				// Compute the addresses of the next panels of A and B.
				const char* a2 = bli_trmm_get_next_a_upanel( a1, rstep_a, 1 );
				if ( bli_is_last_iter_sl( i, m_iter ) )
				{
					a2 = a_cast;
					b2 = bli_trmm_get_next_b_upanel( b1, ps_b_cur, 1 );
					bli_auxinfo_set_next_b( b2, &aux );
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object.
				bli_auxinfo_set_next_a( a2, &aux );

				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  m_cur,
				  n_cur,
				  k_b1121,
				  ( void* )alpha_cast,
				  ( void* )a1_i,
				  ( void* )b1,
				  ( void* )beta_cast,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Increment the microtile counter and check if the thread is done.
				ut += 1; if ( ut == n_ut_for_me ) return;
			}

			// Upon reaching the end of the column of microtiles, reset the ir
			// loop index so that we're ready to start the next pass through the
			// m dimension (i.e., the next jr loop iteration).
			i = 0;

			b1 += ps_b_cur;
		}
		else if ( bli_is_strictly_below_diag_n( diagoffb_j, k, NR ) )
		{
			// Loop over the m dimension (MR rows at a time).
			for ( ; i < m_iter; ++i )
			{
				const char* a1  = a_cast + i * rstep_a;
				      char* c11 = c1     + i * rstep_c;

				const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
				                      ? MR : m_left );

				// Compute the addresses of the next panels of A and B.
				const char* a2 = bli_trmm_get_next_a_upanel( a1, rstep_a, 1 );
				if ( bli_is_last_iter_sl( i, m_iter ) )
				{
					a2 = a_cast;
					b2 = bli_trmm_get_next_b_upanel( b1, cstep_b, 1 );
					bli_auxinfo_set_next_b( b2, &aux );
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object.
				bli_auxinfo_set_next_a( a2, &aux );

				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  m_cur,
				  n_cur,
				  k,
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )one,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Increment the microtile counter and check if the thread is done.
				ut += 1; if ( ut == n_ut_for_me ) return;
			}

			// Upon reaching the end of the column of microtiles, reset the ir
			// loop index so that we're ready to start the next pass through the
			// m dimension (i.e., the next jr loop iteration).
			i = 0;

			b1 += cstep_b;
		}
	}
}

//PASTEMAC(ch,fprintm)( stdout, "trmm_rl_ker_var2: a1", MR, k_b1121, a1, 1, MR, "%4.1f", "" );
//PASTEMAC(ch,fprintm)( stdout, "trmm_rl_ker_var2: b1", k_b1121, NR, b1_i, NR, 1, "%4.1f", "" );

