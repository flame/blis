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

typedef void (*xpbys_mxn_u_vft)
    (
            doff_t diagoff,
            dim_t  m,
            dim_t  n,
      const void*  x, inc_t rs_x, inc_t cs_x,
      const void*  b,
            void*  y, inc_t rs_y, inc_t cs_y
    );

#undef  GENTFUNC
#define GENTFUNC(ctype,ch,op) \
\
BLIS_INLINE void PASTEMAC(ch,op) \
    ( \
            doff_t diagoff, \
            dim_t  m, \
            dim_t  n, \
      const void*  x, inc_t rs_x, inc_t cs_x, \
      const void*  b, \
            void*  y, inc_t rs_y, inc_t cs_y \
    ) \
{ \
	const ctype* restrict x_cast = x; \
	const ctype* restrict b_cast = b; \
	      ctype* restrict y_cast = y; \
\
	PASTEMAC3(ch,ch,ch,xpbys_mxn_u) \
	( \
	  diagoff, \
	  m, \
	  n, \
	  x_cast, rs_x, cs_x, \
	  b_cast, \
	  y_cast, rs_y,  cs_y \
	); \
}

INSERT_GENTFUNC_BASIC0(xpbys_mxn_u_fn);

static xpbys_mxn_u_vft GENARRAY(xpbys_mxn_u, xpbys_mxn_u_fn);

// -----------------------------------------------------------------------------

void bli_gemmt_u_ker_var2b
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	const num_t  dt_exec   = bli_obj_exec_dt( c );
	const num_t  dt_c      = bli_obj_dt( c );

	      doff_t diagoffc  = bli_obj_diag_offset( c );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	      dim_t  m         = bli_obj_length( c );
	      dim_t  n         = bli_obj_width( c );
	      dim_t  k         = bli_obj_width( a );

	const void*  buf_a     = bli_obj_buffer_at_off( a );
	const inc_t  is_a      = bli_obj_imag_stride( a );
	const dim_t  pd_a      = bli_obj_panel_dim( a );
	const inc_t  ps_a      = bli_obj_panel_stride( a );

	const void*  buf_b     = bli_obj_buffer_at_off( b );
	const inc_t  is_b      = bli_obj_imag_stride( b );
	const dim_t  pd_b      = bli_obj_panel_dim( b );
	const inc_t  ps_b      = bli_obj_panel_stride( b );

	      void*  buf_c     = bli_obj_buffer_at_off( c );
	const inc_t  rs_c      = bli_obj_row_stride( c );
	const inc_t  cs_c      = bli_obj_col_stride( c );

	// Detach and multiply the scalars attached to A and B.
	obj_t scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	const void* buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	const void* buf_beta  = bli_obj_internal_scalar_buffer( c );

	const siz_t dt_size   = bli_dt_size( dt_exec );
	const siz_t dt_c_size = bli_dt_size( dt_c );

	// Alias some constants to simpler names.
	const dim_t MR = pd_a;
	const dim_t NR = pd_b;

	// Query the context for the micro-kernel address and cast it to its
	// function pointer type.
	gemm_ukr_ft     gemm_ukr        = bli_cntx_get_l3_vir_ukr_dt( dt_exec, BLIS_GEMM_UKR, cntx );
	xpbys_mxn_u_vft xpbys_mxn_u_ukr = xpbys_mxn_u[ dt_exec ];

	// Temporary C buffer for edge cases. Note that the strides of this
	// temporary buffer are set so that they match the storage of the
	// original C matrix. For example, if C is column-stored, ct will be
	// column-stored as well.
	      char  ct[ BLIS_STACK_BUF_MAX_SIZE ]
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
	const bool  col_pref    = bli_cntx_ukr_prefers_cols_dt( dt_exec, BLIS_GEMM_VIR_UKR, cntx );
	const inc_t rs_ct       = ( col_pref ? 1 : NR );
	const inc_t cs_ct       = ( col_pref ? MR : 1 );

	const void* zero       = bli_obj_buffer_for_const( dt_exec, &BLIS_ZERO );
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

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Safeguard: If the current panel of C is entirely below the diagonal,
	// it is not stored. So we do nothing.
	if ( bli_is_strictly_below_diag_n( diagoffc, m, n ) ) return;

	// If there is a zero region to the left of where the diagonal of C
	// intersects the top edge of the panel, adjust the pointer to C and B
	// and treat this case as if the diagonal offset were zero.
	// NOTE: It's possible that after this pruning that the diagonal offset
	// is still positive (though it is guaranteed to be less than NR).
	if ( diagoffc > 0 )
	{
		const dim_t jp = diagoffc / NR;
		const dim_t j  = jp * NR;

		n        = n - j;
		diagoffc = diagoffc % NR;
		c_cast   = c_cast + (j  )*cs_c*dt_c_size;
		b_cast   = b_cast + (jp )*ps_b*dt_size;
	}

	// If there is a zero region below where the diagonal of C intersects
	// the right edge of the panel, shrink it to prevent "no-op" iterations
	// from executing.
	if ( -diagoffc + n < m )
	{
		m = -diagoffc + n;
	}

	// Compute number of primary and leftover components of the m and n
	// dimensions.
	const dim_t n_iter = n / NR + ( n % NR ? 1 : 0 );
	const dim_t n_left = n % NR;

	const dim_t m_iter = m / MR + ( m % MR ? 1 : 0 );
	const dim_t m_left = m % MR;

	// Determine some increments used to step through A, B, and C.
	const inc_t rstep_a = ps_a * dt_size;

	const inc_t cstep_b = ps_b * dt_size;

	const inc_t rstep_c = rs_c * MR * dt_c_size;
	const inc_t cstep_c = cs_c * NR * dt_c_size;

	auxinfo_t aux;

	// Save the pack schemas of A and B to the auxinfo_t object.
	bli_auxinfo_set_schema_a( schema_a, &aux );
	bli_auxinfo_set_schema_b( schema_b, &aux );

	// Save the imaginary stride of A and B to the auxinfo_t object.
	bli_auxinfo_set_is_a( is_a, &aux );
	bli_auxinfo_set_is_b( is_b, &aux );

	// The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	// loop around the microkernel. Here we query the thrinfo_t node for the
	// 1st (ir) loop around the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( thread_par );
	//thrinfo_t* caucus = bli_thrinfo_sub_node( thread );

	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );
	//const dim_t ir_nt  = bli_thrinfo_n_way( caucus );
	//const dim_t ir_tid = bli_thrinfo_work_id( caucus );

	// Determine the starting microtile offsets and number of microtiles to
	// compute for each thread. Note that assignment of microtiles is done
	// according to the tlb policy.
	dim_t jr_st, ir_st;
	const dim_t n_ut_for_me
	=
	bli_thread_range_tlb_u( jr_nt, jr_tid, diagoffc, m_iter, n_iter, MR, NR,
	                        &jr_st, &ir_st );

	// It's possible that there are so few microtiles relative to the number
	// of threads that one or more threads gets no work. If that happens, those
	// threads can return early.
	if ( n_ut_for_me == 0 ) return;

	// Start the jr/ir loops with the current thread's microtile offsets computed
	// by bli_thread_range_tlb().
	dim_t i = ir_st;
	dim_t j = jr_st;

	// Initialize a counter to track the number of microtiles computed by the
	// current thread.
	dim_t ut = 0;

	// Loop over the n dimension (NR columns at a time).
	for ( ; true; ++j )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		// Compute the diagonal offset for the column of microtiles at (0,j).
		const doff_t diagoffc_j = diagoffc - ( doff_t )j*NR;

		// Compute the current microtile's width.
		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		bli_auxinfo_set_next_b( b2, &aux );

		// Interior loop over the m dimension (MR rows at a time).
		for ( ; i < m_iter; ++i )
		{
			// Compute the diagonal offset for the microtile at (i,j).
			const doff_t diagoffc_ij = diagoffc_j + ( doff_t )i*MR;

			// Compute the current microtile's length.
			const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
			                      ? MR : m_left );

			// If the diagonal intersects the current MR x NR microtile, we
			// compute it the temporary buffer and then add in the elements
			// on or below the diagonal.
			// Otherwise, if the microtile is strictly above the diagonal,
			// we compute and store as we normally would.
			// And if we're strictly below the diagonal, we simply advance
			// to last microtile before the bottom of the matrix.
			if ( bli_intersects_diag_n( diagoffc_ij, m_cur, n_cur ) )
			{
				const char* a1  = a_cast + i * rstep_a;
				      char* c11 = c1     + i * rstep_c;

				// Compute the addresses of the next panels of A and B.
				const char* a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, 1 );
				if ( bli_is_last_iter_tlb_u( diagoffc_ij, MR, NR ) )
				{
					a2 = bli_gemmt_u_wrap_a_upanel( a_cast, rstep_a, diagoffc_j, MR, NR );
					b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, 1 );
					bli_auxinfo_set_next_b( b2, &aux );
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object.
				bli_auxinfo_set_next_a( a2, &aux );

				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  MR,
				  NR,
				  k,
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )zero,
				  ct, rs_ct, cs_ct,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Scale C and add the result to only the stored part.
				xpbys_mxn_u_ukr
				(
				  diagoffc_ij,
				  m_cur, n_cur,
				  ct,  rs_ct, cs_ct,
				  ( void* )beta_cast,
				  c11, rs_c,  cs_c
				);

				// Increment the microtile counter and check if the thread is done.
				ut += 1;
				if ( ut == n_ut_for_me ) return;
			}
			else if ( bli_is_strictly_above_diag_n( diagoffc_ij, m_cur, n_cur ) )
			{
				const char* a1  = a_cast + i * rstep_a;
				      char* c11 = c1     + i * rstep_c;

				// Compute the addresses of the next panel of A.
				const char* a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, 1 );

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
				  ( void* )beta_cast,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Increment the microtile counter and check if the thread is done.
				ut += 1;
				if ( ut == n_ut_for_me ) return;
			}
			else // if ( bli_is_strictly_below_diag_n( diagoffc_ij, m_cur, n_cur ) )
			{
				// Skip past the microtiles strictly below the diagonal.
				i = m_iter - 1;
			}
		}

		// Upon reaching the end of the column of microtiles, get ready to begin
		// at the beginning of the next column (i.e., the next jr loop iteration).
		i = 0;
	}
}

