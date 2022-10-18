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
      void*  x, inc_t rs_x, inc_t cs_x,
      void*  b,
      void*  y, inc_t rs_y, inc_t cs_y
    );

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
\
void PASTEMAC(ch,op) \
    ( \
      doff_t diagoff, \
      dim_t  m, \
      dim_t  n, \
      void*  x, inc_t rs_x, inc_t cs_x, \
      void*  b, \
      void*  y, inc_t rs_y, inc_t cs_y \
    ) \
{ \
	ctype* restrict x_cast = x; \
	ctype* restrict b_cast = b; \
	ctype* restrict y_cast = y; \
\
	PASTEMAC3(ch,ch,ch,xpbys_mxn_u) \
	( \
      diagoff, \
	  m, n, \
	  x_cast, rs_x, cs_x, \
	  b_cast, \
	  y_cast, rs_y,  cs_y \
	); \
}

INSERT_GENTFUNC_BASIC0(xpbys_mxn_u_fn);

static xpbys_mxn_u_vft GENARRAY(xpbys_mxn_u, xpbys_mxn_u_fn);

void bli_gemmt_u_ker_var2
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
       const cntl_t* cntl,
             thrinfo_t* thread
     )
{
	const num_t  dt        = bli_obj_exec_dt( c );
    const dim_t  dt_size   = bli_dt_size( dt );

	      doff_t diagoffc  = bli_obj_diag_offset( c );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	      dim_t  m         = bli_obj_length( c );
	      dim_t  n         = bli_obj_width( c );
	      dim_t  k         = bli_obj_width( a );

	const void*  buf_a     = bli_obj_buffer_at_off( a );
	const inc_t  cs_a      = bli_obj_col_stride( a );
	const inc_t  is_a      = bli_obj_imag_stride( a );
	const dim_t  pd_a      = bli_obj_panel_dim( a );
	const inc_t  ps_a      = bli_obj_panel_stride( a );

	const void*  buf_b     = bli_obj_buffer_at_off( b );
	const inc_t  rs_b      = bli_obj_row_stride( b );
	const inc_t  is_b      = bli_obj_imag_stride( b );
	const dim_t  pd_b      = bli_obj_panel_dim( b );
	const inc_t  ps_b      = bli_obj_panel_stride( b );

	      void*  buf_c     = bli_obj_buffer_at_off( c );
	const inc_t  rs_c      = bli_obj_row_stride( c );
	const inc_t  cs_c      = bli_obj_col_stride( c );

	// Detach and multiply the scalars attached to A and B.
	obj_t  scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	const void* buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	const void* buf_beta  = bli_obj_internal_scalar_buffer( c );

	/* Alias some constants to simpler names. */
	const dim_t     MR         = pd_a;
	const dim_t     NR         = pd_b;
	/*const dim_t     PACKMR     = cs_a;*/
	/*const dim_t     PACKNR     = rs_b;*/

	/* Query the context for the micro-kernel address and cast it to its
	   function pointer type. */
	gemm_ukr_vft    gemm_ukr        = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx );
    xpbys_mxn_u_vft xpbys_mxn_u_ukr = xpbys_mxn_u[ dt ];

	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */
	char           ct[ BLIS_STACK_BUF_MAX_SIZE ]
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
	const bool      col_pref    = bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_VIR_UKR, cntx );
	const inc_t     rs_ct       = ( col_pref ? 1 : NR );
	const inc_t     cs_ct       = ( col_pref ? MR : 1 );

	const void* zero       = bli_obj_buffer_for_const( dt, &BLIS_ZERO );
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

	/* If any dimension is zero, return immediately. */
	if ( bli_zero_dim3( m, n, k ) ) return;

	/* Safeguard: If the current panel of C is entirely below the diagonal,
	   it is not stored. So we do nothing. */
	if ( bli_is_strictly_below_diag_n( diagoffc, m, n ) ) return;

	/* If there is a zero region to the left of where the diagonal of C
	   intersects the top edge of the panel, adjust the pointer to C and B
	   and treat this case as if the diagonal offset were zero.
	   NOTE: It's possible that after this pruning that the diagonal offset
	   is still positive (though it is guaranteed to be less than NR). */
	if ( diagoffc > 0 )
	{
		dim_t jp       = diagoffc / NR;
		dim_t j        = jp * NR;
		      n        = n - j;
		      diagoffc = diagoffc % NR;
		      c_cast   = c_cast + (j  )*cs_c;
		      b_cast   = b_cast + (jp )*ps_b;
	}

	/* If there is a zero region below where the diagonal of C intersects
	   the right edge of the panel, shrink it to prevent "no-op" iterations
	   from executing. */
	if ( -diagoffc + n < m )
	{
		m = -diagoffc + n;
	}

	/* Compute number of primary and leftover components of the m and n
	   dimensions. */
	dim_t n_iter = n / NR;
	dim_t n_left = n % NR;

	dim_t m_iter = m / MR;
	dim_t m_left = m % MR;

	if ( n_left ) ++n_iter;
	if ( m_left ) ++m_iter;

	/* Determine some increments used to step through A, B, and C. */
	inc_t rstep_a = ps_a;

	inc_t cstep_b = ps_b;

	inc_t rstep_c = rs_c * MR;
	inc_t cstep_c = cs_c * NR;

	/* Save the pack schemas of A and B to the auxinfo_t object. */
    auxinfo_t aux;
	bli_auxinfo_set_schema_a( schema_a, &aux );
	bli_auxinfo_set_schema_b( schema_b, &aux );

	/* Save the imaginary stride of A and B to the auxinfo_t object. */
	bli_auxinfo_set_is_a( is_a, &aux );
	bli_auxinfo_set_is_b( is_b, &aux );

	/* Save the desired output datatype (indicating no typecasting). */
	/*bli_auxinfo_set_dt_on_output( dt, &aux );*/

	/* The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	   loop around the microkernel. Here we query the thrinfo_t node for the
	   1st (ir) loop around the microkernel. */
	thrinfo_t* caucus = bli_thrinfo_sub_node( thread );

	/* Query the number of threads and thread ids for each loop. */
	dim_t jr_nt  = bli_thread_n_way( thread );
	dim_t jr_tid = bli_thread_work_id( thread );
	dim_t ir_nt  = bli_thread_n_way( caucus );
	dim_t ir_tid = bli_thread_work_id( caucus );

	dim_t jr_start, jr_end;
	dim_t ir_start, ir_end;
	dim_t jr_inc,   ir_inc;

	/* Note that we partition the 2nd loop into two regions: the triangular
	   part of C, and the rectangular portion. */
	dim_t n_iter_tri;
	dim_t n_iter_rct;

	if ( bli_is_strictly_above_diag_n( diagoffc, m, n ) )
	{
		/* If the entire panel of C does not intersect the diagonal, there is
		   no triangular region, and therefore we can skip the first set of
		   loops. */
		n_iter_tri = 0;
		n_iter_rct = n_iter;
	}
	else
	{
		/* If the panel of C does intersect the diagonal, compute the number of
		   iterations in the triangular (or trapezoidal) region by dividing NR
		   into the number of rows in C. A non-zero remainder means we need to
		   add one additional iteration. That is, we want the triangular region
		   to contain as few columns of whole microtiles as possible while still
		   including all microtiles that intersect the diagonal. The number of
		   iterations in the rectangular region is computed as the remaining
		   number of iterations in the n dimension. */
		n_iter_tri = ( m + diagoffc ) / NR + ( ( m + diagoffc ) % NR ? 1 : 0 );
		n_iter_rct = n_iter - n_iter_tri;
	}

	/* Use round-robin assignment of micropanels to threads in the 2nd loop
	   and the default (slab or rr) partitioning in the 1st loop for the
	   initial triangular region of C (if it exists). */
	bli_thread_range_jrir_rr( thread, n_iter_tri, 1, FALSE, &jr_start, &jr_end, &jr_inc );
	bli_thread_range_jrir   ( caucus, m_iter,     1, FALSE, &ir_start, &ir_end, &ir_inc );

	/* Loop over the n dimension (NR columns at a time). */
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left );

		/* Initialize our next panel of B to be the current panel of B. */
		const char* b2 = b1;

		/* Interior loop over the m dimension (MR rows at a time). */
		for ( dim_t i = ir_start; i < ir_end; i += ir_inc )
		{
			const char* a1  = a_cast + i * rstep_a;
			      char* c11 = c1     + i * rstep_c;

			/* Compute the diagonal offset for the submatrix at (i,j). */
			doff_t diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR;

			dim_t  m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left );

			/* Compute the addresses of the next panels of A and B. */
			const char* a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc );
			if ( bli_is_last_iter( i, m_iter, ir_tid, ir_nt ) )
			{
				a2 = a_cast;
				b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, jr_inc );
				if ( bli_is_last_iter_rr( j, n_iter, jr_tid, jr_nt ) )
					b2 = b_cast;
			}

			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */
			bli_auxinfo_set_next_a( a2, &aux );
			bli_auxinfo_set_next_b( b2, &aux );

			/* If the diagonal intersects the current MR x NR submatrix, we
			   compute it the temporary buffer and then add in the elements
			   on or below the diagonal.
			   Otherwise, if the submatrix is strictly above the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly below the diagonal, we do nothing and
			   continue. */
			if ( bli_intersects_diag_n( diagoffc_ij, m_cur, n_cur ) )
			{
				/* Invoke the gemm micro-kernel. */
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

				/* Scale C and add the result to only the stored part. */
				xpbys_mxn_u_ukr( diagoffc_ij,
		                         m_cur, n_cur,
		                         ct,  rs_ct, cs_ct,
		                         ( void* )beta_cast,
		                         c11, rs_c,  cs_c );
			}
			else if ( bli_is_strictly_above_diag_n( diagoffc_ij, m_cur, n_cur ) )
			{
				/* Invoke the gemm micro-kernel. */
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
			}
		}
	}

	/* If there is no rectangular region, then we're done. */
	if ( n_iter_rct == 0 ) return;

	/* Determine the thread range and increment for the 2nd loop of the
	   remaining rectangular region of C (and also use default partitioning
	   for the 1st loop).
	   NOTE: The definition of bli_thread_range_jrir() will depend on whether
	   slab or round-robin partitioning was requested at configure-time. */
	bli_thread_range_jrir( thread, n_iter_rct, 1, FALSE, &jr_start, &jr_end, &jr_inc );

	/* Advance the start and end iteration offsets for the rectangular region
	   by the number of iterations used for the triangular region. */
	jr_start += n_iter_tri;
	jr_end   += n_iter_tri;

	/* Loop over the n dimension (NR columns at a time). */
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left );

		/* Initialize our next panel of B to be the current panel of B. */
		const char* b2 = b1;

		/* Interior loop over the m dimension (MR rows at a time). */
		for ( dim_t i = ir_start; i < ir_end; i += ir_inc )
		{
			const char* a1  = a_cast + i * rstep_a;
			      char* c11 = c1     + i * rstep_c;

			/* No need to compute the diagonal offset for the rectangular
			   region. */
			/*diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR;*/

			dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left );

			/* Compute the addresses of the next panels of A and B. */
			const char* a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc );
			if ( bli_is_last_iter( i, m_iter, ir_tid, ir_nt ) )
			{
				a2 = a_cast;
				b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, jr_inc );
				if ( bli_is_last_iter( j, n_iter, jr_tid, jr_nt ) )
					b2 = b_cast;
			}

			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */
			bli_auxinfo_set_next_a( a2, &aux );
			bli_auxinfo_set_next_b( b2, &aux );

			/* If the diagonal intersects the current MR x NR submatrix, we
			   compute it the temporary buffer and then add in the elements
			   on or below the diagonal.
			   Otherwise, if the submatrix is strictly above the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly below the diagonal, we do nothing and
			   continue. */
			{
				/* Invoke the gemm micro-kernel. */
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
			}
		}
	}
}

