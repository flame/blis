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

typedef void (*xpbys_mxn_vft)
    (
            dim_t m,
            dim_t n,
      const void* x, inc_t rs_x, inc_t cs_x,
      const void* b,
            void* y, inc_t rs_y, inc_t cs_y
    );

#undef  GENTFUNC2
#define GENTFUNC2(ctypex,ctypey,chx,chy,op) \
\
BLIS_INLINE void PASTEMAC2(chx,chy,op) \
    ( \
            dim_t m, \
            dim_t n, \
      const void* x, inc_t rs_x, inc_t cs_x, \
      const void* b, \
            void* y, inc_t rs_y, inc_t cs_y \
    ) \
{ \
	const ctypex* restrict x_cast = x; \
	const ctypey* restrict b_cast = b; \
	      ctypey* restrict y_cast = y; \
\
	PASTEMAC3(chx,chy,chy,xpbys_mxn) \
	( \
	  m, n, \
	  x_cast, rs_x, cs_x, \
	  b_cast, \
	  y_cast, rs_y,  cs_y \
	); \
}

INSERT_GENTFUNC2_BASIC(xpbys_mxn_fn);
INSERT_GENTFUNC2_MIX_DP(xpbys_mxn_fn);

static xpbys_mxn_vft GENARRAY2_ALL(xpbys_mxn, xpbys_mxn_fn);


void bli_gemm_ker_var2
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	      num_t  dt_exec   = bli_obj_exec_dt( c );
	      num_t  dt_c      = bli_obj_dt( c );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	      dim_t  m         = bli_obj_length( c );
	      dim_t  n         = bli_obj_width( c );
	      dim_t  k         = bli_obj_width( a );

	const char*  a_cast    = bli_obj_buffer_at_off( a );
	const inc_t  is_a      = bli_obj_imag_stride( a );
	      dim_t  pd_a      = bli_obj_panel_dim( a );
	      inc_t  ps_a      = bli_obj_panel_stride( a );

	const char*  b_cast    = bli_obj_buffer_at_off( b );
	const inc_t  is_b      = bli_obj_imag_stride( b );
	      dim_t  pd_b      = bli_obj_panel_dim( b );
	      inc_t  ps_b      = bli_obj_panel_stride( b );

	      char*  c_cast    = bli_obj_buffer_at_off( c );
	      inc_t  rs_c      = bli_obj_row_stride( c );
	      inc_t  cs_c      = bli_obj_col_stride( c );

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Detach and multiply the scalars attached to A and B.
	// NOTE: We know that the internal scalars of A and B are already of the
	// target datatypes because the necessary typecasting would have already
	// taken place during bli_packm_init().
	obj_t scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	// NOTE: We know that scalar_b is of type dt_exec due to the above code
	// that casts the scalars of A and B to dt_exec via scalar_a and scalar_b,
	// and we know that the internal scalar in C is already of the type dt_c
	// due to the casting in the implementation of bli_obj_scalar_attach().
	const char* alpha_cast = bli_obj_internal_scalar_buffer( &scalar_b );
	const char* beta_cast  = bli_obj_internal_scalar_buffer( c );

#if 1
	// Under certain conditions, we can avoid the overhead of calling the 1m
	// virtual microkernel by having the real-domain macrokernel execute with
	// the real-domain microkernel. (See the function definition for details.)
	if ( bli_cntx_method( cntx ) == BLIS_1M )
	{
		bli_gemm_ind_recast_1m_params
		(
		  &dt_exec,
		  &dt_c,
		  schema_a,
		  c,
		  &m, &n, &k,
		  &pd_a, &ps_a,
		  &pd_b, &ps_b,
		  &rs_c, &cs_c,
		  cntx
		);
	}
#endif

#ifdef BLIS_ENABLE_GEMM_MD
	// Tweak parameters in select mixed domain cases (rcc, crc, ccr).
	if ( bli_cntx_method( cntx ) == BLIS_NAT )
	{
		bli_gemm_md_ker_var2_recast
		(
		  &dt_exec,
		  bli_obj_dt( a ),
		  bli_obj_dt( b ),
		  &dt_c,
		  &m, &n, &k,
		  &pd_a, &ps_a,
		  &pd_b, &ps_b,
		  c,
		  &rs_c, &cs_c
		);
	}
#endif

	const siz_t dt_size   = bli_dt_size( dt_exec );
	const siz_t dt_c_size = bli_dt_size( dt_c );

	// Alias some constants to simpler names.
	const dim_t MR = pd_a;
	const dim_t NR = pd_b;

	// Query the context for the micro-kernel address and cast it to its
	// function pointer type.
	gemm_ukr_ft gemm_ukr = bli_cntx_get_l3_vir_ukr_dt( dt_exec, BLIS_GEMM_UKR, cntx );

	// Query the params field from the obj_t. If it is non-NULL, grab the ukr
	// field of the params struct. If that function pointer is non-NULL, use it
	// as our microkernel instead of the default microkernel queried from the
	// cntx above.
	const gemm_ker_params_t* params = bli_obj_ker_params( c );
	gemm_ukr_ft user_ukr = params ? params->ukr : NULL;
	if ( user_ukr ) gemm_ukr = user_ukr;

	// Temporary C buffer for edge cases. Note that the strides of this
	// temporary buffer are set so that they match the storage of the
	// original C matrix. For example, if C is column-stored, ct will be
	// column-stored as well.
	char        ct[ BLIS_STACK_BUF_MAX_SIZE ]
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
	const bool  col_pref    = bli_cntx_ukr_prefers_cols_dt( dt_exec, BLIS_GEMM_VIR_UKR, cntx );
	const inc_t rs_ct       = ( col_pref ? 1 : NR );
	const inc_t cs_ct       = ( col_pref ? MR : 1 );
	const char* zero        = bli_obj_buffer_for_const( dt_exec, &BLIS_ZERO );

	//
	// Assumptions/assertions:
	//   rs_a == 1
	//   cs_a == PACKMR
	//   pd_a == MR
	//   ps_a == stride to next micro-panel of A
	//   rs_b == PACKNR
	//   cs_b == 1
	//   pd_b == NR
	//   ps_b == stride to next micro-panel of B
	//   rs_c == (no assumptions)
	//   cs_c == (no assumptions)
	//

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

	// Save the virtual microkernel address and the params.
	bli_auxinfo_set_ukr( gemm_ukr, &aux );
	bli_auxinfo_set_params( params, &aux );

	dim_t jr_start, jr_end, jr_inc;
	dim_t ir_start, ir_end, ir_inc;

#ifdef BLIS_ENABLE_JRIR_TLB

	// Query the number of threads and thread ids for the jr loop around
	// the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( thread_par );
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );

	const dim_t ir_nt  = 1;
	const dim_t ir_tid = 0;

	dim_t n_ut_for_me
	=
	bli_thread_range_tlb_d( jr_nt, jr_tid, m_iter, n_iter, MR, NR,
	                        &jr_start, &ir_start );

	// Always increment by 1 in both dimensions.
	jr_inc = 1;
	ir_inc = 1;

	// Each thread iterates over the entire panel of C until it exhausts its
	// assigned set of microtiles.
	jr_end = n_iter;
	ir_end = m_iter;

	// Successive iterations of the ir loop should start at 0.
	const dim_t ir_next = 0;

#else // ifdef ( _SLAB || _RR )

	// Query the number of threads and thread ids for the ir loop around
	// the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( thread_par );
	thrinfo_t* caucus = bli_thrinfo_sub_node( thread );
	const dim_t ir_nt  = bli_thrinfo_n_way( caucus );
	const dim_t ir_tid = bli_thrinfo_work_id( caucus );

	// Determine the thread range and increment for the 2nd and 1st loops.
	// NOTE: The definition of bli_thread_range_slrr() will depend on whether
	// slab or round-robin partitioning was requested at configure-time.
	bli_thread_range_slrr( thread, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );
	bli_thread_range_slrr( caucus, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc );

	// Calculate the total number of microtiles assigned to this thread.
	dim_t n_ut_for_me = ( ( ir_end + ir_inc - 1 - ir_start ) / ir_inc ) *
	                    ( ( jr_end + jr_inc - 1 - jr_start ) / jr_inc );

	// Each succesive iteration of the ir loop always starts at ir_start.
	const dim_t ir_next = ir_start;

#endif

	// It's possible that there are so few microtiles relative to the number
	// of threads that one or more threads gets no work. If that happens, those
	// threads can return early.
	if ( n_ut_for_me == 0 ) return;

	// Loop over the n dimension (NR columns at a time).
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		// Compute the current microtile's width.
		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		// Loop over the m dimension (MR rows at a time).
		for ( dim_t i = ir_start; i < ir_end; i += ir_inc )
		{
			const char* a1  = a_cast + i * rstep_a;
			      char* c11 = c1     + i * rstep_c;

			// Compute the current microtile's length.
			const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
			                      ? MR : m_left );

			// Compute the addresses of the next panels of A and B.
			const char* a2 = bli_gemm_get_next_a_upanel( a1, rstep_a, ir_inc );
			if ( bli_is_last_iter_slrr( i, ir_end, ir_tid, ir_nt ) )
			{
				a2 = a_cast;
				b2 = bli_gemm_get_next_b_upanel( b1, cstep_b, jr_inc );
			}

			// Save addresses of next panels of A and B to the auxinfo_t
			// object.
			bli_auxinfo_set_next_a( a2, &aux );
			bli_auxinfo_set_next_b( b2, &aux );

			// Edge case handling now occurs within the microkernel itself, but
			// we must still explicitly accumulate to a temporary microtile in
			// situations where a virtual microkernel is being used, such as
			// during the 1m method or some cases of mixed datatypes.
			if ( dt_exec == dt_c )
			{
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
			}
			else
			{
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
				           &ct, rs_ct, cs_ct,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Accumulate to C with typecasting.
				xpbys_mxn[ dt_exec ][ dt_c ]
				(
				  m_cur, n_cur,
				  &ct, rs_ct, cs_ct,
				  ( void* )beta_cast,
				  c11, rs_c, cs_c
				);
			}

			// Decrement the number of microtiles assigned to the thread; once
			// it reaches zero, return immediately.
			n_ut_for_me -= 1; if ( n_ut_for_me == 0 ) return;
		}

		ir_start = ir_next;
	}
}

//PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2: b1", k, NR, b1, NR, 1, "%4.1f", "" );
//PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2: a1", MR, k, a1, 1, MR, "%4.1f", "" );
//PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2: c after", m_cur, n_cur, c11, rs_c, cs_c, "%4.1f", "" );

