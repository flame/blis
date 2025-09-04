/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

//
// -- var1n --------------------------------------------------------------------
//

void bli_gemmsup_ref_var1n
     (
             trans_t trans,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
             stor3_t stor_id,
       const cntx_t* cntx,
       const rntm_t* rntm,
             thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);
#if 0
	obj_t at, bt;

	bli_obj_alias_to( a, &at );
	bli_obj_alias_to( b, &bt );

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( &at ) ) { bli_obj_induce_fast_trans( &at ); }
	if ( bli_obj_has_trans( &bt ) ) { bli_obj_induce_fast_trans( &bt ); }

	const num_t    dt        = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );

	const dim_t    k         = bli_obj_width( &at );

	void* restrict buf_a     = bli_obj_buffer_at_off( &at );
	const inc_t    rs_a      = bli_obj_row_stride( &at );
	const inc_t    cs_a      = bli_obj_col_stride( &at );

	const dim_t  dt_size = bli_dt_size( dt );

	      bool   packa   = bli_rntm_pack_a( rntm );
	      bool   packb   = bli_rntm_pack_b( rntm );

	      conj_t conja   = bli_obj_conj_status( a );
	      conj_t conjb   = bli_obj_conj_status( b );

	      dim_t  m       = bli_obj_length( c );
	      dim_t  n       = bli_obj_width( c );
	      dim_t  k;

	const void*  buf_a   = bli_obj_buffer_at_off( a );
	      inc_t  rs_a;
	      inc_t  cs_a;

	const void*  buf_b   = bli_obj_buffer_at_off( b );
	      inc_t  rs_b;
	      inc_t  cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}

	      void* buf_c     = bli_obj_buffer_at_off( c );
	      inc_t rs_c      = bli_obj_row_stride( c );
	      inc_t cs_c      = bli_obj_col_stride( c );

	const void* buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	const void* buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#if 1
	// Optimize some storage/packing cases by transforming them into others.
	// These optimizations are expressed by changing trans and/or stor_id.
	bli_gemmsup_ref_var1n2m_opt_cases( dt, &trans, packa, packb, &stor_id, cntx );
#endif

	// Note: This code explicitly performs the swaps that could be done
	// implicitly in other BLIS contexts where a type-specific helper function
	// was being called.
	if ( bli_is_trans( trans ) )
	{
		      bool   packtmp = packa; packa = packb; packb = packtmp;
		      conj_t conjtmp = conja; conja = conjb; conjb = conjtmp;
		      dim_t  len_tmp =     m;     m =     n;     n = len_tmp;
		const void*  buf_tmp = buf_a; buf_a = buf_b; buf_b = buf_tmp;
		      inc_t  str_tmp =  rs_a;  rs_a =  cs_b;  cs_b = str_tmp;
		             str_tmp =  cs_a;  cs_a =  rs_b;  rs_b = str_tmp;
		             str_tmp =  rs_c;  rs_c =  cs_c;  cs_c = str_tmp;

		stor_id = bli_stor3_trans( stor_id );
	}

	// This transposition of the stor3_t id value is inherent to variant 1.
	// The reason: we assume that variant 2 is the "main" variant. The
	// consequence of this is that we assume that the millikernels that
	// iterate over m are registered to the "primary" kernel group associated
	// with the kernel IO preference; similarly, mkernels that iterate over
	// n are assumed to be registered to the "non-primary" group associated
	// with the ("non-primary") anti-preference. Note that this pattern holds
	// regardless of whether the mkernel set has a row or column preference.)
	// See bli_l3_sup_int.c for a higher-level view of how this choice is made.
	stor_id = bli_stor3_trans( stor_id );

	// Query the context for various blocksizes.
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t NC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
	const dim_t MC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );

	// Disable modification of KC since it seems to negatively impact certain
	// operations (#644).
	dim_t KC = KC0;

	/*
	if      ( packa && packb )
	{
		KC = KC0;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

	// Nudge NC up to a multiple of MR and MC up to a multiple of NR.
	// NOTE: This is unique to variant 1 (ie: not performed in variant 2)
	// because MC % MR == 0 and NC % NR == 0 is already enforced at runtime.
	const dim_t NC  = bli_align_dim_to_mult( NC0, MR );
	const dim_t MC  = bli_align_dim_to_mult( MC0, NR );

	// Query the maximum blocksize for MR, which implies a maximum blocksize
	// extension for the final iteration.
	const dim_t MRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_MR, cntx );
	const dim_t MRE = MRM - MR;

	// Compute partitioning step values for each matrix of each loop.
	const inc_t jcstep_c = rs_c * dt_size;
	const inc_t jcstep_a = rs_a * dt_size;

	const inc_t pcstep_a = cs_a * dt_size;
	const inc_t pcstep_b = rs_b * dt_size;

	const inc_t icstep_c = cs_c * dt_size;
	const inc_t icstep_b = cs_b * dt_size;

	const inc_t jrstep_c = rs_c * MR * dt_size;

void bli_gemmsup_ref_var2m
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       stor3_t eff_id,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);
#if 0
	obj_t at, bt;

	// Query the context for the sup microkernel address and cast it to its
	// function pointer type.
	gemmsup_ker_ft gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );

	const char* a_00       = buf_a;
	const char* b_00       = buf_b;
	      char* c_00       = buf_c;
	const void* one        = bli_obj_buffer_for_const( dt, &BLIS_ONE );

	auxinfo_t aux;

	// Determine whether we are using more than one thread.
	const bool is_mt = ( bli_rntm_calc_num_threads( rntm ) > 1 );

	thrinfo_t* thread_jc = bli_thrinfo_sub_node( thread );
	thrinfo_t* thread_pc = bli_thrinfo_sub_node( thread_jc );
	thrinfo_t* thread_pa = bli_thrinfo_sub_node( thread_pc );
	thrinfo_t* thread_ic = bli_thrinfo_sub_node( thread_pa );
	thrinfo_t* thread_pb = bli_thrinfo_sub_node( thread_ic );
	thrinfo_t* thread_jr = bli_thrinfo_sub_node( thread_pb );

	// Compute the JC loop thread range for the current thread.
	dim_t jc_start, jc_end;
	bli_thread_range_sub( thread_jc, m, MR, FALSE, &jc_start, &jc_end );
	const dim_t m_local = jc_end - jc_start;

	// Compute number of primary and leftover components of the JC loop.
	//const dim_t jc_iter = ( m_local + NC - 1 ) / NC;
	const dim_t jc_left =   m_local % NC;

	// Loop over the m dimension (NC rows/columns at a time).
	//for ( dim_t jj = 0; jj < jc_iter; jj += 1 )
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		// Calculate the thread's current JC block dimension.
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		const char* a_jc = a_00 + jj * jcstep_a;
		      char* c_jc = c_00 + jj * jcstep_c;

		// Compute the PC loop thread range for the current thread.
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		// Compute number of primary and leftover components of the PC loop.
		//const dim_t pc_iter = ( k_local + KC - 1 ) / KC;
		const dim_t pc_left =   k_local % KC;

		// Loop over the k dimension (KC rows/columns at a time).
		//for ( dim_t pp = 0; pp < pc_iter; pp += 1 )
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			// Calculate the thread's current PC block dimension.
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			const char* a_pc = a_jc + pp * pcstep_a;
			const char* b_pc = b_00 + pp * pcstep_b;

			// Only apply beta to the first iteration of the pc loop.
			const void* beta_use = ( pp == 0 ? buf_beta : one );

			      char* a_use;
			      inc_t rs_a_use, cs_a_use, ps_a_use;

			// Determine the packing buffer and related parameters for matrix
			// A. (If A will not be packed, then a_use will be set to point to
			// a and the _a_use strides will be set accordingly.) Then call
			// the packm sup variant chooser, which will call the appropriate
			// implementation based on the schema deduced from the stor_id.
			// NOTE: packing matrix A in this panel-block algorithm corresponds
			// to packing matrix B in the block-panel algorithm.
			bli_packm_sup
			(
			  packa,
			  BLIS_BUFFER_FOR_B_PANEL, // This algorithm packs matrix A to
			  stor_id,                 // a "panel of B".
			  BLIS_NO_TRANSPOSE,
			  dt,
			  NC,     KC,       // This "panel of B" is (at most) NC x KC.
			  nc_cur, kc_cur, MR,
			  one,
			  a_pc,   rs_a,      cs_a,
			  ( void** )&a_use, &rs_a_use, &cs_a_use,
			                    &ps_a_use,
			  cntx,
			  thread_pa
			);

			// Alias a_use so that it's clear this is our current block of
			// matrix A.
			const char* a_pc_use = a_use;

			// We don't need to embed the panel stride of A within the auxinfo_t
			// object because this variant iterates through A in the jr loop,
			// which occurs here, within the macrokernel, not within the
			// millikernel.
			//bli_auxinfo_set_ps_a( ps_a_use, &aux );

			// Compute the IC loop thread range for the current thread.
			dim_t ic_start, ic_end;
			bli_thread_range_sub( thread_ic, n, NR, FALSE, &ic_start, &ic_end );
			const dim_t n_local = ic_end - ic_start;

			// Compute number of primary and leftover components of the IC loop.
			//const dim_t ic_iter = ( n_local + MC - 1 ) / MC;
			const dim_t ic_left =   n_local % MC;

			// Loop over the n dimension (MC rows at a time).
			//for ( dim_t ii = 0; ii < ic_iter; ii += 1 )
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				// Calculate the thread's current IC block dimension.
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );

				const char* b_ic = b_pc + ii * icstep_b;
				      char* c_ic = c_jc + ii * icstep_c;

				      char* b_use;
				      inc_t rs_b_use, cs_b_use, ps_b_use;

				// Determine the packing buffer and related parameters for matrix
				// B. (If B will not be packed, then b_use will be set to point to
				// b and the _b_use strides will be set accordingly.) Then call
				// the packm sup variant chooser, which will call the appropriate
				// implementation based on the schema deduced from the stor_id.
				// NOTE: packing matrix B in this panel-block algorithm corresponds
				// to packing matrix A in the block-panel algorithm.
				bli_packm_sup
				(
				  packb,
				  BLIS_BUFFER_FOR_A_BLOCK, // This algorithm packs matrix B to
				  stor_id,                 // a "block of A".
				  BLIS_NO_TRANSPOSE,
				  dt,
				  MC,     KC,       // This "block of A" is (at most) KC x MC.
				  mc_cur, kc_cur, NR,
				  one,
				  b_ic,   cs_b,      rs_b,
				  ( void** )&b_use, &cs_b_use, &rs_b_use,
				                    &ps_b_use,
				  cntx,
				  thread_pb
				);

				// Alias b_use so that it's clear this is our current block of
				// matrix B.
				const char* b_ic_use = b_use;

				// Embed the panel stride of B within the auxinfo_t object. The
				// millikernel will query and use this to iterate through
				// micropanels of B.
				bli_auxinfo_set_ps_b( ps_b_use, &aux );

				// Compute number of primary and leftover components of the JR loop.
				dim_t jr_iter = ( nc_cur + MR - 1 ) / MR;
				dim_t jr_left =   nc_cur % MR;

				// An optimization: allow the last jr iteration to contain up to MRE
				// rows of C and A. (If MRE > MR, the mkernel has agreed to handle
				// these cases.) Note that this prevents us from declaring jr_iter and
				// jr_left as const. NOTE: We forgo this optimization when packing A
				// since packing an extended edge case is not yet supported.
				if ( !packa && !is_mt )
				if ( MRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= MRE )
				{
					jr_iter--; jr_left += MR;
				}

				// Compute the JR loop thread range for the current thread.
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				// Loop over the m dimension (NR columns at a time).
				//for ( dim_t j = 0; j < jr_iter; j += 1 )
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? MR : jr_left );

					//ctype* a_jr = a_pc + j * jrstep_a;
					const char* a_jr = a_pc_use + j * ps_a_use * dt_size;
					      char* c_jr = c_ic     + j * jrstep_c;

					//const dim_t ir_iter = ( mc_cur + NR - 1 ) / NR;
					//const dim_t ir_left =   mc_cur % NR;

					// Loop over the n dimension (MR rows at a time).
					{
						// Invoke the gemmsup millikernel.
						gemmsup_ker
						(
						  conja,
						  conjb,
						  nr_cur, // Notice: nr_cur <= MR.
						  mc_cur, // Recall: mc_cur partitions the n dimension!
						  kc_cur,
						  ( void* )buf_alpha,
						  ( void* )a_jr,     rs_a_use, cs_a_use,
						  ( void* )b_ic_use, rs_b_use, cs_b_use,
						  ( void* )beta_use,
						  ( void* )c_jr,     rs_c,     cs_c,
						  &aux,
						  ( cntx_t* )cntx
						);
					}
				}
			}

			// NOTE: This barrier is only needed if we are packing A (since
			// that matrix is packed within the pc loop of this variant).
			if ( packa ) bli_thrinfo_barrier( thread_pa );
		}
	}

	// Release any memory that was acquired for packing matrices A and B.
	bli_packm_sup_finalize_mem
	(
	  packa,
	  thread_pa
	);
	bli_packm_sup_finalize_mem
	(
	  packb,
	  thread_pb
	);

/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}


//
// -- var2m --------------------------------------------------------------------
//

void bli_gemmsup_ref_var2m
     (
             trans_t    trans,
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
             stor3_t    stor_id,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     )
{
	const num_t  dt      = bli_obj_dt( c );
	const dim_t  dt_size = bli_dt_size( dt );

	      bool   packa   = bli_rntm_pack_a( rntm );
	      bool   packb   = bli_rntm_pack_b( rntm );

	      conj_t conja   = bli_obj_conj_status( a );
	      conj_t conjb   = bli_obj_conj_status( b );

	      dim_t  m       = bli_obj_length( c );
	      dim_t  n       = bli_obj_width( c );
	      dim_t  k;

	const void*  buf_a   = bli_obj_buffer_at_off( a );
	      inc_t  rs_a;
	      inc_t  cs_a;

	const void*  buf_b   = bli_obj_buffer_at_off( b );
	      inc_t  rs_b;
	      inc_t  cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}

	      void* buf_c     = bli_obj_buffer_at_off( c );
	      inc_t rs_c      = bli_obj_row_stride( c );
	      inc_t cs_c      = bli_obj_col_stride( c );

	const void* buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	const void* buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#if 1
	// Optimize some storage/packing cases by transforming them into others.
	// These optimizations are expressed by changing trans and/or stor_id.
	bli_gemmsup_ref_var1n2m_opt_cases( dt, &trans, packa, packb, &stor_id, cntx );
#endif

	// Note: This code explicitly performs the swaps that could be done
	// implicitly in other BLIS contexts where a type-specific helper function
	// was being called.
	if ( bli_is_trans( trans ) )
	{
		      bool   packtmp = packa; packa = packb; packb = packtmp;
		      conj_t conjtmp = conja; conja = conjb; conjb = conjtmp;
		      dim_t  len_tmp =     m;     m =     n;     n = len_tmp;
		const void*  buf_tmp = buf_a; buf_a = buf_b; buf_b = buf_tmp;
		      inc_t  str_tmp =  rs_a;  rs_a =  cs_b;  cs_b = str_tmp;
		             str_tmp =  cs_a;  cs_a =  rs_b;  rs_b = str_tmp;
		             str_tmp =  rs_c;  rs_c =  cs_c;  cs_c = str_tmp;

		stor_id = bli_stor3_trans( stor_id );
	}

	// Query the context for various blocksizes.
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
	const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );

	// Disable modification of KC since it seems to negatively impact certain
	// operations (#644).
	dim_t KC = KC0;

	/*
	if      ( packa && packb )
	{
		KC = KC0;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

	// Query the maximum blocksize for NR, which implies a maximum blocksize
	// extension for the final iteration.
	const dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx );
	const dim_t NRE = NRM - NR;

	// Compute partitioning step values for each matrix of each loop.
	const inc_t jcstep_c = cs_c * dt_size;
	const inc_t jcstep_b = cs_b * dt_size;

	const inc_t pcstep_a = cs_a * dt_size;
	const inc_t pcstep_b = rs_b * dt_size;

	const inc_t icstep_c = rs_c * dt_size;
	const inc_t icstep_a = rs_a * dt_size;

	const inc_t jrstep_c = cs_c * NR * dt_size;

	//const inc_t jrstep_b = cs_b * NR;
	//( void )jrstep_b;

	//const inc_t irstep_c = rs_c * MR;
	//const inc_t irstep_a = rs_a * MR;

	// Query the context for the sup microkernel address and cast it to its
	// function pointer type.
	gemmsup_ker_ft gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );

	const char* a_00       = buf_a;
	const char* b_00       = buf_b;
	      char* c_00       = buf_c;
	const void* one        = bli_obj_buffer_for_const( dt, &BLIS_ONE );

	auxinfo_t aux;

	// Determine whether we are using more than one thread.
	const bool is_mt = ( bli_rntm_calc_num_threads( rntm ) > 1 );

	thrinfo_t* thread_jc = bli_thrinfo_sub_node( thread );
	thrinfo_t* thread_pc = bli_thrinfo_sub_node( thread_jc );
	thrinfo_t* thread_pb = bli_thrinfo_sub_node( thread_pc );
	thrinfo_t* thread_ic = bli_thrinfo_sub_node( thread_pb );
	thrinfo_t* thread_pa = bli_thrinfo_sub_node( thread_ic );
	thrinfo_t* thread_jr = bli_thrinfo_sub_node( thread_pa );

	// Compute the JC loop thread range for the current thread.
	dim_t jc_start, jc_end;
	bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	// Compute number of primary and leftover components of the JC loop.
	//const dim_t jc_iter = ( n_local + NC - 1 ) / NC;
	const dim_t jc_left =   n_local % NC;

	// Loop over the n dimension (NC rows/columns at a time).
	//for ( dim_t jj = 0; jj < jc_iter; jj += 1 )
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		// Calculate the thread's current JC block dimension.
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		const char* b_jc = b_00 + jj * jcstep_b;
		      char* c_jc = c_00 + jj * jcstep_c;

		// Compute the PC loop thread range for the current thread.
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		// Compute number of primary and leftover components of the PC loop.
		//const dim_t pc_iter = ( k_local + KC - 1 ) / KC;
		const dim_t pc_left =   k_local % KC;

		// Loop over the k dimension (KC rows/columns at a time).
		//for ( dim_t pp = 0; pp < pc_iter; pp += 1 )
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			// Calculate the thread's current PC block dimension.
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			const char* a_pc = a_00 + pp * pcstep_a;
			const char* b_pc = b_jc + pp * pcstep_b;

			// Only apply beta to the first iteration of the pc loop.
			const void* beta_use = ( pp == 0 ? buf_beta : one );

			      char* b_use;
			      inc_t rs_b_use, cs_b_use, ps_b_use;

			// Determine the packing buffer and related parameters for matrix
			// B. (If B will not be packed, then a_use will be set to point to
			// b and the _b_use strides will be set accordingly.) Then call
			// the packm sup variant chooser, which will call the appropriate
			// implementation based on the schema deduced from the stor_id.
			bli_packm_sup
			(
			  packb,
			  BLIS_BUFFER_FOR_B_PANEL, // This algorithm packs matrix B to
			  stor_id,                 // a "panel of B."
			  BLIS_NO_TRANSPOSE,
			  dt,
			  NC,     KC,       // This "panel of B" is (at most) KC x NC.
			  nc_cur, kc_cur, NR,
			  one,
			  b_pc,   cs_b,      rs_b,
			  ( void** )&b_use, &cs_b_use, &rs_b_use,
			                    &ps_b_use,
			  cntx,
			  thread_pb
			);

			// Alias b_use so that it's clear this is our current block of
			// matrix B.
			char* b_pc_use = b_use;

			// We don't need to embed the panel stride of B within the auxinfo_t
			// object because this variant iterates through B in the jr loop,
			// which occurs here, within the macrokernel, not within the
			// millikernel.
			//bli_auxinfo_set_ps_b( ps_b_use, &aux );

			// Compute the IC loop thread range for the current thread.
			dim_t ic_start, ic_end;
			bli_thread_range_sub( thread_ic, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;

			// Compute number of primary and leftover components of the IC loop.
			//const dim_t ic_iter = ( m_local + MC - 1 ) / MC;
			const dim_t ic_left =   m_local % MC;

			// Loop over the m dimension (MC rows at a time).
			//for ( dim_t ii = 0; ii < ic_iter; ii += 1 )
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				// Calculate the thread's current IC block dimension.
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );

				const char* a_ic = a_pc + ii * icstep_a;
				      char* c_ic = c_jc + ii * icstep_c;

				      char* a_use;
				      inc_t rs_a_use, cs_a_use, ps_a_use;

				// Determine the packing buffer and related parameters for matrix
				// A. (If A will not be packed, then a_use will be set to point to
				// a and the _a_use strides will be set accordingly.) Then call
				// the packm sup variant chooser, which will call the appropriate
				// implementation based on the schema deduced from the stor_id.
				bli_packm_sup
				(
				  packa,
				  BLIS_BUFFER_FOR_A_BLOCK, // This algorithm packs matrix A to
				  stor_id,                 // a "block of A."
				  BLIS_NO_TRANSPOSE,
				  dt,
				  MC,     KC,       // This "block of A" is (at most) MC x KC.
				  mc_cur, kc_cur, MR,
				  one,
				  a_ic,   rs_a,      cs_a,
				  ( void** )&a_use, &rs_a_use, &cs_a_use,
				                    &ps_a_use,
				  cntx,
				  thread_pa
				);

				// Alias a_use so that it's clear this is our current block of
				// matrix A.
				char* a_ic_use = a_use;

				// Embed the panel stride of A within the auxinfo_t object. The
				// millikernel will query and use this to iterate through
				// micropanels of A (if needed).
				bli_auxinfo_set_ps_a( ps_a_use, &aux );

				// Compute number of primary and leftover components of the JR loop.
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR;
				dim_t jr_left =   nc_cur % NR;

				// An optimization: allow the last jr iteration to contain up to NRE
				// columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				// these cases.) Note that this prevents us from declaring jr_iter and
				// jr_left as const. NOTE: We forgo this optimization when packing B
				// since packing an extended edge case is not yet supported.
				if ( !packb && !is_mt )
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE )
				{
					jr_iter--; jr_left += NR;
				}

				// Compute the JR loop thread range for the current thread.
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				// Loop over the n dimension (NR columns at a time).
				//for ( dim_t j = 0; j < jr_iter; j += 1 )
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );

					//ctype* b_jr = b_pc_use + j * jrstep_b;
					const char* b_jr = b_pc_use + j * ps_b_use * dt_size;
					      char* c_jr = c_ic     + j * jrstep_c;

					//const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR;
					//const dim_t ir_left =   mc_cur % MR;

					// Loop over the m dimension (MR rows at a time).
					{
						// Invoke the gemmsup millikernel.
						gemmsup_ker
						(
						  conja,
						  conjb,
						  mc_cur,
						  nr_cur,
						  kc_cur,
						  ( void* )buf_alpha,
						  ( void* )a_ic_use, rs_a_use, cs_a_use,
						  ( void* )b_jr,     rs_b_use, cs_b_use,
						  ( void* )beta_use,
						  ( void* )c_jr,     rs_c,     cs_c,
						  &aux,
						  ( cntx_t* )cntx
						);
					}
				}
			}

			// NOTE: This barrier is only needed if we are packing B (since
			// that matrix is packed within the pc loop of this variant).
			if ( packb ) bli_thrinfo_barrier( thread_pb );
		}
	}

	// Release any memory that was acquired for packing matrices A and B.
	bli_packm_sup_finalize_mem
	(
	  packa,
	  thread_pa
	);
	bli_packm_sup_finalize_mem
	(
	  packb,
	  thread_pb
	);

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,scalm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  beta, \
			  c, rs_c, cs_c \
			); \
		} \
		return; \
	} \
\
	/* Query the context for various blocksizes. */ \
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	dim_t KC; \
	if      ( packa && packb ) \
	{ \
		KC = KC0; \
	} \
	else if ( packb ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else if ( packa ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else /* if ( !packa && !packb ) */ \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( m <=   MR && n <=   NR ) KC = KC0; \
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
		/* Will revisit setting this KC value - larger m and n demands larger KC */	\
		else                               KC = KC0; /* (( KC0 / 5 ) / 4 ) * 4; VK */ \
	} \
\
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */ \
	const dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
	const dim_t NRE = NRM - NR; \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_c; \
	const inc_t jcstep_b = cs_b; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = rs_c; \
	const inc_t icstep_a = rs_a; \
\
	const inc_t jrstep_c = cs_c * NR; \
\
	/*
	const inc_t jrstep_b = cs_b * NR; \
	( void )jrstep_b; \
\
	const inc_t irstep_c = rs_c * MR; \
	const inc_t irstep_a = rs_a * MR; \
	*/ \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ \
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/ \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */ \
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids; \
\
	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */ \
	if ( packa ) { if ( packb ) bszids = bszids_packab; \
	               else         bszids = bszids_packa; } \
	else         { if ( packb ) bszids = bszids_packb; \
	               else         bszids = bszids_nopack; } \
\
	/* Determine whether we are using more than one thread. */ \
	const bool is_mt = ( bli_rntm_calc_num_threads( rntm ) > 1 ); \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end ); \
	const dim_t n_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n_local % NC; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			ctype* b_use; \
			inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */ \
			bszid_t*   restrict bszids_pb; \
			if ( packb ) { bszids_pb = &bszids_pc[1]; \
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); } \
			else         { bszids_pb = &bszids_pc[0]; \
			               thread_pb = thread_pc; } \
\
			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */ \
			PASTEMAC(ch,packm_sup_b) \
			( \
			  packb, \
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */ \
			  stor_id,                 /* a "panel of B."                  */ \
			  BLIS_NO_TRANSPOSE, \
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */ \
			  kc_cur, nc_cur, NR, \
			  &one_local, \
			  b_pc,   rs_b,      cs_b, \
			  &b_use, &rs_b_use, &cs_b_use, \
			                     &ps_b_use, \
			  cntx, \
			  rntm, \
			  &mem_b, \
			  thread_pb  \
			); \
\
			/* Alias b_use so that it's clear this is our current block of
			   matrix B. */ \
			ctype* restrict b_pc_use = b_use; \
\
			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */ \
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t*   restrict bszids_ic = &bszids_pb[1]; \
			                    thread_ic = bli_thrinfo_sub_node( thread_pb ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_sub( thread_ic, m, MR, FALSE, &ic_start, &ic_end ); \
			const dim_t m_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   m_local % MC; \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				ctype* a_use; \
				inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */ \
				bszid_t*   restrict bszids_pa; \
				if ( packa ) { bszids_pa = &bszids_ic[1]; \
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); } \
				else         { bszids_pa = &bszids_ic[0]; \
							   thread_pa = thread_ic; } \
\
				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */ \
				PASTEMAC(ch,packm_sup_a) \
				( \
				  packa, \
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */ \
				  stor_id,                 /* a "block of A."                  */ \
				  BLIS_NO_TRANSPOSE, \
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */ \
				  mc_cur, kc_cur, MR, \
				  &one_local, \
				  a_ic,   rs_a,      cs_a, \
				  &a_use, &rs_a_use, &cs_a_use, \
				                     &ps_a_use, \
				  cntx, \
				  rntm, \
				  &mem_a, \
				  thread_pa  \
				); \
\
				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */ \
				ctype* restrict a_ic_use = a_use; \
\
				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */ \
				bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t*   restrict bszids_jr = &bszids_pa[1]; \
				                    thread_jr = bli_thrinfo_sub_node( thread_pa ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
				dim_t jr_left =   nc_cur % NR; \
\
				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */ \
				if ( !packb && !is_mt ) \
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
				{ \
					jr_iter--; jr_left += NR; \
				} \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					/*
					ctype* restrict b_jr = b_pc_use + j * jrstep_b; \
					*/ \
					ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
					/*
					const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR; \
					const dim_t ir_left =   mc_cur % MR; \
					*/ \
\
					/* Loop over the m dimension (MR rows at a time). */ \
					{ \
						/* Invoke the gemmsup millikernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  mc_cur, \
						  nr_cur, \
						  kc_cur, \
						  alpha_cast, \
						  a_ic_use, rs_a_use, cs_a_use, \
						  b_jr,     rs_b_use, cs_b_use, \
						  beta_use, \
						  c_jr,     rs_c,     cs_c, \
						  &aux, \
						  cntx  \
						); \
					} \
				} \
			} \
\
			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */ \
			if ( packb ) bli_thread_barrier( thread_pb ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packm_sup_finalize_mem_a) \
	( \
	  packa, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}

