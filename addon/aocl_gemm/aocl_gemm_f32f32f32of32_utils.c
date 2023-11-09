/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "aocl_gemm_interface_apis.h"
#include "lpgemm_config.h"
#include "lpgemm_utils.h"

AOCL_GEMM_GET_REORDER_BUF_SIZE(f32f32f32of32)
{
	if ( ( k <= 0 ) || ( n <= 0 ) )
	{
		return 0; // Error.
	}

	// Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform f32f32f32 gemm.", __FILE__, __LINE__ );
		return 0; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Initialize lpgemm context.
	aocl_lpgemm_init_global_cntx();

	// Query the global cntx.
	cntx_t* cntx = bli_gks_query_cntx();

	num_t dt = BLIS_FLOAT;

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return 0; // A reorder not supported.
	}

	const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );

	// Extra space since packing does width in multiples of NR.
	const dim_t n_reorder = ( ( n + NR - 1 ) / NR ) * NR;

	siz_t size_req = sizeof( float ) * k * n_reorder;

	return size_req;
}

// Pack B into row stored column panels.
AOCL_GEMM_REORDER(float,f32f32f32of32)
{
	if ( ( input_buf_addr == NULL ) || ( reorder_buf_addr == NULL ) ||
	     ( k <= 0 ) || ( n <= 0 ) || ( ldb < n ) )
	{
		return; // Error.
	}

	// Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform f32f32f32 gemm.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Initialize lpgemm context.
	aocl_lpgemm_init_global_cntx();

	// Query the global cntx.
	cntx_t* cntx = bli_gks_query_cntx();

	num_t dt = BLIS_FLOAT;

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return; // A reorder not supported.
	}

	const dim_t NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
	const dim_t KC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );

	// Only supports row major packing now.
	inc_t rs_b = ldb;
	inc_t cs_b = 1;

	inc_t rs_p = NR;

	float one_local  = *PASTEMAC(s,1);
	float* restrict kappa_cast = &one_local;

	// Set the schema to "row stored column panels" to indicate packing to
	// conventional row-stored column panels.
	pack_t schema = BLIS_PACKED_COL_PANELS;
	trans_t transc = BLIS_NO_TRANSPOSE;
	conj_t conjc = bli_extract_conj( transc );

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );

	dim_t n_threads = bli_rntm_num_threads( &rntm_g );
	n_threads = ( n_threads > 0 ) ? n_threads : 1;

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp parallel num_threads(n_threads)" )
	{
		// Initialise a local thrinfo obj for work split across threads.
		thrinfo_t thread_jc;
		bli_thrinfo_set_n_way( n_threads, &thread_jc );
		bli_thrinfo_set_work_id( omp_get_thread_num(), &thread_jc );
#else
	{
		// Initialise a local thrinfo obj for work split across threads.
		thrinfo_t thread_jc;
		bli_thrinfo_set_n_way( 1, &thread_jc );
		bli_thrinfo_set_work_id( 0, &thread_jc );
#endif
		// Compute the JC loop thread range for the current thread. Per thread
		// gets multiple of NR columns.
		dim_t jc_start, jc_end;
		bli_thread_range_sub( &thread_jc, n, NR, FALSE, &jc_start, &jc_end );

		for ( dim_t jc = jc_start; jc < jc_end; jc += NC )
		{
			dim_t nc0 = bli_min( ( jc_end - jc ), NC );

			dim_t jc_cur_loop = jc;
			dim_t jc_cur_loop_rem = 0;
			dim_t n_sub_updated;

			get_B_panel_reordered_start_offset_width
			(
			  jc, n, NC, NR,
			  &jc_cur_loop, &jc_cur_loop_rem,
			  &nc0, &n_sub_updated
			);

			// Compute the total number of iterations we'll need.
			dim_t n_iter = ( nc0 + NR - 1 ) / NR;

			for ( dim_t pc = 0; pc < k; pc += KC )
			{
				dim_t kc0 = bli_min( ( k - pc ), KC );
				inc_t ps_p = kc0 * NR;

				const float* b_temp = input_buf_addr + ( jc * cs_b ) + ( pc * rs_b );

				// The offsets are calculated in such a way that it resembles
				// the reorder buffer traversal in single threaded reordering.
				// The panel boundaries (KCxNC) remain as it is accessed in
				// single thread, and as a consequence a thread with jc_start
				// inside the panel cannot consider NC range for reorder. It
				// has to work with NC' < NC, and the offset is calulated using
				// prev NC panels spanning k dim + cur NC panel spaning pc loop
				// cur iteration + (NC - NC') spanning current kc0 (<= KC).
				//
				//Eg: Consider the following reordered buffer diagram:
				//          t1              t2
				//          |               |
				//          |           |..NC..|
				//          |           |      |
				//          |.NC. |.NC. |NC'|NC"
				//     pc=0-+-----+-----+---+--+
				//        KC|     |     |   |  |
				//          |  1  |  3  |   5  |
				//    pc=KC-+-----+-----+---st-+
				//        KC|     |     |   |  |
				//          |  2  |  4  | 6 | 7|
				// pc=k=2KC-+-----+-----+---+--+
				//          |jc=0 |jc=NC|jc=2NC|
				//
				// The numbers 1,2..6,7 denotes the order in which reordered
				// KCxNC blocks are stored in memory, ie: block 1 followed by 2
				// followed by 3, etc. Given two threads t1 and t2, and t2 needs
				// to acces point st in the reorder buffer to write the data:
				// The offset calulation logic will be:
				// jc_cur_loop = 2NC, jc_cur_loop_rem = NC', pc = KC,
				// n_sub_updated = NC, k = 2KC, kc0_updated = KC
				//
				// st = ( jc_cur_loop * k )    <traverse blocks 1,2,3,4>
				//    + ( n_sub_updated * pc ) <traverse block 5>
				//    + ( NC' * kc0_updated)   <traverse block 6>
				float* p_temp = reorder_buf_addr + ( jc_cur_loop * k ) +
								( n_sub_updated * pc ) + ( jc_cur_loop_rem * kc0 );

				dim_t jr, it;
				// Iterate over every logical micropanel in the source matrix.
				for ( jr = 0, it = 0; it < n_iter; jr += NR, it += 1 )
				{
					dim_t panel_dim_i = bli_min( NR, nc0 - jr );

					const float* b_use = b_temp + ( jr * cs_b );
					float* p_use = p_temp;

					PASTEMAC(s,packm_cxk)
					(
					  conjc,
					  schema,
					  panel_dim_i,
					  NR,
					  kc0,
					  kc0,
					  kappa_cast,
					  ( float* )b_use, cs_b, rs_b,
					  p_use, rs_p,
					  cntx
					);

					p_temp += ps_p;
				}
			}

			adjust_B_panel_reordered_jc( &jc, jc_cur_loop );
		}
	}
}
