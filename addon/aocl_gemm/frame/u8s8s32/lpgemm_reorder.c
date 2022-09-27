/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_utils.h"
#include "lpgemm_reorder.h"
#include "lpgemm_packa.h"
#include "lpgemm_packb.h"
#include "lpgemm_config.h"

void reorderb_nr64_u8s8s32o32
     (
       lpgemm_obj_t* b,
       lpgemm_obj_t* b_reorder
     )
{
	dim_t NC = lpgemm_get_block_size_NC_global_cntx( U8S8S32OS32 );
	dim_t NR = lpgemm_get_block_size_NR_global_cntx( U8S8S32OS32 );
	dim_t KC = lpgemm_get_block_size_KC_global_cntx( U8S8S32OS32 );

	dim_t rs_b = b->rs;
	dim_t rs_b_reorder;
	dim_t cs_b_reorder;

	dim_t n = b->width;
	dim_t k = b->length;

	// k needs to be a multiple of 4 so that it can be used with vpdpbusd
	// instruction. Padding is added in cases this condition is not
	// satisfied, and therefore the k offset used for packed/reordered
	// buffer needs to be updated.
	dim_t k_updated = make_multiple_of_n( k, 4 );

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
		// Compute the JC loop thread range for the current thread.
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
			  jc, n, NC, get_packb_u8s8s32o32_min_NR(),
			  &jc_cur_loop, &jc_cur_loop_rem,
			  &nc0, &n_sub_updated
			);

			for ( dim_t pc = 0; pc < k; pc += KC )
			{
				dim_t kc0 = bli_min( ( k - pc ), KC );

				// kc0 needs to be a multiple of 4 so that it can be used with
				// vpdpbusd instruction. Padding is added in cases this
				// condition is not satisfied, and therefore the kc0 offsets
				// used for packed/reordered buffers needs to be updated.
				dim_t kc0_updated = make_multiple_of_n( kc0, 4 );

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
#ifdef BLIS_KERNELS_ZEN4
				packb_nr64_u8s8s32o32
				(
				  ( ( ( int8_t* )b_reorder->storage.aligned_buffer ) +
					( jc_cur_loop * k_updated ) + ( n_sub_updated * pc ) +
					( jc_cur_loop_rem * kc0_updated ) ),
				  ( ( ( int8_t* )b->storage.aligned_buffer ) +
					( rs_b * pc ) + jc ),
				  rs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder
				);
#else
				// Silence compiler warnings.
				rs_b_reorder = 0;
				cs_b_reorder = 0;
				( void )kc0_updated;
				( void )k_updated;
				( void )rs_b;
#endif
			}

			adjust_B_panel_reordered_jc( &jc, jc_cur_loop );
		}
	}

	b_reorder->rs = rs_b_reorder;
	b_reorder->cs = cs_b_reorder;
	b_reorder->mtag = REORDERED;
}

void reordera_mr6_u8s8s32o32
     (
       lpgemm_obj_t* a,
       lpgemm_obj_t* a_reorder
     )
{
	dim_t MC = lpgemm_get_block_size_MC_global_cntx( U8S8S32OS32 );
	dim_t KC = lpgemm_get_block_size_KC_global_cntx( U8S8S32OS32 );

	dim_t rs_a = a->rs;
	dim_t rs_a_reorder;
	dim_t cs_a_reorder;

	dim_t k = a->width;
	dim_t m = a->length;

	for ( dim_t pc = 0; pc < k; pc += KC )
	{
		dim_t kc0 = bli_min( ( k - pc ), KC );

		// kc0 needs to be a multiple of 4 so that it can be used with
		// vpdpbusd instruction. Padding is added in cases this
		// condition is not satisfied, and therefore the kc0 offsets
		// used for packed/reordered buffers needs to be updated.
		dim_t kc0_updated = make_multiple_of_n( kc0, 4 );

		for ( dim_t ic = 0; ic < m; ic += MC )
		{
			dim_t mc0 = bli_min( ( m - ic ), MC );

#ifdef BLIS_KERNELS_ZEN4
			packa_k64_u8s8s32o32
			(
			  ( ( ( uint8_t* )a_reorder->storage.aligned_buffer ) + ( pc * m ) +
				( ic * kc0_updated ) ),
			  ( ( ( uint8_t* )a->storage.aligned_buffer ) + ( rs_a * ic ) + pc ),
			  rs_a, mc0, kc0, &rs_a_reorder, &cs_a_reorder
			);
#else
			rs_a_reorder = 0;
			cs_a_reorder = 0;
			( void )kc0_updated;
			( void )rs_a;
			( void )mc0;
#endif
		}
	}

	a_reorder->rs = rs_a_reorder;
	a_reorder->cs = cs_a_reorder;
	a_reorder->mtag = REORDERED;
}
