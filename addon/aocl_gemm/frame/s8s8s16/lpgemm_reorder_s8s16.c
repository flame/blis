/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_utils_s8.h"
#include "lpgemm_reorder_s8s16.h"
#include "lpgemm_packb_s8s16.h"
#include "lpgemm_config.h"

void aocl_reorderb_nr32_s8s8s16o16
     (
       lpgemm_obj_t*  b,
       lpgemm_obj_t*  b_reorder,
       rntm_t*        rntm,
       lpgemm_cntx_t* lcntx
     )
{
	dim_t NC = lcntx->blksz.NC;
	dim_t KC = lcntx->blksz.KC;
	dim_t NR = lcntx->blksz.NR;

	// Extracting the matrix properties from the lpgemm object
	dim_t rs_b = b->rs;
	dim_t n = b->width;
	dim_t k = b->length;

	lpgemm_mod_block_size_s16(0, n, k, NULL, &NC, &KC);

	dim_t rs_b_reorder;
	dim_t cs_b_reorder;

	dim_t k_updated = k;

	// Making multiple of 2 to suit k in vpmaddubsw
	k_updated += (k_updated & 0x1);

    dim_t n_updated = make_multiple_of_n( n, 16 );

	dim_t n_threads = bli_rntm_num_threads( rntm );
	n_threads = ( n_threads > 0 ) ? n_threads : 1;

    // To access the last row of B matrix - Column sum of B matrix
    int16_t* pack_b_column_sum = ( int16_t* ) ( b_reorder->storage.aligned_buffer + ( sizeof( int8_t ) * n_updated * k_updated ));
	for (dim_t idx = 0; idx < n_updated; idx++ )
	{
		*( pack_b_column_sum + idx ) =  0;
	}

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
			  jc, n, NC, 16,
			  &jc_cur_loop, &jc_cur_loop_rem,
			  &nc0, &n_sub_updated
			);

			for ( dim_t pc = 0; pc < k; pc += KC )
			{
				dim_t kc0 = bli_min( ( k - pc ), KC );

				// kc0 needs to be a multiple of 2 so that it can be used with
				// vmaddubsw instruction. Padding is added in cases this
				// condition is not satisfied, and therefore the kc0 offsets
				// used for packed/reordered buffers needs to be updated.
				dim_t kc0_updated = make_multiple_of_n( kc0, 2 );

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
				( ( packb_s16_s8 )lcntx->packb_fun_ptr )
				(
				  ( ( ( int8_t* )b_reorder->storage.aligned_buffer ) +
					( jc_cur_loop * k_updated ) + ( n_sub_updated * pc ) +
					( jc_cur_loop_rem * kc0_updated ) ),
                    pack_b_column_sum + jc,
				  ( ( ( int8_t* )b->storage.aligned_buffer ) +
					( rs_b * pc ) + jc ),
				  rs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder
				);
			}

			adjust_B_panel_reordered_jc( &jc, jc_cur_loop );
		}
	}

	// Changing the packed matrix properties in the packed matrix object
	b_reorder->rs = rs_b_reorder;
	b_reorder->cs = cs_b_reorder;
	b_reorder->mtag = REORDERED;
}
