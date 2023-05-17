/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_packa_s8.h"
#include "lpgemm_packb_s8.h"
#include "lpgemm_kernels.h"
#include "lpgemm_utils_s8.h"
#include "lpgemm_thrinfo_utils.h"
#include "lpgemm_config.h"

// Kernel function prototypes
typedef void (*lpgemm_rowvar_s32_s8)
     (
       const dim_t,
       const dim_t,
       const dim_t,
       const int8_t*,
       const dim_t,
       const dim_t,
       const dim_t,
       const int8_t*,
       const dim_t,
       const dim_t,
       int32_t*,
       const dim_t,
       const dim_t,
       const int32_t,
       const int32_t,
       lpgemm_post_op*,
       lpgemm_post_op_attr
     );

// B should always be packed.
LPGEMM_5LOOP(int8_t,int8_t,int32_t,s8s8s32o32)
{
	dim_t NC = lcntx->blksz.NC;
	dim_t KC = lcntx->blksz.KC;
	dim_t MC = lcntx->blksz.MC;
	dim_t NR = lcntx->blksz.NR;
	dim_t MR = lcntx->blksz.MR;

	if ( mtag_b == UNPACKED )
	{
		//Error: can only work with packed B now.
		return;
	}

	// Strides are updated based on matrix packing/reordering.
	const int8_t* a_use = NULL;
	dim_t rs_a_use = rs_a;
	dim_t cs_a_use = cs_a;
	dim_t a_block_stride = 0;

	const int8_t* b_use = NULL;
	dim_t rs_b_use = rs_b;
	dim_t cs_b_use = cs_b;

	int32_t* c_use_jc = NULL;
	int32_t* c_use_ic = NULL;
	dim_t rs_c_use = rs_c;
	dim_t rs_c_downscale = rs_c;

	// Pack buffer for A.
	int8_t* pack_a_buffer_s8s8s32o32;
	mem_t mem_a = BLIS_MEM_INITIALIZER;
	siz_t mem_a_size_req = 0;

	// Pack buffer for B.
	int8_t* pack_b_buffer_s8s8s32o32;
	mem_t mem_b = BLIS_MEM_INITIALIZER;
	siz_t mem_b_size_req = 0;
	dim_t packb_min_NR = get_packb_s8s8s32o32_min_NR();

	// Temporary buffer for C accumulation when downscaling is required.
	int32_t* temp_scal_c_buffer_s8s8s32o32;
	mem_t mem_scale_c = BLIS_MEM_INITIALIZER;
	siz_t mem_scale_c_size_req = 0;

	// kc needs to be a multiple of 4 so that it can be used with vpdpbusd
	// instruction. Padding is added in cases this condition is not
	// satisfied, and therefore the k offset used for packed/reordered
	// buffer needs to be updated.
	dim_t k_updated = make_multiple_of_n( k, 4 );
	dim_t n_updated = make_multiple_of_n( n, 16 );

	// To decide whether to apply post ops or not.
	bool is_last_k = FALSE;

	// To decide whether to use original s8 C or temp buffer for beta scale.
	bool is_first_k = FALSE;

	lpgemm_post_op_attr post_ops_attr;
	if ( c_downscale == TRUE )
	{
		post_ops_attr.buf_downscale = c;
	}
	else
	{
		post_ops_attr.buf_downscale = NULL;
	}

	// Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
	thrinfo_t thread_jc;
	thrinfo_t thread_ic;

	lpgemm_gen_thrinfo( thread, &thread_jc, &thread_ic );

	// Compute the JC, IC loop thread range for the current thread.
	dim_t jc_start, jc_end;
	bli_thread_range_sub( &thread_jc, n, NR, FALSE, &jc_start, &jc_end );

	dim_t ic_start, ic_end;
	bli_thread_range_sub( &thread_ic, m, MR, FALSE, &ic_start, &ic_end );

	for ( dim_t jc = jc_start; jc < jc_end; jc += NC )
	{
		dim_t nc0 = bli_min( ( jc_end - jc ), NC );

		dim_t jc_cur_loop = jc;
		dim_t jc_cur_loop_rem = 0;
		dim_t n_sub_updated = 0;

		if ( mtag_b == REORDERED )
		{
			get_B_panel_reordered_start_offset_width
			(
			  jc, n, NC, packb_min_NR,
			  &jc_cur_loop, &jc_cur_loop_rem,
			  &nc0, &n_sub_updated
			);
		}

		if ( c_downscale == FALSE )
		{
			c_use_jc = c + jc;
		}
		// Temp accumulaton buffer for C allocation.
		else if ( c_downscale == TRUE )
		{
			// Buffer memory is only required if output needs to be
			// persisted across iterations of the pc/KC loop.
			// It was observed that the locks used while checking out
			// a buffer from memory pool had an impact on performance
			// and is better to not checkout if k <= KC.
			if ( k > KC )
			{
				mem_scale_c_size_req = sizeof( int32_t ) * nc0 * ( ic_end - ic_start );

				lpgemm_alloc_mem_panel
				(
				  mem_scale_c_size_req, BLIS_BUFFER_FOR_C_PANEL,
				  &mem_scale_c, rntm
				);

				temp_scal_c_buffer_s8s8s32o32 = bli_mem_buffer( &mem_scale_c );

				c_use_jc = ( int32_t* )temp_scal_c_buffer_s8s8s32o32;
			}

			// The temp c buffer stride is modified as opposed to original C matrix.
			rs_c_use = nc0;
		}

		int32_t* pack_b_column_sum = NULL;

		for ( dim_t pc = 0; pc < k; pc += KC )
		{
			int32_t beta0 = ( pc == 0 ) ? beta : 1;
			dim_t kc0 = bli_min( ( k - pc ), KC );

			// kc0 needs to be a multiple of 4 so that it can be
			// used with vpdpbusd instruction. Padding is added in
			// cases this condition is not satisfied, and therefore
			// the kc0 offsets used for packed/reordered buffers
			// needs to be updated.
			dim_t kc0_updated = make_multiple_of_n( kc0, 4 );

			// No parallelization in k dim, k always starts at 0.
			is_first_k = ( pc == 0 ) ? ( TRUE ) : ( FALSE );
			post_ops_attr.is_first_k = is_first_k;

			is_last_k = ( ( pc + KC ) >= k ) ? ( TRUE ) : ( FALSE );
			post_ops_attr.is_last_k = is_last_k;

			if ( mtag_b == PACK )
			{
				// Pack B chunks are based on jc work id.
				dim_t jc_work_id = bli_thread_work_id( &thread_jc );

				// Using child thrinfo (thread_ic) tid to decide chief thread
				// per B matrix chunk (jc work id group)
				dim_t nc0_updated = make_multiple_of_n( nc0, packb_min_NR );

				if ( bli_thread_am_ochief( &thread_ic ) )
				{
					// nc0 needs to be a multiple of 16 since this gives maximum
					// vectorization. Packing B always results in buffers with width
					// which is a multiple of 16. Subsequently the nc0 offsets used
					// for packed/reordered buffers needs to be updated.pack

					mem_b_size_req = sizeof( int8_t ) * nc0_updated * kc0_updated + ( nc0_updated * sizeof( int32_t ) );

					lpgemm_alloc_mem_panel
					(
					  mem_b_size_req, BLIS_BUFFER_FOR_B_PANEL,
					  &mem_b, rntm
					);

					thread->comm[jc_work_id].sent_object = bli_mem_buffer( &mem_b );
				}

				// All threads in work group should wait till chief thread has
				// finished allocating the packing buffers.
				bli_thrcomm_barrier
				(
				  bli_thread_ocomm_id( &thread_ic ),
				  &thread->comm[jc_work_id]
				);

				pack_b_buffer_s8s8s32o32 =
						( int8_t* ) thread->comm[jc_work_id].sent_object;

				// Compute the B panel per thread loop range for parallel
				// packing using ic_ways number of threads. Since atmost only
				// ic_ways threads can be used, the thread_ic attributes are
				// used to split the loop range.
				dim_t jc_packb_start, jc_packb_end;
				bli_thread_range_sub
				(
				  &thread_ic, nc0, NR, FALSE,
				  &jc_packb_start, &jc_packb_end
				);

				if ( pc == 0)
				{
					pack_b_column_sum = ( int32_t* )( pack_b_buffer_s8s8s32o32 + ( sizeof( int8_t ) * nc0_updated * kc0_updated ) );
				}

				// Ensure thread ranges are valid, especially cases where no:
				// of threads available for parallelization are greater than
				// no: of B panel NR chunks.
				if ( ( jc_packb_end > jc_packb_start ) &&
					 ( jc_packb_start < ( jc + nc0 ) ) )
				{
					if ( pc == 0 )
					{
						for (dim_t idx = jc_packb_start; idx < jc_packb_end; idx++ )
						{
							*( pack_b_column_sum + idx ) =  0;
						}
					}

					( ( packb_s32_s8 )lcntx->packb_fun_ptr )
					(
					  pack_b_buffer_s8s8s32o32 + ( jc_packb_start * kc0_updated ),
					  pack_b_column_sum + ( cs_b * jc_packb_start ),
					  ( b + ( rs_b * pc ) + ( cs_b * jc ) +
					    ( cs_b * jc_packb_start ) ), rs_b,
					  ( jc_packb_end - jc_packb_start ), kc0,
					  &rs_b_use, &cs_b_use
					);
				}
				else
				{
					lpgemm_get_packb_strides( lcntx, &rs_b_use, &cs_b_use );
				}

				// All threads in work group should wait till B matrix packing
				// is completed by the participating threads.
				bli_thrcomm_barrier
				(
				  bli_thread_ocomm_id( &thread_ic ),
				  &thread->comm[jc_work_id]
				);
				b_use = pack_b_buffer_s8s8s32o32;

				post_ops_attr.b_col_sum_vec = pack_b_column_sum;
			}
			else if ( mtag_b == REORDERED )
			{
				// In multi-threaded scenarios, an extra offset into a given
				// packed B panel is required, since the jc loop split can
				// result in per thread start offset inside the panel, instead
				// of panel boundaries.
				b_use = b + ( jc_cur_loop * k_updated ) +
						( n_sub_updated * pc ) +
						( jc_cur_loop_rem * kc0_updated );

				lpgemm_get_packb_strides( lcntx, &rs_b_use, &cs_b_use );

				post_ops_attr.b_col_sum_vec = ( ( int32_t* )( b + ( k_updated * n_updated ) ) ) + jc;
			}
			else
			{
				//Unpacked B not supported.
				return;
			}

			for ( dim_t ic = ic_start; ic < ic_end; ic += MC )
			{
				dim_t mc0 = bli_min( ( ic_end - ic ), MC );

				// Only per thread C matrix is stored in temp buffer, so both
				// per thread jc and ic start should be normalized to zero.
				if ( c_downscale == TRUE )
				{
					c_use_ic = c_use_jc + ( rs_c_use * ( ic - ic_start ) );
				}
				else
				{
					c_use_ic = c_use_jc + ( rs_c_use * ic );
				}

				// Matrix A packed and reordered code path is not triggerred
				// currently since we do not support it yet.
				if ( mtag_a == PACK )
				{
					mem_a_size_req = sizeof( int8_t ) * mc0 * kc0_updated;

					lpgemm_alloc_mem_panel
					(
					  mem_a_size_req, BLIS_BUFFER_FOR_A_BLOCK,
					  &mem_a, rntm
					);
					pack_a_buffer_s8s8s32o32 = ( int8_t* )bli_mem_buffer( &mem_a );

					( ( packa_s32_s8 )lcntx->packa_fun_ptr )
					(
					  pack_a_buffer_s8s8s32o32,
					  ( a + ( rs_a * ic ) + pc ), rs_a,
					  mc0, kc0,
					  &rs_a_use, &cs_a_use
					);
					a_use = pack_a_buffer_s8s8s32o32;
					a_block_stride = kc0_updated;
				}

				else
				{
					a_use = a + ( rs_a * ic ) + ( cs_a * pc );

					// Int8 kernel reads 4 elements, totalling 4 bytes in a
					// single broadcast for use in vnni instruction.
					// Non vnni based kernel requires update to this code.
					cs_a_use = 4;
					a_block_stride = rs_a;
				}

				post_ops_attr.b_sum_offset = 0;

				for ( dim_t jr = 0; jr < nc0; jr += NR )
				{
					dim_t nr0 = bli_min( ( nc0 - jr ), NR );

					// Post ops meta attributes.
					post_ops_attr.post_op_c_i = ic;
					post_ops_attr.post_op_c_j = ( jc + jr );
					post_ops_attr.rs_c_downscale = rs_c_downscale;
					//post_ops_attr.b_col_sum_vec = ( int32_t* )( b_use + ( rs_b * kc0_updated ) );

					// Reorder/Packed B, Reorder/Packed/Unpacked A call.
					( ( lpgemm_rowvar_s32_s8 )lcntx->kern_fun_ptr )
					(
					  mc0, nr0, kc0,
					  a_use, rs_a_use, cs_a_use, a_block_stride,
					  ( b_use + ( jr * kc0_updated ) ), rs_b_use, cs_b_use,
					  ( c_use_ic + jr ), rs_c_use, 1,
					  alpha, beta0,
					  post_op_list, post_ops_attr
					);
					post_ops_attr.b_sum_offset += NR;
				}
			}
		}
		if ( mtag_b == REORDERED )
		{
			adjust_B_panel_reordered_jc( &jc, jc_cur_loop );
		}
	}

	// Release pack buffers.
	if ( mtag_b == PACK )
	{
		// All threads in work group should wait till B matrix usage is
		// completed by the participating threads.
		bli_thrcomm_barrier
		(
		  bli_thread_ocomm_id( &thread_jc ),
		  &thread->comm[bli_thread_work_id( &thread_jc)]
		);

		if ( bli_thread_am_ochief( &thread_ic ) )
		{
			if ( bli_mem_is_alloc( &mem_b ) )
			{
				bli_membrk_release( rntm, &mem_b );
			}
		}
	}
	if ( mtag_a == PACK )
	{
		if ( bli_mem_is_alloc( &mem_a ) )
		{
			bli_membrk_release( rntm, &mem_a );
		}
	}
	if ( c_downscale == TRUE )
	{
		if ( bli_mem_is_alloc( &mem_scale_c ) )
		{
			bli_membrk_release( rntm, &mem_scale_c );
		}
	}
}
