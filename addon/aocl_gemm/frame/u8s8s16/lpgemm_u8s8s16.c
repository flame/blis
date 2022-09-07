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
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_packb_s16.h"
#include "lpgemm_kernels.h"
#include "lpgemm_utils.h"
#include "lpgemm_config.h"
#include "lpgemm_thrinfo_utils.h"

// B should always be packed.
LPGEMM_5LOOP(uint8_t,int8_t,int16_t,u8s8s16o16)
{
	const dim_t NC = lpgemm_get_block_size_NC_global_cntx( U8S8S16OS16 );
	const dim_t KC = lpgemm_get_block_size_KC_global_cntx( U8S8S16OS16 );
	const dim_t MC = lpgemm_get_block_size_MC_global_cntx( U8S8S16OS16 );
	const dim_t NR = lpgemm_get_block_size_NR_global_cntx( U8S8S16OS16 );
	const dim_t MR = lpgemm_get_block_size_MR_global_cntx( U8S8S16OS16 );
	
	if (mtag_b == UNPACKED)
	{
		// Error: can only work with packed B now.
		return;
	}

	const int8_t *b_use;
	const uint8_t *a_use;
	dim_t rs_a_use = rs_a;
	dim_t cs_a_use = cs_a;

	dim_t rs_b_use = rs_b;
	dim_t cs_b_use = cs_b;

	int16_t *c_use_jc = NULL;
	int16_t *c_use_ic = NULL;
	dim_t rs_c_use = rs_c;
	dim_t rs_c_downscale = rs_c;

	// Pack buffer for B.
	int8_t *pack_b_buffer_u8s8s16o16;
	mem_t mem_b = BLIS_MEM_INITIALIZER;
	dim_t packb_min_NR = 16;
	siz_t mem_b_size_req = 0;

	// Temporary buffer for C accumulation when downscaling is required.
	int16_t* temp_scal_c_buffer_u8s8s16o16;
	mem_t mem_scale_c = BLIS_MEM_INITIALIZER;
	siz_t mem_scale_c_size_req = 0;

	// Making multiple of 2 to suit k in vpmaddubsw
	dim_t k_updated = make_multiple_of_n( k, 2 );

	// Is required to decide whether to apply post ops or not.
	bool is_last_k = FALSE;

	// Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
	thrinfo_t thread_jc;
	thrinfo_t thread_ic;

	lpgemm_gen_thrinfo(thread, &thread_jc, &thread_ic);

	// Compute the JC, IC loop thread range for the current thread.
	dim_t jc_start, jc_end;
	bli_thread_range_sub(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

	dim_t ic_start, ic_end;
	bli_thread_range_sub(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

	for (dim_t jc = jc_start; jc < jc_end; jc += NC)
	{
		dim_t nc0 = bli_min((jc_end - jc), NC);

		dim_t jc_cur_loop = jc;
		dim_t jc_cur_loop_rem = 0;
		dim_t n_sub_updated = 0;

		if (mtag_b == REORDERED)
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
			mem_scale_c_size_req = sizeof( int16_t ) * nc0 * ( ic_end - ic_start );

			lpgemm_alloc_mem_panel
			(
			  mem_scale_c_size_req, BLIS_BUFFER_FOR_C_PANEL,
			  &mem_scale_c, rntm
			);

			temp_scal_c_buffer_u8s8s16o16 = bli_mem_buffer( &mem_scale_c );

			c_use_jc = ( int16_t* )temp_scal_c_buffer_u8s8s16o16;

			if ( beta != 0 )
			{
				dim_t i_temp = 0;
				dim_t j_temp = 0;
				// Upscale out C to temporary C matrix.
				for ( dim_t i_dscale = ic_start; i_dscale < ic_end; ++i_dscale )
				{
					j_temp = 0;
					for ( dim_t j_dscale = jc; j_dscale < ( jc + nc0 ); ++j_dscale )
					{
						*( temp_scal_c_buffer_u8s8s16o16 +
								( nc0 * i_temp ) + j_temp ) =
								( int16_t )( *( ( ( int8_t* )c ) +
								( rs_c * i_dscale ) + j_dscale ) );

						j_temp++;
					}
					i_temp++;
				}
			}

			// The temp c buffer stride is modified as opposed to original C matrix.
			rs_c_use = nc0;
		}

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			int16_t beta0 = (pc == 0) ? beta : 1;
			dim_t kc0 = bli_min((k - pc), KC);

			is_last_k = ( ( pc + KC ) >= k ) ? ( TRUE ) : ( FALSE );

			// kc0 needs to be a multiple of 2 so that it can be
			// used with vpmaddubsw instruction. Padding is added in
			// cases this condition is not satisfied, and therefore
			// the kc0 offsets used for packed/reordered buffers
			// needs to be updated.
			dim_t kc0_updated = make_multiple_of_n(kc0, 2);

			if (mtag_b == PACK)
			{
				// Pack B chunks are based on jc work id.
				dim_t jc_work_id = bli_thread_work_id(&thread_jc);

				// Using child thrinfo (thread_ic) tid to decide chief thread
				// per B matrix chunk (jc work id group)
				if (bli_thread_am_ochief(&thread_ic))
				{
					// nc0 needs to be a multiple of 16 since this gives maximum
					// vectorization. Packing B always results in buffers with width
					// which is a multiple of 16. Subsequently the nc0 offsets used
					// for packed/reordered buffers needs to be updated.
					dim_t nc0_updated = make_multiple_of_n(nc0, packb_min_NR);
					mem_b_size_req = sizeof(int8_t) * nc0_updated * kc0_updated;

					lpgemm_alloc_mem_panel(
						mem_b_size_req, BLIS_BUFFER_FOR_B_PANEL,
						&mem_b, rntm);

					thread->comm[jc_work_id].sent_object =
						bli_mem_buffer(&mem_b);
				}

				// All threads in work group should wait till chief thread has
				// finished allocating the packing buffers.
				bli_thrcomm_barrier(
					bli_thread_ocomm_id(&thread_ic),
					&thread->comm[jc_work_id]);

				pack_b_buffer_u8s8s16o16 =
					(int8_t *)thread->comm[jc_work_id].sent_object;

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

				// Ensure thread ranges are valid, especially cases where no:
				// of threads available for parallelization are greater than
				// no: of B panel NR chunks.
				if ((jc_packb_end > jc_packb_start) &&
					(jc_packb_start < (jc + nc0)))
				{
					packb_nr32_u8s8s16o16
					(
						pack_b_buffer_u8s8s16o16 + 
						 (jc_packb_start * kc0_updated),
						(b + (rs_b * pc) + (cs_b * jc) +
						 (cs_b * jc_packb_start)),
						rs_b,
						(jc_packb_end - jc_packb_start), kc0,
						&rs_b_use, &cs_b_use
						);
				}
				else
				{
					get_packb_nr32_u8s8s16o16_strides(&rs_b_use, &cs_b_use);
				}

				// All threads in work group should wait till B matrix packing
				// is completed by the participating threads.
				bli_thrcomm_barrier
				(
					bli_thread_ocomm_id(&thread_ic),
					&thread->comm[jc_work_id]
				);

				b_use = pack_b_buffer_u8s8s16o16;
			}
			else if (mtag_b == REORDERED)
			{
				// In multi-threaded scenarios, an extra offset into a given
				// packed B panel is required, since the jc loop split can
				// result in per thread start offset inside the panel, instead
				// of panel boundaries.
				b_use = b + (jc_cur_loop * k_updated) +
						(n_sub_updated * pc) +
						(jc_cur_loop_rem * kc0_updated);

				get_packb_nr32_u8s8s16o16_strides(&rs_b_use, &cs_b_use);
			}
			else
			{
				// Unpacked B not supported.
				return;
			}

			for (dim_t ic = ic_start; ic < ic_end; ic += MC)
			{
				dim_t mc0 = bli_min((ic_end - ic), MC);

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

				a_use = a + (rs_a * ic) + (cs_a * pc);
				cs_a_use = 1;

				dim_t a_block_stride = rs_a;

				for (dim_t jr = 0; jr < nc0; jr += NR)
				{
					dim_t nr0 = bli_min((nc0 - jr), NR);

					// Calls for reorder B
					lpgemm_rowvar_u8s8s16o16_6x32
					(
						mc0, nr0, kc0,
						a_use, rs_a_use, cs_a_use, a_block_stride,
						(b_use + (jr * kc0_updated)), rs_b_use, cs_b_use,
						(c_use_ic + jr), rs_c_use, 1,
						alpha, beta0,
					  	is_last_k, ic, ( jc + jr ), post_op_list, rs_c_downscale
					);
				}
			}
		}

		if (mtag_b == REORDERED)
		{
			adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
		}
	}

	// Release pack buffers.
	if (mtag_b == PACK)
	{
		// All threads in work group should wait till B matrix usage is
		// completed by the participating threads.
		bli_thrcomm_barrier(
			bli_thread_ocomm_id(&thread_jc),
			&thread->comm[bli_thread_work_id(&thread_jc)]);

		if (bli_thread_am_ochief(&thread_ic))
		{
			if (bli_mem_is_alloc(&mem_b))
			{
				bli_membrk_release(rntm, &mem_b);
			}
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
