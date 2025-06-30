/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "aocl_gemm_check.h"
#include "lpgemm_types.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_thread_decor_openmp.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_config.h"
#include "lpgemm_utils.h"
#include "lpgemm_logger.h"

AOCL_BGEMM_MATMUL(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "s8s8s32os32", \
	  order, transa, transb, \
	  group_count, group_size, m, n, k, \
	  alpha, \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  beta, \
	  ldc, post_op_unparsed \
	);

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512vnni_supported() == FALSE )
	{
		bli_print_msg(" AVX512_VNNI ISA not supported by processor, "
				"cannot perform s8s8s32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// offset to get subsequent matrix when group_count > 1
	dim_t mat_idx = 0;

	// check for validity of params.
	int err_no = 0;

	for( dim_t gc_i = 0; gc_i < group_count; gc_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
			"batch_s8s8s32os32",
			order[gc_i], transa[gc_i], transb[gc_i],
			group_count, group_size[gc_i],
			m[gc_i], n[gc_i], k[gc_i],
			lda[gc_i], mem_format_a[gc_i],
			ldb[gc_i], mem_format_b[gc_i],
			ldc[gc_i],
			err_no
		);

		if ( err_no != 0 )
		{
			goto err_hndl;
		}

		dim_t g_sz = group_size[gc_i];

		inc_t rs_a[g_sz];
		inc_t cs_a[g_sz];

		inc_t rs_b[g_sz];
		inc_t cs_b[g_sz];

		inc_t rs_c[g_sz];
		inc_t cs_c[g_sz];

		AOCL_MEMORY_TAG mtag_a[g_sz];
		AOCL_MEMORY_TAG mtag_b[g_sz];

		int8_t *a_local[g_sz];
		int8_t *b_local[g_sz];
		dim_t m_local[g_sz], n_local[g_sz], k_local[g_sz];

		// Convert post op struct to post op linked list format.
		lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

		err_t err = lpgemm_translate_to_post_ops_list
			(
				post_op_unparsed[gc_i], post_op_list,
				( void* )c[gc_i], ( void* )( order +  gc_i ),
				m[gc_i], n[gc_i]
			);

		if( err != BLIS_SUCCESS ) goto err_hndl;

		trans_t blis_transa;
		trans_t blis_transb;

		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[gc_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[gc_i], &blis_transb );

		bool is_column_major = ((order[gc_i] == 'c') || (order[gc_i] == 'C'));

		for( dim_t gs_i = 0; gs_i < g_sz; gs_i++ )
		{
			if ( is_column_major == TRUE )
			{
				// Column major support disabled for int API's till micro-kernel
				// post-ops are updated to account for column major.
				if (post_op_unparsed[gs_i] != NULL )
				{
					bli_print_msg("Column major inputs not supported with Post-ops.",
								__FILE__, __LINE__);
					goto err_hndl;
				}

				rs_a[gs_i] = ldb[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = ldb[gc_i];
				}

				rs_b[gs_i] = lda[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = lda[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_b[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_a[gs_i]) );

				// Inputs swapped in column major, A becomes B from kernel point of view.
				// Reorder is not supported for column major matrices.
				if ( ( ( mtag_b[gs_i] == REORDERED ) || ( mtag_a[gs_i] == REORDERED ) ) )
				{
					bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				// Inputs swapped in column major, A becomes B from kernel point of view.
				if ( bli_is_trans(blis_transb ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// swap m & n in case of col-major matrices
				m_local[gs_i] = n[gc_i];
				n_local[gs_i] = m[gc_i];

				// swap a & b pointers in case of col-major matrices
				a_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
			}
			else // row-major
			{
				rs_a[gs_i] = lda[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = lda[gc_i];
				}

				rs_b[gs_i] = ldb[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = ldb[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_a[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_b[gs_i]) );

				// Reorder is not supported for A matrix
				if(  mtag_a[gs_i] == REORDERED )
				{
					bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				if( bli_is_trans(blis_transa ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// copy the values of m & n
				m_local[gs_i] = m[gc_i];
				n_local[gs_i] = n[gc_i];

				// copy the values of a & b pointers
				a_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
			}

			k_local[gs_i] = k[gc_i];

			rs_c[gs_i] = ldc[gc_i];
			cs_c[gs_i] = 1;

			// From 5-loop function point of view
			// B matrix needs to be packed in a certain format in order to be loaded
			// and used in bf16 instrution. As such the mtag_b always needs to be either
			// packed or reordered. B matrix as it is (unpacked) cannot be used, and
			// the mtag_b is set to packed to enable runtime packing.
			if ( mtag_b[gs_i] == UNPACKED )
			{
				mtag_b[gs_i] = PACK;
			}
		}

		// Initialize a local runtime with global settings if necessary. Note
		// that in the case that a runtime is passed in, we make a local copy.
		rntm_t rntm_g;
		bli_rntm_init_from_global( &rntm_g );
		bli_pba_rntm_set_pba( &rntm_g );

		lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S32OS32 );

	#ifdef BLIS_ENABLE_OPENMP
		batch_lpgemm_s8s8s32o32_openmp_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, S32
		);
	#else
		batch_lpgemm_s8s8s32o32_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, S32
		);
	#endif
		mat_idx += g_sz;
	}

err_hndl:;
	LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "s8s8s32os8", \
	  order, transa, transb, \
	  group_count, group_size, m, n, k, \
	  alpha, \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  beta, \
	  ldc, post_op_unparsed \
	);

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512vnni_supported() == FALSE )
	{
		bli_print_msg(" AVX512_VNNI ISA not supported by processor, "
				"cannot perform s8s8s32obf16 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// offset to get subsequent matrix when group_count > 1
	dim_t mat_idx = 0;

	// check for validity of params.
	int err_no = 0;

	for( dim_t gc_i = 0; gc_i < group_count; gc_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
			"batch_s8s8s32os8",
			order[gc_i], transa[gc_i], transb[gc_i],
			group_count, group_size[gc_i],
			m[gc_i], n[gc_i], k[gc_i],
			lda[gc_i], mem_format_a[gc_i],
			ldb[gc_i], mem_format_b[gc_i],
			ldc[gc_i],
			err_no
		);

		if( err_no != 0 )
		{
			goto err_hndl;
		}

		// Group_size is used across
		dim_t g_sz = group_size[gc_i];

		trans_t blis_transa;
		trans_t blis_transb;

		inc_t rs_a[g_sz];
		inc_t cs_a[g_sz];

		inc_t rs_b[g_sz];
		inc_t cs_b[g_sz];

		inc_t rs_c[g_sz];
		inc_t cs_c[g_sz];

		AOCL_MEMORY_TAG mtag_a[g_sz];
		AOCL_MEMORY_TAG mtag_b[g_sz];

		int8_t *a_local[g_sz], *b_local[g_sz];
		dim_t m_local[g_sz], n_local[g_sz], k_local[g_sz];

		// Convert post op struct to post op linked list format.
		lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

		err_t err = lpgemm_translate_to_post_ops_list
			(
				post_op_unparsed[gc_i], post_op_list,
				( void* )c[gc_i], ( void* )( order + gc_i ),
				m[gc_i], n[gc_i]
			);

		if( err != BLIS_SUCCESS ) goto err_hndl;

		bli_param_map_netlib_to_blis_trans( transa[gc_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[gc_i], &blis_transb );

		bool is_column_major = ( (order[gc_i] == 'c') || (order[gc_i] == 'C'));

		for( dim_t gs_i = 0; gs_i < g_sz; gs_i++ )
		{
			if( is_column_major == TRUE )
			{
				// Column major support disabled for int API's till micro-kernel
				// post-ops are updated to account for column major
				if (post_op_unparsed[mat_idx + gs_i] != NULL )
				{
					bli_print_msg("Column major inputs not supported with Post-ops.",
								__FILE__, __LINE__);
					goto err_hndl;
				}

				rs_a[gs_i] = ldb[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = ldb[gc_i];
				}

				rs_b[gs_i] = lda[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = lda[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_b[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_a[gs_i]) );

				if ( ( ( mtag_b[gs_i] == REORDERED ) || ( mtag_a[gs_i] == REORDERED ) ) )
				{
					bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
					goto err_hndl;
				}

				if ( bli_is_trans(blis_transb ) )
				{
					mtag_a[gs_i] = PACK;
				}

				m_local[gs_i] = n[gc_i];
				n_local[gs_i] = m[gc_i];

				a_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
			}
			else //row_major
			{
				rs_a[gs_i] = lda[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = lda[gc_i];
				}

				rs_b[gs_i] = ldb[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = ldb[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_a[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_b[gs_i]) );

				if(  mtag_a[gs_i] == REORDERED )
				{
					bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
					goto err_hndl;
				}

				if( bli_is_trans(blis_transa ) )
				{
					mtag_a[gs_i] = PACK;
				}

				m_local[gs_i] = m[gc_i];
				n_local[gs_i] = n[gc_i];

				a_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
			}

			k_local[gs_i] = k[gc_i];

			rs_c[gs_i] = ldc[gc_i];
			cs_c[gs_i] = 1;

			if ( mtag_b[gs_i] == UNPACKED )
			{
				mtag_b[gs_i] = PACK;
			}
		}

		// Initialize a local runtime with global settings if necessary. Note
		// that in the case that a runtime is passed in, we make a local copy.
		rntm_t rntm_g;
		bli_rntm_init_from_global( &rntm_g );
		bli_pba_rntm_set_pba( &rntm_g );

		lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S32OS32 );

	#ifdef BLIS_ENABLE_OPENMP
		batch_lpgemm_s8s8s32o32_openmp_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, S8
		);
	#else
		batch_lpgemm_s8s8s32o32_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, S8
		);
	#endif
		mat_idx += g_sz;
	}
err_hndl:;
	LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(int8_t,int8_t,float,int32_t,s8s8s32of32)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "s8s8s32of32", \
	  order, transa, transb, \
	  group_count, group_size, m, n, k, \
	  alpha, \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  beta, \
	  ldc, post_op_unparsed \
	);

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512vnni_supported() == FALSE )
	{
		bli_print_msg(" AVX512_VNNI ISA not supported by processor, "
				"cannot perform s8s8s32of32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	trans_t blis_transa;
	trans_t blis_transb;

	// offset to get subsequent matrix when group_count > 1
	dim_t mat_idx = 0;

	// check for validity of params.
	int err_no = 0;

	for( dim_t gc_i = 0; gc_i < group_count; gc_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
			"batch_s8s8s32of32",
			order[gc_i], transa[gc_i], transb[gc_i],
			group_count, group_size[gc_i],
			m[gc_i], n[gc_i], k[gc_i],
			lda[gc_i], mem_format_a[gc_i],
			ldb[gc_i], mem_format_b[gc_i],
			ldc[gc_i],
			err_no
		);

		if( err_no != 0 )
		{
			goto err_hndl;
		}

		dim_t g_sz = group_size[gc_i];

		inc_t rs_a[g_sz];
		inc_t cs_a[g_sz];

		inc_t rs_b[g_sz];
		inc_t cs_b[g_sz];

		inc_t rs_c[g_sz];
		inc_t cs_c[g_sz];

		AOCL_MEMORY_TAG mtag_a[g_sz];
		AOCL_MEMORY_TAG mtag_b[g_sz];

		int8_t *a_local[g_sz];
		int8_t *b_local[g_sz];
		dim_t m_local[g_sz], n_local[g_sz], k_local[g_sz];

		// Convert post op struct to post op linked list format.
		lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

		err_t err = lpgemm_translate_to_post_ops_list
			(
				post_op_unparsed[gc_i], post_op_list,
				( void* )c[gc_i], ( void* )( order + gc_i ),
				m[gc_i], n[gc_i]
			);

			if( err != BLIS_SUCCESS )
				goto err_hndl;

		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[gc_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[gc_i], &blis_transb );

		bool is_column_major = ( ( order[gc_i] == 'c' ) || ( order[gc_i] == 'C' ) );

		for( dim_t gs_i = 0; gs_i < g_sz; gs_i++ )
		{
			if( is_column_major == TRUE )
			{
				// Column major support disabled for int API's till micro-kernel
				// post-ops are updated to account for column major.
				if (post_op_unparsed[mat_idx + gs_i] != NULL )
				{
					bli_print_msg("Column major inputs not supported with Post-ops.",
								__FILE__, __LINE__);
					goto err_hndl;
				}

				rs_a[gs_i] = ldb[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = ldb[gc_i];
				}

				rs_b[gs_i] = lda[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = lda[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_b[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_a[gs_i]) );

				// Inputs swapped in column major, A becomes B from kernel point of view.
				// Reorder is not supported for column major matrices.
				if ( ( ( mtag_b[gs_i] == REORDERED ) || ( mtag_a[gs_i] == REORDERED ) ) )
				{
					bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				// Inputs swapped in column major, A becomes B from kernel point of view.
				if ( bli_is_trans(blis_transb ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// swap m & n in case of col-major matrices
				m_local[gs_i] = n[gc_i];
				n_local[gs_i] = m[gc_i];

				// swap a & b pointers in case of col-major matrices
				a_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
			}
			else // row-major
			{
				rs_a[gs_i] = lda[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = lda[gc_i];
				}

				rs_b[gs_i] = ldb[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = ldb[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_a[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_b[gs_i]) );

				// Reorder is not supported for A matrix
				if(  mtag_a[gs_i] == REORDERED )
				{
					bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				if( bli_is_trans(blis_transa ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// copy the values of m & n
				m_local[gs_i] = m[gc_i];
				n_local[gs_i] = n[gc_i];

				// copy the values of a & b pointers
				a_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
			}

			k_local[gs_i] = k[gc_i];

			rs_c[gs_i] = ldc[gc_i];
			cs_c[gs_i] = 1;

			// From 5-loop function point of view
			// B matrix needs to be packed in a certain format in order to be loaded
			// and used in bf16 instrution. As such the mtag_b always needs to be either
			// packed or reordered. B matrix as it is (unpacked) cannot be used, and
			// the mtag_b is set to packed to enable runtime packing.
			if ( mtag_b[gs_i] == UNPACKED )
			{
				mtag_b[gs_i] = PACK;
			}
		}

		// Initialize a local runtime with global settings if necessary. Note
		// that in the case that a runtime is passed in, we make a local copy.
		rntm_t rntm_g;
		bli_rntm_init_from_global( &rntm_g );
		bli_pba_rntm_set_pba( &rntm_g );

		lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S32OS32 );

	#ifdef BLIS_ENABLE_OPENMP
		batch_lpgemm_s8s8s32o32_openmp_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, F32
		);


	#else
		batch_lpgemm_s8s8s32o32_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, F32
		);
	#endif
		mat_idx += g_sz;
	}


err_hndl:;
	LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(int8_t,int8_t,bfloat16,int32_t,s8s8s32obf16)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "s8s8s32obf16", \
	  order, transa, transb, \
	  group_count, group_size, m, n, k, \
	  alpha, \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  beta, \
	  ldc, post_op_unparsed \
	);

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512vnni_supported() == FALSE )
	{
		bli_print_msg(" AVX512_VNNI ISA not supported by processor, "
				"cannot perform s8s8s32obf16 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// offset to get subsequent matrix when group_count > 1
	dim_t mat_idx = 0;

	// check for validity of params.
	int err_no = 0;

	for( dim_t gc_i = 0; gc_i < group_count; gc_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
			"batch_s8s8s32obf16",
			order[gc_i], transa[gc_i], transb[gc_i],
			group_count, group_size[gc_i],
			m[gc_i], n[gc_i], k[gc_i],
			lda[gc_i], mem_format_a[gc_i],
			ldb[gc_i], mem_format_b[gc_i],
			ldc[gc_i],
			err_no
		);

		if( err_no != 0 )
		{
			goto err_hndl;
		}

		// Group_size is used across
		dim_t g_sz = group_size[gc_i];

		trans_t blis_transa;
		trans_t blis_transb;

		inc_t rs_a[g_sz];
		inc_t cs_a[g_sz];

		inc_t rs_b[g_sz];
		inc_t cs_b[g_sz];

		inc_t rs_c[g_sz];
		inc_t cs_c[g_sz];

		AOCL_MEMORY_TAG mtag_a[g_sz];
		AOCL_MEMORY_TAG mtag_b[g_sz];

		int8_t *a_local[g_sz];
		int8_t *b_local[g_sz];
		dim_t m_local[g_sz], n_local[g_sz], k_local[g_sz];

		// Convert post op struct to post op linked list format.
		lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

		err_t err = lpgemm_translate_to_post_ops_list
			(
				post_op_unparsed[gc_i], post_op_list,
				( void* )c[gc_i], ( void* )( order +  gc_i ),
				m[gc_i], n[gc_i]
			);

		if( err != BLIS_SUCCESS ) goto err_hndl;

		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[gc_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[gc_i], &blis_transb );

		bool is_column_major = ( ( order[gc_i] == 'c' ) || ( order[gc_i] == 'C' ) );

		for( dim_t gs_i = 0; gs_i < g_sz; gs_i++ )
		{
			if( is_column_major == TRUE )
			{
				// Column major support disabled for int API's till micro-kernel
				// post-ops are updated to account for column major.
				if (post_op_unparsed[mat_idx + gs_i] != NULL )
				{
					bli_print_msg("Column major inputs not supported with Post-ops.",
								__FILE__, __LINE__);
					goto err_hndl;
				}

				rs_a[gs_i] = ldb[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = ldb[gc_i];
				}

				rs_b[gs_i] = lda[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = lda[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_b[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_a[gs_i]) );

				// Inputs swapped in column major, A becomes B from kernel point of view.
				// Reorder is not supported for column major matrices.
				if ( ( ( mtag_b[gs_i] == REORDERED ) || ( mtag_a[gs_i] == REORDERED ) ) )
				{
					bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				// Inputs swapped in column major, A becomes B from kernel point of view.
				if ( bli_is_trans(blis_transb ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// swap m & n in case of col-major matrices
				m_local[gs_i] = n[gc_i];
				n_local[gs_i] = m[gc_i];

				// swap a & b pointers in case of col-major matrices
				a_local[gs_i] = (int8_t*)(b[mat_idx+gs_i]);
				b_local[gs_i] = (int8_t*)(a[mat_idx+gs_i]);
			}
			else // row-major
			{
				rs_a[gs_i] = lda[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = lda[gc_i];
				}

				rs_b[gs_i] = ldb[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = ldb[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_a[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_b[gs_i]) );

				// Reorder is not supported for A matrix
				if(  mtag_a[gs_i] == REORDERED )
				{
					bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				if( bli_is_trans(blis_transa ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// copy the values of m & n
				m_local[gs_i] = m[gc_i];
				n_local[gs_i] = n[gc_i];

				a_local[gs_i] = (int8_t*)(a[mat_idx + gs_i]);
				b_local[gs_i] = (int8_t*)(b[mat_idx + gs_i]);
			}

			k_local[gs_i] = k[gc_i];

			rs_c[gs_i] = ldc[gc_i];
			cs_c[gs_i] = 1;

			// From 5-loop function point of view
			// B matrix needs to be packed in a certain format in order to be loaded
			// and used in bf16 instrution. As such the mtag_b always needs to be either
			// packed or reordered. B matrix as it is (unpacked) cannot be used, and
			// the mtag_b is set to packed to enable runtime packing.
			if ( mtag_b[gs_i] == UNPACKED )
			{
				mtag_b[gs_i] = PACK;
			}
		}

		// Initialize a local runtime with global settings if necessary. Note
		// that in the case that a runtime is passed in, we make a local copy.
		rntm_t rntm_g;
		bli_rntm_init_from_global( &rntm_g );
		bli_pba_rntm_set_pba( &rntm_g );

		lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S32OS32 );

	#ifdef BLIS_ENABLE_OPENMP
		batch_lpgemm_s8s8s32o32_openmp_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, BF16
		);


	#else
		batch_lpgemm_s8s8s32o32_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, BF16
		);
	#endif
		mat_idx+=g_sz;
	}
err_hndl:;
	LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(int8_t,int8_t,uint8_t,int32_t,s8s8s32ou8)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "s8s8s32ou8", \
	  order, transa, transb, \
	  group_count, group_size, m, n, k, \
	  alpha, \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  beta, \
	  ldc, post_op_unparsed \
	);

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512vnni_supported() == FALSE )
	{
		bli_print_msg(" AVX512_VNNI ISA not supported by processor, "
				"cannot perform s8s8s32ou8 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	trans_t blis_transa;
	trans_t blis_transb;

	// offset to get subsequent matrix when group_count > 1
	dim_t mat_idx = 0;

	// check for validity of params.
	int err_no = 0;

	for( dim_t gc_i = 0; gc_i < group_count; gc_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
			"batch_s8s8s32ou8",
			order[gc_i], transa[gc_i], transb[gc_i],
			group_count, group_size[gc_i],
			m[gc_i], n[gc_i], k[gc_i],
			lda[gc_i], mem_format_a[gc_i],
			ldb[gc_i], mem_format_b[gc_i],
			ldc[gc_i],
			err_no
		);

		if( err_no != 0 )
		{
			goto err_hndl;
		}

		dim_t g_sz = group_size[gc_i];

		inc_t rs_a[g_sz];
		inc_t cs_a[g_sz];

		inc_t rs_b[g_sz];
		inc_t cs_b[g_sz];

		inc_t rs_c[g_sz];
		inc_t cs_c[g_sz];

		AOCL_MEMORY_TAG mtag_a[g_sz];
		AOCL_MEMORY_TAG mtag_b[g_sz];

		int8_t *a_local[g_sz];
		int8_t *b_local[g_sz];
		dim_t m_local[g_sz], n_local[g_sz], k_local[g_sz];

		// Convert post op struct to post op linked list format.
		lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

		err_t err = lpgemm_translate_to_post_ops_list
		(
			post_op_unparsed[gc_i], post_op_list,
			( void* )c[gc_i], ( void* )( order + gc_i ),
			m[gc_i], n[gc_i]
		);

		if( err != BLIS_SUCCESS ) goto err_hndl;


		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[gc_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[gc_i], &blis_transb );

		bool is_column_major = ( ( order[gc_i] == 'c' ) || ( order[gc_i] == 'C' ) );

		for( dim_t gs_i = 0; gs_i < g_sz; gs_i++ )
		{
			if( is_column_major == TRUE )
			{
				// Column major support disabled for int API's till micro-kernel
				// post-ops are updated to account for column major.
				if (post_op_unparsed[gs_i] != NULL )
				{
					bli_print_msg("Column major inputs not supported with Post-ops.",
								__FILE__, __LINE__);
					goto err_hndl;
				}

				rs_a[gs_i] = ldb[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = ldb[gc_i];
				}

				rs_b[gs_i] = lda[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = lda[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_b[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_a[gs_i]) );

				// Inputs swapped in column major, A becomes B from kernel point of view.
				// Reorder is not supported for column major matrices.
				if ( ( ( mtag_b[gs_i] == REORDERED ) || ( mtag_a[gs_i] == REORDERED ) ) )
				{
					bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				// Inputs swapped in column major, A becomes B from kernel point of view.
				if ( bli_is_trans(blis_transb ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// swap m & n in case of col-major matrices
				m_local[gs_i] = n[gc_i];
				n_local[gs_i] = m[gc_i];

				// swap a & b pointers in case of col-major matrices
				a_local[gs_i] = (int8_t*)(b[mat_idx+gs_i]);
				b_local[gs_i] = (int8_t*)(a[mat_idx+gs_i]);
			}
			else // row-major
			{
				rs_a[gs_i] = lda[gc_i];
				cs_a[gs_i] = 1;

				if( bli_is_trans( blis_transa ) )
				{
					rs_a[gs_i] = 1;
					cs_a[gs_i] = lda[gc_i];
				}

				rs_b[gs_i] = ldb[gc_i];
				cs_b[gs_i] = 1;

				if( bli_is_trans( blis_transb ) )
				{
					rs_b[gs_i] = 1;
					cs_b[gs_i] = ldb[gc_i];
				}

				bli_param_map_char_to_lpmtag( mem_format_a[gc_i], &(mtag_a[gs_i]) );
				bli_param_map_char_to_lpmtag( mem_format_b[gc_i], &(mtag_b[gs_i]) );

				// Reorder is not supported for A matrix
				if(  mtag_a[gs_i] == REORDERED )
				{
					bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
					goto err_hndl;
				}
				// From 5-loop function point of view,
				// A matrix when in column major storage needs to be packed to row-major
				// storage as kernel expects A matrix to be in row-major format.
				if( bli_is_trans(blis_transa ) )
				{
					mtag_a[gs_i] = PACK;
				}

				// copy the values of m & n
				m_local[gs_i] = m[gc_i];
				n_local[gs_i] = n[gc_i];

				// copy the values of a & b pointers
				a_local[gs_i] = (int8_t*)(a[mat_idx+gs_i]);
				b_local[gs_i] = (int8_t*)(b[mat_idx+gs_i]);
			}

			k_local[gs_i] = k[gc_i];

			rs_c[gs_i] = ldc[gc_i];
			cs_c[gs_i] = 1;

			// From 5-loop function point of view
			// B matrix needs to be packed in a certain format in order to be loaded
			// and used in bf16 instrution. As such the mtag_b always needs to be either
			// packed or reordered. B matrix as it is (unpacked) cannot be used, and
			// the mtag_b is set to packed to enable runtime packing.
			if ( mtag_b[gs_i] == UNPACKED )
			{
				mtag_b[gs_i] = PACK;
			}
		}

		// Initialize a local runtime with global settings if necessary. Note
		// that in the case that a runtime is passed in, we make a local copy.
		rntm_t rntm_g;
		bli_rntm_init_from_global( &rntm_g );
		bli_pba_rntm_set_pba( &rntm_g );

		lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S32OS32 );

	#ifdef BLIS_ENABLE_OPENMP
		batch_lpgemm_s8s8s32o32_openmp_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, U8
		);


	#else
		batch_lpgemm_s8s8s32o32_thread_decorator
		(
			g_sz, m_local, n_local, k_local,
			(const int8_t**)a_local, rs_a, cs_a, mtag_a,
			(const int8_t**)b_local, rs_b, cs_b, mtag_b,
			(int32_t**)&c[mat_idx], rs_c, cs_c,
			alpha[gc_i], beta[gc_i],
			&rntm_g, lcntx_g,
			post_op_list, U8
		);
	#endif
		mat_idx+=g_sz;
	}
err_hndl:;
	LPGEMM_STOP_LOGGER();
}

