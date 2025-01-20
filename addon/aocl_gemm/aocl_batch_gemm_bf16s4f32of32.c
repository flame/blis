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

AOCL_BGEMM_MATMUL(bfloat16,int8_t,float,float,bf16s4f32of32)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "bf16s4f32of32", \
	  order, transa, transb, \
	  batch_size, m, n, k, \
	  ( ( float* ) alpha ), \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  ( ( float* ) beta ), \
	  ldc, post_op_unparsed \
	);

	inc_t rs_a[batch_size];
	inc_t cs_a[batch_size];

	inc_t rs_b[batch_size];
	inc_t cs_b[batch_size];

	inc_t rs_c[batch_size];
	inc_t cs_c[batch_size];

	AOCL_MEMORY_TAG mtag_a[batch_size];
	AOCL_MEMORY_TAG mtag_b[batch_size];

	lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];
    lpgemm_pre_op pre_op_list[batch_size][AOCL_MAX_PRE_OPS];

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512bf16_supported() == FALSE )
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
				"cannot perform bf16s4f32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

#ifdef LPGEMM_BF16_JIT
	bli_print_msg(" WOQ is not supported by JIT kernels.", __FILE__, __LINE__ );
	return;
#endif

	trans_t blis_transa;
	trans_t blis_transb;

	// check for validity of params.
	int err_no = 0;

	for( dim_t bs_i = 0; bs_i < batch_size; bs_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
		  "batch_bf16s4f32of32",
		  order[bs_i], transa[bs_i], transb[bs_i],
		  bs_i,
		  m[bs_i], n[bs_i], k[bs_i],
		  a[bs_i], lda[bs_i], mem_format_a[bs_i],
		  b[bs_i], ldb[bs_i], mem_format_b[bs_i],
		  c[bs_i], ldc[bs_i],
		  err_no
		);

		if ( err_no != 0 )
		{
			goto err_hndl;
		}
		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[bs_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[bs_i], &blis_transb );

		bool is_column_major = ( ( order[bs_i] == 'c' ) || ( order[bs_i] == 'C' ) );

		if( is_column_major == TRUE )
		{
			bli_print_msg("Column major inputs not supported.",
					  __FILE__, __LINE__);
			goto err_hndl;
		}
		else // row-major
		{
			rs_a[bs_i] = lda[bs_i];
			cs_a[bs_i] = 1;

			if( bli_is_trans( blis_transa ) )
			{
				rs_a[bs_i] = 1;
				cs_a[bs_i] = lda[bs_i];
			}

			rs_b[bs_i] = ldb[bs_i];
			cs_b[bs_i] = 1;

			if( bli_is_trans( blis_transb ) )
			{
				rs_b[bs_i] = 1;
				cs_b[bs_i] = ldb[bs_i];
			}

			bli_param_map_char_to_lpmtag( mem_format_a[bs_i], &(mtag_a[bs_i]) );
			bli_param_map_char_to_lpmtag( mem_format_b[bs_i], &(mtag_b[bs_i]) );

			// Reorder is not supported for A matrix
			if(  mtag_a[bs_i] == REORDERED )
			{
				bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
				goto err_hndl;
			}
			// From 5-loop function point of view,
			// A matrix when in column major storage needs to be packed to row-major
			// storage as kernel expects A matrix to be in row-major format.
			if( bli_is_trans(blis_transa ) )
			{
				mtag_a[bs_i] = PACK;
			}
		}

		rs_c[bs_i] = ldc[bs_i];
		cs_c[bs_i] = 1;

		// From 5-loop function point of view
		// B matrix needs to be packed in a certain format in order to be loaded
		// and used in bf16 instrution. As such the mtag_b always needs to be either
		// packed or reordered. B matrix as it is (unpacked) cannot be used, and
		// the mtag_b is set to packed to enable runtime packing.
		if ( mtag_b[bs_i] == UNPACKED )
		{
			mtag_b[bs_i] = PACK;
		}

		// Convert pre op struct to pre op linked list format.
		err_t err = lpgemm_translate_to_pre_ops_list
		            (
		              post_op_unparsed[bs_i]->pre_ops,
		              pre_op_list[bs_i],
		              m[bs_i], n[bs_i], k[bs_i]
		            );
		if (err != BLIS_SUCCESS) goto err_hndl;

		// Convert post op struct to post op linked list format.
		err = lpgemm_translate_to_post_ops_list
		(
		post_op_unparsed[bs_i], post_op_list[bs_i],
		( void* )c[bs_i], ( void* )( (order + bs_i) ),
		m[bs_i], n[bs_i]
		);

		if( err != BLIS_SUCCESS ) goto err_hndl;

	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16S4F32OF32 );

#ifdef BLIS_ENABLE_OPENMP
	batch_lpgemm_bf16s4f32of32_openmp_thread_decorator
	(
	  batch_size, m, n, k,
	  a, rs_a, cs_a, mtag_a,
	  b, rs_b, cs_b, mtag_b,
	  c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  pre_op_list, post_op_list, F32
	);


#else
	batch_lpgemm_bf16s4f32of32_thread_decorator
	(
	  batch_size, m, n, k,
	  a, rs_a, cs_a, mtag_a,
	  b, rs_b, cs_b, mtag_b,
	  c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  pre_op_list, post_op_list, F32
	);
#endif

err_hndl:;
	LPGEMM_STOP_LOGGER();
}

AOCL_BGEMM_MATMUL(bfloat16,int8_t,bfloat16,float,bf16s4f32obf16)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "bf16s4f32obf16", \
	  order, transa, transb, \
	  batch_size, m, n, k, \
	  ( ( float* ) alpha ), \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  ( ( float* ) beta ), \
	  ldc, post_op_unparsed \
	);

	inc_t rs_a[batch_size];
	inc_t cs_a[batch_size];

	inc_t rs_b[batch_size];
	inc_t cs_b[batch_size];

	inc_t rs_c[batch_size];
	inc_t cs_c[batch_size];

	AOCL_MEMORY_TAG mtag_a[batch_size];
	AOCL_MEMORY_TAG mtag_b[batch_size];

	lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];
    lpgemm_pre_op pre_op_list[batch_size][AOCL_MAX_PRE_OPS];

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512bf16_supported() == FALSE )
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
				"cannot perform bf16bf16f32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

#ifdef LPGEMM_BF16_JIT
	bli_print_msg(" WOQ is not supported by JIT kernels.", __FILE__, __LINE__ );
	return;
#endif

	trans_t blis_transa;
	trans_t blis_transb;

	// check for validity of params.
	int err_no = 0;

	for( dim_t bs_i = 0; bs_i < batch_size; bs_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
		  "batch_bf16s4f32obf16",
		  order[bs_i], transa[bs_i], transb[bs_i],
		  bs_i,
		  m[bs_i], n[bs_i], k[bs_i],
		  a[bs_i], lda[bs_i], mem_format_a[bs_i],
		  b[bs_i], ldb[bs_i], mem_format_b[bs_i],
		  c[bs_i], ldc[bs_i],
		  err_no
		);

		if ( err_no != 0 )
		{
			goto err_hndl;
		}

		/* Map BLAS chars to their corresponding BLIS enumerated type value. */
		bli_param_map_netlib_to_blis_trans( transa[bs_i], &blis_transa );
		bli_param_map_netlib_to_blis_trans( transb[bs_i], &blis_transb );

		bool is_column_major = ( ( order[bs_i] == 'c' ) || ( order[bs_i] == 'C' ) );

		if( is_column_major == TRUE )
		{
			bli_print_msg("Column major inputs not supported.",
					  __FILE__, __LINE__);
			goto err_hndl;
		}
		else // row-major
		{
			rs_a[bs_i] = lda[bs_i];
			cs_a[bs_i] = 1;

			if( bli_is_trans( blis_transa ) )
			{
				rs_a[bs_i] = 1;
				cs_a[bs_i] = lda[bs_i];
			}

			rs_b[bs_i] = ldb[bs_i];
			cs_b[bs_i] = 1;

			if( bli_is_trans( blis_transb ) )
			{
				rs_b[bs_i] = 1;
				cs_b[bs_i] = ldb[bs_i];
			}

			bli_param_map_char_to_lpmtag( mem_format_a[bs_i], &(mtag_a[bs_i]) );
			bli_param_map_char_to_lpmtag( mem_format_b[bs_i], &(mtag_b[bs_i]) );

			// Reorder is not supported for A matrix
			if(  mtag_a[bs_i] == REORDERED )
			{
				bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
				goto err_hndl;
			}
			// From 5-loop function point of view,
			// A matrix when in column major storage needs to be packed to row-major
			// storage as kernel expects A matrix to be in row-major format.
			if( bli_is_trans(blis_transa ) )
			{
				mtag_a[bs_i] = PACK;
			}

		}

		rs_c[bs_i] = ldc[bs_i];
		cs_c[bs_i] = 1;

		// From 5-loop function point of view
		// B matrix needs to be packed in a certain format in order to be loaded
		// and used in bf16 instrution. As such the mtag_b always needs to be either
		// packed or reordered. B matrix as it is (unpacked) cannot be used, and
		// the mtag_b is set to packed to enable runtime packing.
		if ( mtag_b[bs_i] == UNPACKED )
		{
			mtag_b[bs_i] = PACK;
		}

		// Convert pre op struct to pre op linked list format.
		err_t err = lpgemm_translate_to_pre_ops_list
		            (
		              post_op_unparsed[bs_i]->pre_ops,
		              pre_op_list[bs_i],
		              m[bs_i], n[bs_i], k[bs_i]
		            );
		if (err != BLIS_SUCCESS) goto err_hndl;

		// Convert post op struct to post op linked list format.
		err = lpgemm_translate_to_post_ops_list
		      (
		        post_op_unparsed[bs_i], post_op_list[bs_i],
		        ( void* )c[bs_i], ( void* )( (order + bs_i) ),
		        m[bs_i], n[bs_i]
		      );

		if( err != BLIS_SUCCESS ) goto err_hndl;

	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16S4F32OF32 );

#ifdef BLIS_ENABLE_OPENMP
	batch_lpgemm_bf16s4f32of32_openmp_thread_decorator
	(
	  batch_size, m, n, k,
	  a, rs_a, cs_a, mtag_a,
	  b, rs_b, cs_b, mtag_b,
	  (float**)c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  pre_op_list, post_op_list, BF16
	);


#else
	batch_lpgemm_bf16s4f32of32_thread_decorator
	(
	  batch_size, m, n, k,
	  a, rs_a, cs_a, mtag_a,
	  b, rs_b, cs_b, mtag_b,
	  (float**)c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  pre_op_list, post_op_list, BF16
	);
#endif

err_hndl:;
	LPGEMM_STOP_LOGGER();
}
