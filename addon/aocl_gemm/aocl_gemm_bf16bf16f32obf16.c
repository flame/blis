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
#include "aocl_gemm_interface_apis.h"
#include "lpgemm_types.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_thread_decor_openmp.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_config.h"
#include "lpgemm_utils.h"

AOCL_GEMM_MATMUL(bfloat16,bfloat16,bfloat16,bf16bf16f32obf16)
{
	trans_t blis_transa;
	trans_t blis_transb;

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512_bf16_supported() == FALSE )
	{
		printf(" AVX512_BF16 ISA not supported by processor, cannot perform lpgemm.\n");
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// Null check for pointers.
	if ( ( a == NULL ) || ( b == NULL ) || ( c == NULL ) )
	{
		return; // Error.
	}

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans( transa, &blis_transa );
	bli_param_map_netlib_to_blis_trans( transb, &blis_transb );

	/* Perform BLAS parameter checking. */
	// Transpose not supported.
	if ( ( blis_transa != BLIS_NO_TRANSPOSE ) ||
	     ( blis_transb != BLIS_NO_TRANSPOSE ) )
	{
		return; // Error.
	}

	// Sanitize order input.
	char order_use =
			( ( order == 'r' ) || ( order == 'R' ) ||
			  ( order == 'c' ) || ( order == 'C' ) ) ?
			order : 'r';

	bool is_row_major = ( ( order_use == 'r' ) || ( order_use == 'R' ) );
	bool is_column_major = ( ( order_use == 'c' ) || ( order_use == 'C' ) );

	// Row major input expected with leading dimensions >= row stride.
	if ( ( is_row_major == TRUE ) &&
		 ( ( lda < k ) || ( ldb < n ) || ( ldc < n ) ) )
	{
		return; // Error.
	}
	// Column major input expected with leading dimensions >= column stride.
	else if ( ( is_column_major == TRUE ) &&
			  ( ( lda < m ) || ( ldb < k ) || ( ldc < m ) ) )
	{
		return; // Error.
	}

	// Check if dimensions are valid.
	if ( ( m <= 0) || ( n <= 0 ) || ( k <= 0 ) ||
	     ( lda <= 0 ) || ( ldb <= 0 ) || ( ldc <= 0 ) )
	{
		return; // Error.
	}

	const inc_t rs_a = lda;
	const inc_t cs_a = 1;
	const inc_t rs_b = ldb;
	const inc_t cs_b = 1;
	const inc_t rs_c = ldc;
	const inc_t cs_c = 1;

	AOCL_MEMORY_TAG mtag_a;
	AOCL_MEMORY_TAG mtag_b;

	bli_param_map_char_to_lpmtag( mem_format_a, &mtag_a );
	bli_param_map_char_to_lpmtag( mem_format_b, &mtag_b );

	// B matrix needs to be packed in a certain format in order to be loaded
	// and used in bf16 instrution. As such the mtag_b always needs to be either
	// packed or reordered. B matrix as it is (unpacked) cannot be used, and
	// the mtag_b is set to packed to enable runtime packing.
	if ( ( is_row_major == TRUE ) && ( mtag_b == UNPACKED ) )
	{
		mtag_b = PACK;
	}
	// Inputs swapped in column major, A becomes B from kernel point of view.
	else if ( ( is_column_major == TRUE ) && ( mtag_a == UNPACKED ) )
	{
		mtag_a = PACK;
	}

	// Only unpacked A supported now.
	if ( ( is_row_major == TRUE ) && ( mtag_a != UNPACKED ) )
	{
		return; // Error.
	}
	// Inputs swapped in column major, B becomes A from kernel point of view.
	else if ( ( is_column_major == TRUE ) && ( mtag_b != UNPACKED ) )
	{
		return; // Error.
	}

	// Convert post op struct to post op linked list format.
	lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
	lpgemm_translate_to_post_ops_list
	(
	  post_op_unparsed, post_op_list,
	  ( void* )c, ( void* )( &order_use )
	);

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_membrk_rntm_set_membrk( &rntm_g );

#ifdef BLIS_ENABLE_OPENMP
	// Swapping inputs to induce row major computation for column major inputs.
	if ( is_column_major == TRUE )
	{
		lpgemm_bf16bf16f32of32_openmp_thread_decorator
		(
		  n, m, k,
		  b, rs_b, cs_b, mtag_b,
		  a, rs_a, cs_a, mtag_a,
		  ( float* )c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g,
		  post_op_list, TRUE
		);
	}
	else
	{
		lpgemm_bf16bf16f32of32_openmp_thread_decorator
		(
		  m, n, k,
		  a, rs_a, cs_a, mtag_a,
		  b, rs_b, cs_b, mtag_b,
		  ( float* )c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g,
		  post_op_list, TRUE
		);
	}
#else
	// Swapping inputs to induce row major computation for column major inputs.
	if ( is_column_major == TRUE )
	{
		lpgemm_bf16bf16f32of32_thread_decorator
		(
		  n, m, k,
		  b, rs_b, cs_b, mtag_b,
		  a, rs_a, cs_a, mtag_a,
		  ( float* )c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g,
		  post_op_list, TRUE
		);
	}
	else
	{
		lpgemm_bf16bf16f32of32_thread_decorator
		(
		  m, n, k,
		  a, rs_a, cs_a, mtag_a,
		  b, rs_b, cs_b, mtag_b,
		  ( float* )c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g,
		  post_op_list, TRUE
		);
	}
#endif
}
