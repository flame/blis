/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "aocl_eltwise_ops_interface_apis.h"
#include "aocl_gemm_check.h"
#include "lpgemm_types.h"
#include "lpgemm_thread_decor_openmp.h"
#include "lpgemm_utils.h"
#include "lpgemm_config.h"
#include "lpgemm_post_ops.h"

BLIS_INLINE void aocl_eltwise_ops_bf16of32_base
     (
       const char        order,
       const char        transa,
       const char        transb,
       const dim_t       m,
       const dim_t       n,
       const bfloat16*   a,
       const dim_t       lda,
       float*            b,
       const dim_t       ldb,
       aocl_post_op*     post_op_unparsed,
       AOCL_STORAGE_TYPE c_downscale
     )
{
	trans_t blis_transa;
	trans_t blis_transb;

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512bf16_supported() == FALSE )
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
				"cannot perform bf16bf16f32 gemm.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans(transa, &blis_transa);
	bli_param_map_netlib_to_blis_trans(transb, &blis_transb);

	bool is_column_major = ((order == 'c') || (order == 'C'));

	// Column major support disabled for int API's till micro-kernel
	// post-ops are updated to account for column major.
	if ( ( is_column_major == TRUE ) ||
		 ( bli_is_trans( blis_transa ) ) ||
		 ( bli_is_trans( blis_transb ) ) )
	{
		bli_print_msg("Column major and transpose not supported.",
					  __FILE__, __LINE__);
		return;
	}

	// The strides are set assuming a row major kernel.
	inc_t rs_a = lda;
	inc_t cs_a = 1;
	inc_t rs_b = ldb;
	inc_t cs_b = 1;

	// Convert post op struct to post op linked list format.
	lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
	err_t err = lpgemm_translate_to_post_ops_list
	(
	  post_op_unparsed, post_op_list,
	  NULL, ( void* )( &order ),
	  m, n
	);
	if( err != BLIS_SUCCESS ) return;

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_eltwise_ops_cntx_t* lcntx_g =
		lpgemm_eltwise_ops_get_global_cntx_obj( BF16OF32 );

#ifdef BLIS_ENABLE_OPENMP

	lpgemm_eltwise_ops_bf16of32_openmp_thread_decorator
	(
	  m, n,
	  a, rs_a, cs_a,
	  b, rs_b, cs_b,
	  &rntm_g, lcntx_g,
	  post_op_list, c_downscale
	);
#else
	lpgemm_eltwise_ops_bf16of32_thread_decorator
	(
	  m, n,
	  a, rs_a, cs_a,
	  b, rs_b, cs_b,
	  &rntm_g, lcntx_g,
	  post_op_list, c_downscale
	);
#endif
}

AOCL_UTIL_ELTWISE_OPS(bfloat16,float,bf16of32)
{
	AOCL_UTIL_ELTWISE_OPS_CHECK
	(
	  "bf16of32",
	  order, transa, transb,
	  m, n,
	  a, lda,
	  b, ldb
	);

	aocl_eltwise_ops_bf16of32_base
	(
	  order, transa, transb,
	  m, n,
	  a, lda,
	  b, ldb,
	  post_op_unparsed, F32
	);
}

AOCL_UTIL_ELTWISE_OPS(bfloat16,bfloat16,bf16obf16)
{
	AOCL_UTIL_ELTWISE_OPS_CHECK
	(
	  "bf16obf16",
	  order, transa, transb,
	  m, n,
	  a, lda,
	  b, ldb
	);

	// Even though b matrix is typecasted to float*, actual load/store
	// and matrix traversal will happen as bfloat16* type. This typecast
	// is only to ensure code is reused.
	aocl_eltwise_ops_bf16of32_base
	(
	  order, transa, transb,
	  m, n,
	  a, lda,
	  ( float* )b, ldb,
	  post_op_unparsed, BF16
	);
}

AOCL_UTIL_ELTWISE_OPS(float,float,f32of32)
{
	AOCL_UTIL_ELTWISE_OPS_CHECK
	(
	  "f32of32",
	  order, transa, transb,
	  m, n,
	  a, lda,
	  b, ldb
	);

	trans_t blis_transa;
	trans_t blis_transb;

	// Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512bf16_supported() == FALSE )
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
				"cannot perform bf16bf16f32 gemm.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans(transa, &blis_transa);
	bli_param_map_netlib_to_blis_trans(transb, &blis_transb);

	bool is_column_major = ((order == 'c') || (order == 'C'));

	// Column major support disabled for int API's till micro-kernel
	// post-ops are updated to account for column major.
	if ( ( is_column_major == TRUE ) ||
		 ( bli_is_trans( blis_transa ) ) ||
		 ( bli_is_trans( blis_transb ) ) )
	{
		bli_print_msg("Column major and transpose not supported.",
					  __FILE__, __LINE__);
		return;
	}

	// The strides are set assuming a row major kernel.
	inc_t rs_a = lda;
	inc_t cs_a = 1;
	inc_t rs_b = ldb;
	inc_t cs_b = 1;

	// Convert post op struct to post op linked list format.
	lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
	err_t err = lpgemm_translate_to_post_ops_list
	(
	  post_op_unparsed, post_op_list,
	  NULL, ( void* )( &order ),
	  m, n
	);
	if( err != BLIS_SUCCESS ) return;

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_eltwise_ops_cntx_t* lcntx_g =
		lpgemm_eltwise_ops_get_global_cntx_obj( F32OF32 );

#ifdef BLIS_ENABLE_OPENMP

	lpgemm_eltwise_ops_f32of32_openmp_thread_decorator
	(
	  m, n,
	  a, rs_a, cs_a,
	  b, rs_b, cs_b,
	  &rntm_g, lcntx_g,
	  post_op_list, F32
	);
#else
	lpgemm_eltwise_ops_f32of32_thread_decorator
	(
	  m, n,
	  a, rs_a, cs_a,
	  b, rs_b, cs_b,
	  &rntm_g, lcntx_g,
	  post_op_list, F32
	);
#endif
}