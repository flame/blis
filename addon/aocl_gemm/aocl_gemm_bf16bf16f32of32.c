/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

static inline bool is_tiny_input_bf16of32
     (
       dim_t m,
       dim_t n,
       dim_t k,
       lpgemm_cntx_t* lcntx
     )
{
	bool is_tiny = FALSE;

    const dim_t NC = lcntx->blksz.NC;
    const dim_t MC = lcntx->blksz.MC;
    const dim_t KC = lcntx->blksz.KC;
    const dim_t MR = lcntx->blksz.MR;
    const dim_t NR = lcntx->blksz.NR;

	dim_t mnk = m * n * k;
	const dim_t mnk_magic_num = 36 * 128 * 128;
	const dim_t m_thresh = 6 * MR;
	const dim_t n_thresh = 2 * NR;
	const dim_t k_thresh = 1024;

	// Need to explicitly check for MC, NC boundaries for safety.
	if ( ( m <= MC ) && ( n < NC ) && ( k < KC ) &&
		 ( ( m <= m_thresh ) && ( n <= n_thresh ) && ( k <= k_thresh ) &&
		   ( mnk < mnk_magic_num ) ) )
	{
		is_tiny = TRUE;
	}

	return is_tiny;
}

AOCL_GEMM_MATMUL(bfloat16,bfloat16,float,float,bf16bf16f32of32)
{
	LPGEMM_START_LOGGER();
	LPGEMM_WRITE_LOGGER \
	(
	  "bf16bf16f32of32", \
	  order, transa, transb, \
	  m, n, k, \
	  ( ( float ) alpha ), \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  ( ( float ) beta ), \
	  ldc, post_op_unparsed \
	);

	trans_t blis_transa;
	trans_t blis_transb;

	/*BF16 API can still work with AVX2 ISA, where the input data would
	 be converted to F32 and then call the F32 kernels*/
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform bf16bf16f32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// check for validity of params.
	int err_no = 0;
	AOCL_GEMM_CHECK
	(
	  "bf16bf16f32of32",
	  order, transa, transb,
	  m, n, k,
	  a, lda, mem_format_a,
	  b, ldb, mem_format_b,
	  c, ldc,
	  err_no
	);
	if ( err_no != 0 )
	{
		goto err_hndl;
	}

#ifdef LPGEMM_BF16_JIT
	if( get_jit_kernels_generated() == FALSE )
	{
		bli_print_msg(" Could not generate bf16bf16f32of32 "
			" kernels using JIT.", __FILE__, __LINE__ );
			return;
	}
#endif


	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans( transa, &blis_transa );
	bli_param_map_netlib_to_blis_trans( transb, &blis_transb );

	bool is_row_major = ( ( order == 'r' ) || ( order == 'R' ) );
	bool is_column_major = ( ( order == 'c' ) || ( order == 'C' ) );

	// The strides are set assuming a row major kernel.
	inc_t rs_a = lda;
	inc_t cs_a = 1;

	if ( bli_is_trans( blis_transa ) )
	{
		rs_a = 1;
		cs_a = lda;
	}

	inc_t rs_b = ldb;
	inc_t cs_b = 1;

	if( bli_is_trans( blis_transb ) )
	{
		rs_b = 1;
		cs_b = ldb;
	}
	const inc_t rs_c = ldc;
	const inc_t cs_c = 1;

	AOCL_MEMORY_TAG mtag_a;
	AOCL_MEMORY_TAG mtag_b;

	bli_param_map_char_to_lpmtag( mem_format_a, &mtag_a );
	bli_param_map_char_to_lpmtag( mem_format_b, &mtag_b );

	// Reorder is not supported for A matrix
	if( ( is_row_major == TRUE ) && ( mtag_a == REORDERED ) )
	{
		bli_print_msg(" Reordering of A matrix is not supported in row major case.", __FILE__, __LINE__ );
		goto err_hndl;
	}
	// Inputs swapped in column major, A becomes B from kernel point of view.
	// Reorder is not supported for column major matrices.
	else if ( ( is_column_major == TRUE ) && ( ( mtag_b == REORDERED ) || ( mtag_a == REORDERED ) ) )
	{
		bli_print_msg(" Reordering of column major matrices is not supported.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	// From 5-loop function point of view
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

	// From 5-loop function point of view,
	// A matrix when in column major storage needs to be packed to row-major
	// storage as kernel expects A matrix to be in row-major format.
	if( ( is_row_major == TRUE ) && ( bli_is_trans(blis_transa ) ) )
	{
		mtag_a = PACK;
	}
	// Inputs swapped in column major, A becomes B from kernel point of view.
	else if ( ( is_column_major == TRUE ) && ( bli_is_trans(blis_transb ) ) )
	{
		mtag_b = PACK;
	}

	// Convert post op struct to post op linked list format.
	lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
	err_t err = lpgemm_translate_to_post_ops_list
	(
	  post_op_unparsed, post_op_list,
	  ( void* )c, ( void* )( &order ),
	  m, n
	);

	if( err != BLIS_SUCCESS )
	{
		goto err_hndl;
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16BF16F32OF32 );

#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	/* While AOCL_ENABLE_INSTRUCTIONS=AVX2 is enabled in machines that supports BF16/VNNI
	*  with only the ISA check the exeution could enter tiny path and result in seg fault
	*  as the tiny path for BF16->FP32 is not available. Hence the arch_id also has to be
	*  verified here.
	*/
	arch_t arch_id =  bli_arch_query_id();
	if( ( bli_cpuid_is_avx512bf16_supported() == TRUE ) && ( ( arch_id == BLIS_ARCH_ZEN4 ) || ( arch_id == BLIS_ARCH_ZEN5 ) ) &&
		 ( is_tiny_input_bf16of32( m, n, k, lcntx_g ) == TRUE ) &&
		 ( is_single_thread( &rntm_g ) == TRUE) &&
		 ( is_row_major == TRUE ) )
	{
		lpgemm_rowvar_tiny_bf16bf16f32of32
		(
		  m, n, k,
		  a, rs_a, cs_a, mtag_a,
		  b, rs_b, cs_b, mtag_b,
		  c, rs_c, cs_c,
		  alpha, beta,
		  lcntx_g,
		  post_op_list, F32
		);
		return;
	}
#endif

#ifdef BLIS_ENABLE_OPENMP
	// Swapping inputs to induce row major computation for column major inputs.
	if ( is_column_major == TRUE )
	{
		lpgemm_bf16bf16f32of32_openmp_thread_decorator
		(
		  n, m, k,
		  b, rs_b, cs_b, mtag_b,
		  a, rs_a, cs_a, mtag_a,
		  c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g, lcntx_g,
		  post_op_list, F32
		);
	}
	else
	{
		lpgemm_bf16bf16f32of32_openmp_thread_decorator
		(
		  m, n, k,
		  a, rs_a, cs_a, mtag_a,
		  b, rs_b, cs_b, mtag_b,
		  c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g, lcntx_g,
		  post_op_list, F32
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
		  c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g, lcntx_g,
		  post_op_list, F32
		);
	}
	else
	{
		lpgemm_bf16bf16f32of32_thread_decorator
		(
		  m, n, k,
		  a, rs_a, cs_a, mtag_a,
		  b, rs_b, cs_b, mtag_b,
		  c, rs_c, cs_c,
		  alpha, beta,
		  &rntm_g, lcntx_g,
		  post_op_list, F32
		);
	}
#endif

err_hndl:;
	LPGEMM_STOP_LOGGER();
}
