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
#include "aocl_gemm_interface_apis.h"
#include "lpgemm_types.h"
#include "lpgemm_config.h"
#include "lpgemm_utils_s8.h"
#include "lpgemm_reorder_s8s16.h"

AOCL_GEMM_GET_REORDER_BUF_SIZE(s8s8s16os16)
{
	if ((k <= 0) || (n <= 0))
	{
		return 0; // Error.
	}

	// Check if AVX2 ISA is supported, lpgemm s8s8s16os16 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform s8s8s16 gemm.", __FILE__, __LINE__ );
		return 0; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

	if (input_mat_type == A_MATRIX)
	{
		return 0; // A reorder not supported.
	}

	// Extra space since packing does width in multiples of 16. The vpmaddubsw
	// instruction can be used as long as atleast one ymm register can be fully
	// loaded; and since k_dim needs to be at least 2, having n_dim atleast 16
	// should give 2x16=32 elements, enough for 1 ymm register.The padding is
	// not rounded to NR (=16), since that would result in memory wastage.
	dim_t n_reorder = make_multiple_of_n(n, 16);

	// Extra space since packing does length in multiples of 2.
	dim_t k_reorder = make_multiple_of_n(k, 2);

	// Extra memory of n_reorder * sizeof( int16_t ) to store sum of every column of B matrix buffer
    siz_t size_req = sizeof(int8_t) * k_reorder * n_reorder + ( n_reorder * sizeof( int16_t ));

	return size_req;
}

AOCL_GEMM_REORDER(int8_t,s8s8s16os16)
{
	if ((input_buf_addr == NULL) || (reorder_buf_addr == NULL) ||
		(k <= 0) || (n <= 0) || (ldb < n))
	{
		return; // Error.
	}

	// Check if AVX2 ISA is supported, lpgemm s8s8s16os16 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform s8s8s16 gemm.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

	if (input_mat_type == A_MATRIX)
	{
		return; // A reorder not supported.
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global(&rntm_g);
	bli_pba_rntm_set_pba(&rntm_g);

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( S8S8S16OS16 );

	// Create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = reorder_buf_addr;

	// Create dummy original b obj;
	lpgemm_obj_t b;
	b.storage.aligned_buffer = (void *)input_buf_addr;
	b.rs = ldb;
	b.width = n;
	b.length = k;

	aocl_reorderb_nr32_s8s8s16o16( &b, &b_reorder, &rntm_g, lcntx_g );
}
