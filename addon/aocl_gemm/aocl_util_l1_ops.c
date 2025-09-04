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
#include "aocl_util_interface_apis.h"
#include "lpgemm_types.h"
#include "lpgemm_config.h"
#include "lpgemm_utils_kernels.h"

AOCL_UTIL_L1_OP(float,gelu_tanh_f32)
{
	// Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
					"utility l1 operations not supported.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	if ( ( n <= 0 ) || ( x == NULL ) || ( incx <= 0 ) )
	{
		return; // Error.
	}

	lpgemm_util_cntx_t* lutil_cntx_g = lpgemm_util_get_global_cntx_obj( F32_GELU_TANH );
	( ( lpgemm_util_l1_op_f32_kernel_t )lutil_cntx_g->kern_fun_ptr )( n, x, incx );
}

AOCL_UTIL_L1_OP(float,gelu_erf_f32)
{
	// Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
					"utility l1 operations not supported.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	if ( ( n <= 0 ) || ( x == NULL ) || ( incx <= 0 ) )
	{
		return; // Error.
	}

	lpgemm_util_cntx_t* lutil_cntx_g = lpgemm_util_get_global_cntx_obj( F32_GELU_ERF );
	( ( lpgemm_util_l1_op_f32_kernel_t )lutil_cntx_g->kern_fun_ptr )( n, x, incx );
}

AOCL_UTIL_L1_OP(float,softmax_f32)
{
	// Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
					"utility l1 operations not supported.", __FILE__, __LINE__ );
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	if ( ( n <= 0 ) || ( x == NULL ) || ( incx <= 0 ) )
	{
		return; // Error.
	}

	lpgemm_util_cntx_t* lutil_cntx_g = lpgemm_util_get_global_cntx_obj( F32_SOFTMAX );
	( ( lpgemm_util_l1_op_f32_kernel_t )lutil_cntx_g->kern_fun_ptr )( n, x, incx );
}
