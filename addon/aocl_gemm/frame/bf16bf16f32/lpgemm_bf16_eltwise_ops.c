/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_eltwise_ops_interface_apis.h"
#include "lpgemm_eltwise_ops_kernels.h"
#include "lpgemm_utils.h"
#include "lpgemm_thrinfo_utils.h"
#include "lpgemm_config.h"

// Kernel function prototypes.
typedef void (*lpgemm_util_post_ops_kernel_f32)
     (
       const dim_t,
       const dim_t,
       const bfloat16*,
       const dim_t,
       const dim_t,
       float*,
       const dim_t,
       const dim_t,
       lpgemm_post_op*,
       lpgemm_post_op_attr
     );

LPGEMM_ELTWISE_OPS_IFACE(bfloat16,float,bf16of32)
{
	dim_t NR = lcntx->blksz.NR;
	dim_t MR = lcntx->blksz.MR;

	lpgemm_post_op_attr post_ops_attr;
	post_ops_attr.c_stor_type = c_downscale;
	post_ops_attr.buf_downscale = NULL;

	// Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
	thrinfo_t thread_jc;
	thrinfo_t thread_ic;

	lpgemm_gen_thrinfo( thread, &thread_jc, &thread_ic );

	// Compute the JC, IC loop thread range for the current thread.
	dim_t jc_start, jc_end;
	bli_thread_range_sub( &thread_jc, n, NR, FALSE, &jc_start, &jc_end );

	dim_t ic_start, ic_end;
	bli_thread_range_sub( &thread_ic, m, MR, FALSE, &ic_start, &ic_end );

	post_ops_attr.post_op_c_i = ic_start;
	post_ops_attr.post_op_c_j = jc_start;
	post_ops_attr.rs_c_downscale = rs_b;
	post_ops_attr.cs_c_downscale = cs_b;
	post_ops_attr.is_first_k = FALSE;
	post_ops_attr.is_last_k = TRUE; // Should always be TRUE here.

	// Advance the matrix to the right positions based on thread id.
	// To note that float and bfloat16 are both handled using this same
	// frame, so the strides needs to be updated on the actual b matrix
	// datatype or the c_downscale value.
	dim_t dsize = sizeof( float );
	if ( post_ops_attr.c_stor_type == BF16 )
	{
		dsize = sizeof( bfloat16 );
	}

	int8_t* b_i = ( int8_t* )b;

	( ( lpgemm_util_post_ops_kernel_f32 )( lcntx->eltwise_ops_kern_fun_ptr ) )
	(
	  ( ic_end - ic_start ), ( jc_end - jc_start ),
	  a + ( rs_a * ic_start ) + ( cs_a * jc_start ),
	  rs_a, cs_a,
	  ( float* )( b_i + ( dsize * ( ( rs_b * ic_start ) +
				( cs_b * jc_start ) ) ) ), rs_b, cs_b,
	  post_op_list, post_ops_attr
	);
}
