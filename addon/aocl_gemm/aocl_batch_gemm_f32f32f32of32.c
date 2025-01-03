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

AOCL_BGEMM_MATMUL(float,float,float,float,f32f32f32of32)
{
	LPGEMM_START_LOGGER();
	BATCH_LPGEMM_WRITE_LOGGER \
	(
	  "f32f32f32of32", \
	  order, transa, transb, \
	  batch_size, m, n, k, \
	  ( ( float* ) alpha ), \
	  lda, mem_format_a, \
	  ldb, mem_format_b, \
	  ( ( float* ) beta ), \
	  ldc, post_op_unparsed \
	);

	trans_t blis_transa;
	trans_t blis_transb;

	inc_t rs_a[batch_size];
	inc_t cs_a[batch_size];

	inc_t rs_b[batch_size];
	inc_t cs_b[batch_size];

	inc_t rs_c[batch_size];
	inc_t cs_c[batch_size];

	AOCL_MEMORY_TAG mtag_a[batch_size];
	AOCL_MEMORY_TAG mtag_b[batch_size];

	float *a_local[batch_size], *b_local[batch_size];
	dim_t m_local[batch_size], n_local[batch_size];

	// Convert post op struct to post op linked list format.
	lpgemm_post_op post_op_list[batch_size][AOCL_MAX_POST_OPS];

	// Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
	if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
	{
		bli_print_msg(" AVX2 ISA not supported by processor, "
				"cannot perform f32f32f32 gemm.", __FILE__, __LINE__ );
		goto err_hndl;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	// check for validity of params.
	int err_no = 0;

	for( dim_t bs_i = 0; bs_i < batch_size; bs_i++ )
	{
		// check for validity of params.
		AOCL_BATCH_GEMM_CHECK
		(
		  "batch_f32f32f32of32",
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
			rs_a[bs_i] = ldb[bs_i];
			cs_a[bs_i] = 1;

			if( bli_is_trans( blis_transb ) )
			{
				rs_a[bs_i] = 1;
				cs_a[bs_i] = ldb[bs_i];
			}

			rs_b[bs_i] = lda[bs_i];
			cs_b[bs_i] = 1;

			if( bli_is_trans( blis_transa ) )
			{
				rs_b[bs_i] = 1;
				cs_b[bs_i] = lda[bs_i];
			}

			bli_param_map_char_to_lpmtag( mem_format_a[bs_i], &(mtag_b[bs_i]) );
			bli_param_map_char_to_lpmtag( mem_format_b[bs_i], &(mtag_a[bs_i]) );

			// Inputs swapped in column major, A becomes B from kernel point of view.
			// Reorder is not supported for column major matrices.
			if ( ( ( mtag_b[bs_i] == REORDERED ) || ( mtag_a[bs_i] == REORDERED ) ) )
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
				mtag_a[bs_i] = PACK;
			}

			// swap m & n in case of col-major matrices
			m_local[bs_i] = n[bs_i];
			n_local[bs_i] = m[bs_i];

			// swap a & b pointers in case of col-major matrices
			a_local[bs_i] = (float*)(b[bs_i]);
			b_local[bs_i] = (float*)(a[bs_i]);
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

			// copy the values of m & n
			m_local[bs_i] = m[bs_i];
			n_local[bs_i] = n[bs_i];

			// copy the values of a & b pointers
			a_local[bs_i] = (float*)(a[bs_i]);
			b_local[bs_i] = (float*)(b[bs_i]);
		}

		rs_c[bs_i] = ldc[bs_i];
		cs_c[bs_i] = 1;

		// By default enable packing for B matrix. Before the 5 loop, based on
		// the input dimensions, the smart threading logic will adjust it
		// (disable/enable) accordingly.
		if ( mtag_b[bs_i] == UNPACKED )
		{
			mtag_b[bs_i] = PACK;
		}

		err_t err = lpgemm_translate_to_post_ops_list
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

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( F32F32F32OF32 );

#ifdef BLIS_ENABLE_OPENMP
	batch_lpgemm_f32f32f32of32_openmp_thread_decorator
	(
	  batch_size, m_local, n_local, k,
	  (const float**)a_local, rs_a, cs_a, mtag_a,
	  (const float**)b_local, rs_b, cs_b, mtag_b,
	  c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  post_op_list, F32
	);


#else
	// Setting pack A and B by default for non open mp case.
	bli_rntm_set_pack_a( 1, &rntm_g );
	bli_rntm_set_pack_b( 1, &rntm_g );

	batch_lpgemm_f32f32f32of32_thread_decorator
	(
	  batch_size, m_local, n_local, k,
	  (const float**)a_local, rs_a, cs_a, mtag_a,
	  (const float**)b_local, rs_b, cs_b, mtag_b,
	  c, rs_c, cs_c,
	  alpha, beta,
	  &rntm_g, lcntx_g,
	  post_op_list, F32
	);
#endif

err_hndl:;
	LPGEMM_STOP_LOGGER();
}
