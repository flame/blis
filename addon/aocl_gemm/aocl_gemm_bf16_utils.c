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
#include "lpgemm_types.h"
#include "lpgemm_config.h"
#include "lpgemm_utils.h"
#include "lpgemm_reorder_bf16.h"

AOCL_GEMM_REORDER(bfloat16, bf16bf16f32of32_reference)
{

	trans_t blis_trans;

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans( trans, &blis_trans );

	if ( ( input_buf_addr == NULL ) || ( reorder_buf_addr == NULL ) ||
	     ( k <= 0 ) || ( n <= 0 ) )
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;
	if( ( order == 'r') || ( order == 'R' ) )
	{
		if( ( bli_is_notrans( blis_trans ) && ( ldb < n ) ) ||
	        ( bli_is_trans( blis_trans ) && ( ldb < k ) ) )
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans( blis_trans ) ? ldb : 1;
			cs_b = bli_is_notrans( blis_trans ) ? 1 : ldb;
		}
	}
	else if ( ( order == 'c' ) || ( order == 'C' ) )
	{
		if( ( bli_is_notrans( blis_trans ) && ( ldb < k ) ) ||
	        ( bli_is_trans( blis_trans ) && ( ldb < n ) ) )
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans( blis_trans ) ? 1 : ldb;
			cs_b = bli_is_notrans( blis_trans ) ? ldb : 1;
		}
	}
	else
	{
		return; // Error
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return; // A reorder not supported.
	}

#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	if( ( lpgemm_get_enabled_arch() != BLIS_ARCH_ZEN3 ) && ( n == 1 ) )
	{
		if( rs_b == 1 )
		{
			memcpy( reorder_buf_addr, input_buf_addr, ( k * sizeof( bfloat16 ) ) );
		}
		else
		{
			for( dim_t k0 = 0; k0 < k; k0++ )
			{
				reorder_buf_addr[k0] = input_buf_addr[k0*rs_b];
			}
		}
		return;
	}
#endif
	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16BF16F32OF32 );

	// Create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = reorder_buf_addr;

	// Create dummy original b obj;
	lpgemm_obj_t b;
	b.storage.aligned_buffer = ( void* )input_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;

	reorderb_nr64_bf16bf16f32of32_reference( &b, &b_reorder, &rntm_g, lcntx_g );

}

AOCL_GEMM_UNREORDER(bfloat16, bf16bf16f32of32_reference)
{
	if ( ( output_buf_addr == NULL ) || ( reorder_buf_addr == NULL ) ||
	     ( k <= 0 ) || ( n <= 0 ) )
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;

	// Check for the validity of strides.
	if( ( order == 'r' ) || ( order == 'R' ) )
	{
		if( ldb < n ) return; // Error
		else
		{
			rs_b = ldb;
			cs_b = 1;
		}
	}
	else if( ( order == 'c' ) || ( order == 'C' ) )
	{
		if( ldb < k ) return; // Error.
		else
		{
			rs_b = 1;
			cs_b = ldb;
		}
	}
	else
	{
		return; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return; // A reorder not supported.
	}

#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	if( n == 1 )
	{
		if( rs_b == 1 )
		{
			memcpy( output_buf_addr, reorder_buf_addr, ( k * sizeof( bfloat16 ) ) );
		}
		else
		{
			for( dim_t k0 = 0; k0 < k; k0++ )
			{
				output_buf_addr[k0*rs_b] = reorder_buf_addr[k0];
			}
		}
		return;
	}
#endif

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16BF16F32OF32 );

	// create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = ( void* )reorder_buf_addr;

	// create dummy b obj.
	lpgemm_obj_t b;
	b.storage.aligned_buffer = ( void* )output_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;

	unreorderb_nr64_bf16bf16f32of32_reference( &b, &b_reorder, &rntm_g, lcntx_g );
}

AOCL_GEMM_GET_REORDER_BUF_SIZE(bf16bf16f32of32)
{
	if ( ( k <= 0 ) || ( n <= 0 ) )
	{
		return 0; // Error.
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return 0; // A reorder not supported.
	}

	// Extra space since packing does width in multiples of 16. The bf16
	// instruction can be used as long as at least one zmm register can be fully
	// loaded; and since k_dim needs to be at least 2, having n_dim at least 16
	// should give 2x16=32 elements, enough for 1 zmm register.The padding is
	// not rounded to NR (=64), since that would result in memory wastage.
#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	dim_t n_reorder;
	/*It is expected that while bf16 input is passed to AVX2 kernels,
	  the unreorder/conversion of bf16->f32 is done, which expects the
	  reordered matrix to be padded with n multiple of 16, k multiple of 2. */
	if( ( lpgemm_get_enabled_arch() != BLIS_ARCH_ZEN3 ) &&  ( n == 1 ) )
	{
		n_reorder = 1;
	}
	else
	{
		n_reorder = make_multiple_of_n( n, 16 );
	}

	// Extra space since packing does length in multiples of 2.
	dim_t k_reorder;
	if( ( lpgemm_get_enabled_arch() != BLIS_ARCH_ZEN3 ) && ( n == 1 ) )
	{
		k_reorder = k;
	}
	else
	{
		k_reorder = make_multiple_of_n( k, 2 );
	}
#else
	dim_t n_reorder = make_multiple_of_n( n, 16 );;
	dim_t k_reorder = make_multiple_of_n( k, 2 );
#endif
	siz_t size_req = sizeof( int16_t ) * k_reorder * n_reorder;

	return size_req;
}

AOCL_GEMM_REORDER(bfloat16, bf16bf16f32of32)
{
	trans_t blis_trans;

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans( trans, &blis_trans );

	if ( ( input_buf_addr == NULL ) || ( reorder_buf_addr == NULL ) ||
	     ( k <= 0 ) || ( n <= 0 ) )
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;
	if( ( order == 'r') || ( order == 'R' ) )
	{
		if( ( bli_is_notrans( blis_trans ) && ( ldb < n ) ) ||
	        ( bli_is_trans( blis_trans ) && ( ldb < k ) ) )
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans( blis_trans ) ? ldb : 1;
			cs_b = bli_is_notrans( blis_trans ) ? 1 : ldb;
		}
	}
	else if ( ( order == 'c' ) || ( order == 'C' ) )
	{
		if( ( bli_is_notrans( blis_trans ) && ( ldb < k ) ) ||
	        ( bli_is_trans( blis_trans ) && ( ldb < n ) ) )
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans( blis_trans ) ? 1 : ldb;
			cs_b = bli_is_notrans( blis_trans ) ? ldb : 1;
		}
	}
	else
	{
		return; // Error
	}

	// Check if avx-2 or avx512_bf16 ISA is supported, lpgemm matmul only works with it.
	if ( bli_cpuid_is_avx512bf16_supported() == FALSE )
	{
		if ( bli_cpuid_is_avx2fma3_supported() == FALSE )
		{
			bli_print_msg(" AVX512_BF16 ISA  and AVX2 ISA not supported by processor, "
				"cannot perform bf16bf16f32/f32f32f32 gemm.", __FILE__, __LINE__ );
			return; // Error.
		}

		aocl_reorder_bf16bf16f32of32_reference( order,trans ,mat_type, input_buf_addr,
								reorder_buf_addr, k, n, ldb );

		return;
	}

	/* Initialize BLIS. */
	bli_init_auto();

	// Set MC, NC, KC, NR, MR.
	aocl_lpgemm_init_global_cntx();

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return; // A reorder not supported.
	}

#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	/*When AOCL_ENABLE_INSTRUCTIONS=AVX2, f32 kernels would be executed for
	  bf16 input, for which re-ordered bf16 input is converted and unreordered
	  to hold f32 values. The un-reorder/convert API considers the padded bf16 input.*/
	if( lpgemm_get_enabled_arch() == BLIS_ARCH_ZEN3 )
	{
		aocl_reorder_bf16bf16f32of32_reference( order,trans ,mat_type, input_buf_addr,
			reorder_buf_addr, k, n, ldb );
		return;
	}
	if( n == 1 )
	{
		if( rs_b == 1 )
		{
			memcpy( reorder_buf_addr, input_buf_addr, ( k * sizeof( bfloat16 ) ) );
		}
		else
		{
			for( dim_t k0 = 0; k0 < k; k0++ )
			{
				reorder_buf_addr[k0] = input_buf_addr[k0*rs_b];
			}
		}
		return;
	}
#endif
	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16BF16F32OF32 );

	// Create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = reorder_buf_addr;

	// Create dummy original b obj;
	lpgemm_obj_t b;
	b.storage.aligned_buffer = ( void* )input_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;

	reorderb_nr64_bf16bf16f32of32( &b, &b_reorder, &rntm_g, lcntx_g );
}

AOCL_GEMM_REORDER_MXP(float, bfloat16, f32obf16)
{

#ifdef LPGEMM_BF16_JIT
	bli_print_msg(" f32obf16 is not supported by JIT kernels.", __FILE__, __LINE__ );
	return;
#endif
	trans_t blis_trans;

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans(trans, &blis_trans);

	if ((input_buf_addr == NULL) || (reorder_buf_addr == NULL) ||
		(k <= 0) || (n <= 0))
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;
	if ((order == 'r') || (order == 'R'))
	{
		if ((bli_is_notrans(blis_trans) && (ldb < n)) ||
			(bli_is_trans(blis_trans) && (ldb < k)))
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans(blis_trans) ? ldb : 1;
			cs_b = bli_is_notrans(blis_trans) ? 1 : ldb;
		}
	}
	else if ((order == 'c') || (order == 'C'))
	{
		if ((bli_is_notrans(blis_trans) && (ldb < k)) ||
			(bli_is_trans(blis_trans) && (ldb < n)))
		{
			return; // Error.
		}
		else
		{
			rs_b = bli_is_notrans(blis_trans) ? 1 : ldb;
			cs_b = bli_is_notrans(blis_trans) ? ldb : 1;
		}
	}
	else
	{
		return; // Error
	}

	// Check if avx512_bf16 ISA is supported, lpgemm matmul only works with it.
	if (bli_cpuid_is_avx512bf16_supported() == FALSE)
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
					  "cannot perform bf16bf16f32 gemm.",
					  __FILE__, __LINE__);
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

#if (defined(BLIS_KERNELS_ZEN4))
	if (n == 1)
	{
		if (rs_b == 1)
		{
			for (dim_t k0 = 0; k0 < k; k0++)
			{
				memcpy(&reorder_buf_addr[k0], (char *)(&input_buf_addr[k0]) + 2, sizeof(bfloat16));
			}
		}
		else
		{
			for (dim_t k0 = 0; k0 < k; k0++)
			{
				memcpy(&reorder_buf_addr[k0], (char *)(&input_buf_addr[k0 * rs_b]) + 2, sizeof(bfloat16));
			}
		}
		return;
	}
#endif

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global(&rntm_g);
	bli_pba_rntm_set_pba(&rntm_g);

	lpgemm_cntx_t *lcntx_g = lpgemm_get_global_cntx_obj(F32OBF16);

	// Create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = reorder_buf_addr;

	// Create dummy original b obj;
	lpgemm_obj_t b;
	b.storage.aligned_buffer = (void *)input_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;

	reorderb_mxp_nr64_f32obf16(&b, &b_reorder, &rntm_g, lcntx_g);
}

AOCL_GEMM_UNREORDER(bfloat16, bf16bf16f32of32)
{
	if ( ( output_buf_addr == NULL ) || ( reorder_buf_addr == NULL ) ||
	     ( k <= 0 ) || ( n <= 0 ) )
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;

	// Check for the validity of strides.
	if( ( order == 'r' ) || ( order == 'R' ) )
	{
		if( ldb < n ) return; // Error
		else
		{
			rs_b = ldb;
			cs_b = 1;
		}
	}
	else if( ( order == 'c' ) || ( order == 'C' ) )
	{
		if( ldb < k ) return; // Error.
		else
		{
			rs_b = 1;
			cs_b = ldb;
		}
	}
	else
	{
		return; // Error.
	}

	// Check if avx512_bf16 ISA is supported, lpgemm matmul only works with it.
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

	AOCL_MATRIX_TYPE input_mat_type;
	bli_param_map_char_to_lpmat_type( mat_type, &input_mat_type );

	if ( input_mat_type == A_MATRIX )
	{
		return; // A reorder not supported.
	}
#if (defined(BLIS_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
	if( n == 1 )
	{
		if( rs_b == 1 )
		{
			memcpy( output_buf_addr, reorder_buf_addr, ( k * sizeof( bfloat16 ) ) );
		}
		else
		{
			for( dim_t k0 = 0; k0 < k; k0++ )
			{
				output_buf_addr[k0*rs_b] = reorder_buf_addr[k0];
			}
		}
		return;
	}
#endif
	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_g;
	bli_rntm_init_from_global( &rntm_g );
	bli_pba_rntm_set_pba( &rntm_g );

	lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj( BF16BF16F32OF32 );

	// create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = ( void* )reorder_buf_addr;

	// create dummy b obj.
	lpgemm_obj_t b;
	b.storage.aligned_buffer = ( void* )output_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;

	unreorderb_nr64_bf16bf16f32of32( &b, &b_reorder, &rntm_g, lcntx_g );
}

AOCL_GEMM_GET_REORDER_BUF_SIZE(bf16s4f32of32)
{
	if ((k <= 0) || (n <= 0))
	{
		return 0; // Error.
	}

	// Check if avx512_bf16 ISA is supported, lpgemm matmul only works with it.
	if (bli_cpuid_is_avx512bf16_supported() == FALSE)
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
					  "cannot perform bf16bf16f32 gemm.",
					  __FILE__, __LINE__);
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

	dim_t n_reorder;

	/*if (n == 1)
	{
		n_reorder = 1;
	}
	else*/
	{
		n_reorder = make_multiple_of_n(n, 16);
	}

	// Extra space since packing does length in multiples of 2.
	dim_t k_reorder;
	/*if (n == 1)
	{
		k_reorder = k;
	}
	else*/
	{
		k_reorder = make_multiple_of_n(k, 2);
	}

	siz_t size_req = (sizeof(int8_t) * k_reorder * n_reorder)/2;
	return size_req;
}

AOCL_GEMM_REORDER(int8_t, bf16s4f32of32)
{
	trans_t blis_trans;

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans(trans, &blis_trans);

	if ((input_buf_addr == NULL) || (reorder_buf_addr == NULL) ||
		(k <= 0) || (n <= 0) || (bli_is_notrans(blis_trans) && (ldb < n)) ||
		(bli_is_trans(blis_trans) && (ldb < k)))
	{
		return; // Error.
	}

	inc_t rs_b, cs_b;
	if ((order == 'r') || (order == 'R'))
	{
		rs_b = bli_is_notrans(blis_trans) ? ldb : 1;
		cs_b = bli_is_notrans(blis_trans) ? 1 : ldb;
	}
	else if ((order == 'c') || (order == 'C'))
	{
		rs_b = bli_is_notrans(blis_trans) ? 1 : ldb;
		cs_b = bli_is_notrans(blis_trans) ? ldb : 1;
	}
	else
	{
		return; // Error
	}

	// Check if avx512_bf16 ISA is supported, lpgemm matmul only works with it.
	if (bli_cpuid_is_avx512bf16_supported() == FALSE)
	{
		bli_print_msg(" AVX512_BF16 ISA not supported by processor, "
					  "cannot perform bf16bf16f32 gemm.",
					  __FILE__, __LINE__);
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

	lpgemm_cntx_t *lcntx_g = lpgemm_get_global_cntx_obj(BF16S4F32OF32);

	// Create dummy b_reorder obj.
	lpgemm_obj_t b_reorder;
	b_reorder.storage.aligned_buffer = reorder_buf_addr;

	// Create dummy original b obj;
	lpgemm_obj_t b;
	b.storage.aligned_buffer = (void *)input_buf_addr;
	b.rs = rs_b;
	b.cs = cs_b;
	b.width = n;
	b.length = k;
	b.mat_type = input_mat_type;

	reorderb_nr64_bf16s4f32of32(&b, &b_reorder, &rntm_g, lcntx_g);
}

