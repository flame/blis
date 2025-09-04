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

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "../u8s8s32/lpgemm_s32_kern_macros.h"
#include "../u8s8s32/lpgemm_s32_memcpy_macros.h"

// 6x64 int8o32 kernel
LPGEMM_MAIN_KERN2(int8_t,int8_t,int32_t,s8s8s32os32_6x64m_sym_quant)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_6x64_DISABLE,
						  &&POST_OPS_BIAS_6x64,
						  &&POST_OPS_RELU_6x64,
						  &&POST_OPS_RELU_SCALE_6x64,
						  &&POST_OPS_GELU_TANH_6x64,
						  &&POST_OPS_GELU_ERF_6x64,
						  &&POST_OPS_CLIP_6x64,
						  &&POST_OPS_DOWNSCALE_6x64,
						  &&POST_OPS_MATRIX_ADD_6x64,
						  &&POST_OPS_SWISH_6x64,
						  &&POST_OPS_MATRIX_MUL_6x64,
						  &&POST_OPS_TANH_6x64,
						  &&POST_OPS_SIGMOID_6x64
						};

	dim_t MR = 6;
	dim_t NR = 64;

	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t group_size = grp_post_ops_attr.group_size;

	if ( n0 < NR )
	{
		dim_t n0_rem = n0 % 16;

		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization. Any n0 < NR(64) can be expressed as n0 = 48 + n`
		// or n0 = 32 + n` or n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n0 / 48;
		dim_t n0_32 = n0 / 32;
		dim_t n0_16 = n0 / 16;

		// KC when not multiple of 4 will have padding to make it multiple of
		// 4 in packed buffer. Also the k0 cannot be passed as the updated
		// value since A matrix is not packed and requires original k0.
		dim_t k0_updated = k0;
		dim_t k_partial_pieces = k0 % 4;
		if ( k_partial_pieces > 0 )
		{
			k0_updated += ( 4 - k_partial_pieces );
		}

		if ( n0_48 == 1 )
		{
			lpgemm_rowvar_s8s8s32os32_6x48_sym_quant
			(
			  m0, k0,
			  a, rs_a, cs_a, ps_a,
			  b, ( ( rs_b / 4 ) * 3 ), cs_b,
			  c, rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);

			b = b + ( 48 * k0_updated ); // k0x48 packed contiguosly.
			c = c + 48;
			post_ops_attr.post_op_c_j += 48;
			grp_post_ops_attr.grp_post_op_j += 48;
			post_ops_attr.b_sum_offset += 48;
		}
		else if ( n0_32 == 1 )
		{
			lpgemm_rowvar_s8s8s32os32_6x32_sym_quant
			(
			  m0, k0,
			  a, rs_a, cs_a, ps_a,
			  b, ( ( rs_b / 4 ) * 2 ), cs_b,
			  c, rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);

			b = b + ( 32 * k0_updated ); // k0x32 packed contiguosly.
			c = c + 32;
			post_ops_attr.post_op_c_j += 32;
			grp_post_ops_attr.grp_post_op_j += 32;
			post_ops_attr.b_sum_offset += 32;
		}
		else if ( n0_16 == 1 )
		{
			lpgemm_rowvar_s8s8s32os32_6x16_sym_quant
			(
			  m0, k0,
			  a, rs_a, cs_a, ps_a,
			  b, ( ( rs_b / 4 ) * 1 ), cs_b,
			  c, rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);

			b = b + ( 16 * k0_updated ); // k0x16 packed contiguosly.
			c = c + 16;
			post_ops_attr.post_op_c_j += 16;
			grp_post_ops_attr.grp_post_op_j += 16;
			post_ops_attr.b_sum_offset += 16;
		}

		if ( n0_rem > 0 )
		{
			lpgemm_rowvar_s8s8s32os32_6xlt16_sym_quant
			(
			  m0, k0,
			  a, rs_a, cs_a, ps_a,
			  b, ( ( rs_b / 4 ) * 1 ), cs_b,
			  c, rs_c,
			  alpha, beta, n0_rem,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);

			// No leftover fringe after this point.
		}

		return;
	}

	// B matrix storage.
	__m512i b0 = _mm512_setzero_epi32();
	__m512i b1 = _mm512_setzero_epi32();
	__m512i b2 = _mm512_setzero_epi32();
	__m512i b3 = _mm512_setzero_epi32();

	// A matrix storage.
	__m512i a_int32_0 = _mm512_setzero_epi32();
	__m512i a_int32_1 = _mm512_setzero_epi32();

	uint8_t cvt_uint8 = 128;
	__m512i vec_uint8 = _mm512_set1_epi8 (cvt_uint8);

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		__m512 acc_00 = _mm512_setzero_ps();
		__m512 acc_01 = _mm512_setzero_ps();
		__m512 acc_02 = _mm512_setzero_ps();
		__m512 acc_03 = _mm512_setzero_ps();

		__m512 acc_10 = _mm512_setzero_ps();
		__m512 acc_11 = _mm512_setzero_ps();
		__m512 acc_12 = _mm512_setzero_ps();
		__m512 acc_13 = _mm512_setzero_ps();

		__m512 acc_20 = _mm512_setzero_ps();
		__m512 acc_21 = _mm512_setzero_ps();
		__m512 acc_22 = _mm512_setzero_ps();
		__m512 acc_23 = _mm512_setzero_ps();

		__m512 acc_30 = _mm512_setzero_ps();
		__m512 acc_31 = _mm512_setzero_ps();
		__m512 acc_32 = _mm512_setzero_ps();
		__m512 acc_33 = _mm512_setzero_ps();

		__m512 acc_40 = _mm512_setzero_ps();
		__m512 acc_41 = _mm512_setzero_ps();
		__m512 acc_42 = _mm512_setzero_ps();
		__m512 acc_43 = _mm512_setzero_ps();

		__m512 acc_50 = _mm512_setzero_ps();
		__m512 acc_51 = _mm512_setzero_ps();
		__m512 acc_52 = _mm512_setzero_ps();
		__m512 acc_53 = _mm512_setzero_ps();

		dim_t group_start = grp_post_ops_attr.grp_post_op_k / group_size;
		dim_t group_end = ( grp_post_ops_attr.grp_post_op_k + k0 - 1 ) / group_size;

		int8_t *a_group = ( int8_t* )a;
		int8_t *b_group = ( int8_t* )b;

		for( dim_t group = group_start; group <= group_end; group++ )
		{
			dim_t k_start = bli_max( group * group_size,
				                     grp_post_ops_attr.grp_post_op_k );
			dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                grp_post_ops_attr.grp_post_op_k + k0 - 1);

			dim_t kg0 = k_end - k_start + 1;
			dim_t k_full_pieces = kg0 / 4;
			dim_t k_partial_pieces = kg0 % 4;

			// Registers to use for accumulating C.
			__m512i c_int32_0p0 = _mm512_setzero_epi32();
			__m512i c_int32_0p1 = _mm512_setzero_epi32();
			__m512i c_int32_0p2 = _mm512_setzero_epi32();
			__m512i c_int32_0p3 = _mm512_setzero_epi32();

			__m512i c_int32_1p0 = _mm512_setzero_epi32();
			__m512i c_int32_1p1 = _mm512_setzero_epi32();
			__m512i c_int32_1p2 = _mm512_setzero_epi32();
			__m512i c_int32_1p3 = _mm512_setzero_epi32();

			__m512i c_int32_2p0 = _mm512_setzero_epi32();
			__m512i c_int32_2p1 = _mm512_setzero_epi32();
			__m512i c_int32_2p2 = _mm512_setzero_epi32();
			__m512i c_int32_2p3 = _mm512_setzero_epi32();

			__m512i c_int32_3p0 = _mm512_setzero_epi32();
			__m512i c_int32_3p1 = _mm512_setzero_epi32();
			__m512i c_int32_3p2 = _mm512_setzero_epi32();
			__m512i c_int32_3p3 = _mm512_setzero_epi32();

			__m512i c_int32_4p0 = _mm512_setzero_epi32();
			__m512i c_int32_4p1 = _mm512_setzero_epi32();
			__m512i c_int32_4p2 = _mm512_setzero_epi32();
			__m512i c_int32_4p3 = _mm512_setzero_epi32();

			__m512i c_int32_5p0 = _mm512_setzero_epi32();
			__m512i c_int32_5p1 = _mm512_setzero_epi32();
			__m512i c_int32_5p2 = _mm512_setzero_epi32();
			__m512i c_int32_5p3 = _mm512_setzero_epi32();

			for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
			{
				// The instructions are arranged in a mixed way to reduce data
				// chain dependencies.

				// Load 4 rows with 64 elements each from B to 4 ZMM registers. It
				// is to be noted that the B matrix is packed for use in vnni
				// instructions and each load to ZMM register will have 4 elements
				// along k direction and 16 elements across n directions, so 4x16
				// elements to a ZMM register.
				b0 = _mm512_loadu_si512( b_group + ( rs_b * kr ) + ( cs_b * 0 ) );

				// Broadcast a[0,kr:kr+4].
				a_int32_0 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                               ( rs_a * 0 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				b1 = _mm512_loadu_si512( b_group + ( rs_b * kr ) + ( cs_b * 1 ) );
				b2 = _mm512_loadu_si512( b_group + ( rs_b * kr ) + ( cs_b * 2 ) );
				b3 = _mm512_loadu_si512( b_group + ( rs_b * kr ) + ( cs_b * 3 ) );

			// Perform column direction mat-mul with k = 4.
				// c[0,0-63] = a[0,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

				// Broadcast a[1,kr:kr+4].
				a_int32_1 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                                ( rs_a * 1 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_0p1 = _mm512_dpbusd_epi32( c_int32_0p1, a_int32_0, b1 );
				c_int32_0p2 = _mm512_dpbusd_epi32( c_int32_0p2, a_int32_0, b2 );
				c_int32_0p3 = _mm512_dpbusd_epi32( c_int32_0p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[1,0-63] = a[1,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

				// Broadcast a[2,kr:kr+4].
				a_int32_0 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                                ( rs_a * 2 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				c_int32_1p1 = _mm512_dpbusd_epi32( c_int32_1p1, a_int32_1, b1 );
				c_int32_1p2 = _mm512_dpbusd_epi32( c_int32_1p2, a_int32_1, b2 );
				c_int32_1p3 = _mm512_dpbusd_epi32( c_int32_1p3, a_int32_1, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[2,0-63] = a[2,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_0, b0 );

				// Broadcast a[3,kr:kr+4].
				a_int32_1 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                           ( rs_a * 3 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_2p1 = _mm512_dpbusd_epi32( c_int32_2p1, a_int32_0, b1 );
				c_int32_2p2 = _mm512_dpbusd_epi32( c_int32_2p2, a_int32_0, b2 );
				c_int32_2p3 = _mm512_dpbusd_epi32( c_int32_2p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[3,0-63] = a[3,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_1, b0 );

				// Broadcast a[4,kr:kr+4].
				a_int32_0 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                                ( rs_a * 4 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				c_int32_3p1 = _mm512_dpbusd_epi32( c_int32_3p1, a_int32_1, b1 );
				c_int32_3p2 = _mm512_dpbusd_epi32( c_int32_3p2, a_int32_1, b2 );
				c_int32_3p3 = _mm512_dpbusd_epi32( c_int32_3p3, a_int32_1, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[4,0-63] = a[4,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_0, b0 );

				// Broadcast a[5,kr:kr+4].
				a_int32_1 = _mm512_set1_epi32( *( int32_t* )( a_group +
					                                   ( rs_a * 5 ) + ( cs_a * kr ) ) );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_4p1 = _mm512_dpbusd_epi32( c_int32_4p1, a_int32_0, b1 );
				c_int32_4p2 = _mm512_dpbusd_epi32( c_int32_4p2, a_int32_0, b2 );
				c_int32_4p3 = _mm512_dpbusd_epi32( c_int32_4p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[5,0-63] = a[5,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_1, b0 );
				c_int32_5p1 = _mm512_dpbusd_epi32( c_int32_5p1, a_int32_1, b1 );
				c_int32_5p2 = _mm512_dpbusd_epi32( c_int32_5p2, a_int32_1, b2 );
				c_int32_5p3 = _mm512_dpbusd_epi32( c_int32_5p3, a_int32_1, b3 );
			} // k-loop

			a_group += k_full_pieces * cs_a;
			b_group += k_full_pieces * rs_b;

			// Handle k remainder.
			if ( k_partial_pieces > 0 )
			{
				__m128i a_kfringe_buf;
				__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - k_partial_pieces ) );

				b0 = _mm512_loadu_si512( b_group + ( cs_b * 0 ) );

				// Broadcast a[0,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 0 ))
				);
				a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				b1 = _mm512_loadu_si512( b_group + ( cs_b * 1 ) );
				b2 = _mm512_loadu_si512( b_group + ( cs_b * 2 ) );
				b3 = _mm512_loadu_si512( b_group + ( cs_b * 3 ) );

				// Perform column direction mat-mul with k = 4.
				// c[0,0-63] = a[0,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

				// Broadcast a[1,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 1 ) )
				);
				a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_0p1 = _mm512_dpbusd_epi32( c_int32_0p1, a_int32_0, b1 );
				c_int32_0p2 = _mm512_dpbusd_epi32( c_int32_0p2, a_int32_0, b2 );
				c_int32_0p3 = _mm512_dpbusd_epi32( c_int32_0p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[1,0-63] = a[1,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

				// Broadcast a[2,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 2 ) )
				);
				a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				c_int32_1p1 = _mm512_dpbusd_epi32( c_int32_1p1, a_int32_1, b1 );
				c_int32_1p2 = _mm512_dpbusd_epi32( c_int32_1p2, a_int32_1, b2 );
				c_int32_1p3 = _mm512_dpbusd_epi32( c_int32_1p3, a_int32_1, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[2,0-63] = a[2,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_0, b0 );

				// Broadcast a[3,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 3 ) )
				);
				a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_2p1 = _mm512_dpbusd_epi32( c_int32_2p1, a_int32_0, b1 );
				c_int32_2p2 = _mm512_dpbusd_epi32( c_int32_2p2, a_int32_0, b2 );
				c_int32_2p3 = _mm512_dpbusd_epi32( c_int32_2p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[3,0-63] = a[3,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_1, b0 );

				// Broadcast a[4,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 4 ) )
				);
				a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_0 = _mm512_add_epi8( a_int32_0, vec_uint8 );

				c_int32_3p1 = _mm512_dpbusd_epi32( c_int32_3p1, a_int32_1, b1 );
				c_int32_3p2 = _mm512_dpbusd_epi32( c_int32_3p2, a_int32_1, b2 );
				c_int32_3p3 = _mm512_dpbusd_epi32( c_int32_3p3, a_int32_1, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[4,0-63] = a[4,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_0, b0 );

				// Broadcast a[5,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8
				(
				load_mask,
				( a_group + ( rs_a * 5 ) )
				);
				a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

				//convert signed int8 to uint8 for VNNI
				a_int32_1 = _mm512_add_epi8 (a_int32_1, vec_uint8);

				c_int32_4p1 = _mm512_dpbusd_epi32( c_int32_4p1, a_int32_0, b1 );
				c_int32_4p2 = _mm512_dpbusd_epi32( c_int32_4p2, a_int32_0, b2 );
				c_int32_4p3 = _mm512_dpbusd_epi32( c_int32_4p3, a_int32_0, b3 );

				// Perform column direction mat-mul with k = 4.
				// c[5,0-63] = a[5,kr:kr+4]*b[kr:kr+4,0-63]
				c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_1, b0 );
				c_int32_5p1 = _mm512_dpbusd_epi32( c_int32_5p1, a_int32_1, b1 );
				c_int32_5p2 = _mm512_dpbusd_epi32( c_int32_5p2, a_int32_1, b2 );
				c_int32_5p3 = _mm512_dpbusd_epi32( c_int32_5p3, a_int32_1, b3 );
			} // k_partial_pieces

			// Subtract B matrix sum column values to compensate
			// for addition of 128 to A matrix elements
			int32_t* bsumptr = post_ops_attr.b_col_sum_vec +
			                    (group * grp_post_ops_attr.grp_post_op_sum_ld)
			                     + post_ops_attr.b_sum_offset;

			b0 = _mm512_loadu_si512( bsumptr );
			c_int32_0p0 = _mm512_sub_epi32( c_int32_0p0 , b0 );
			c_int32_1p0 = _mm512_sub_epi32( c_int32_1p0 , b0 );
			c_int32_2p0 = _mm512_sub_epi32( c_int32_2p0 , b0 );
			c_int32_3p0 = _mm512_sub_epi32( c_int32_3p0 , b0 );
			c_int32_4p0 = _mm512_sub_epi32( c_int32_4p0 , b0 );
			c_int32_5p0 = _mm512_sub_epi32( c_int32_5p0 , b0 );

			b0 = _mm512_loadu_si512( bsumptr + 16 );

			c_int32_0p1 = _mm512_sub_epi32( c_int32_0p1 , b0 );
			c_int32_1p1 = _mm512_sub_epi32( c_int32_1p1 , b0 );
			c_int32_2p1 = _mm512_sub_epi32( c_int32_2p1 , b0 );
			c_int32_3p1 = _mm512_sub_epi32( c_int32_3p1 , b0 );
			c_int32_4p1 = _mm512_sub_epi32( c_int32_4p1 , b0 );
			c_int32_5p1 = _mm512_sub_epi32( c_int32_5p1 , b0 );

			b0 = _mm512_loadu_si512( bsumptr + 32 );

			c_int32_0p2 = _mm512_sub_epi32( c_int32_0p2 , b0 );
			c_int32_1p2 = _mm512_sub_epi32( c_int32_1p2 , b0 );
			c_int32_2p2 = _mm512_sub_epi32( c_int32_2p2 , b0 );
			c_int32_3p2 = _mm512_sub_epi32( c_int32_3p2 , b0 );
			c_int32_4p2 = _mm512_sub_epi32( c_int32_4p2 , b0 );
			c_int32_5p2 = _mm512_sub_epi32( c_int32_5p2 , b0 );

			b0 = _mm512_loadu_si512( bsumptr + 48 );

			c_int32_0p3 = _mm512_sub_epi32( c_int32_0p3 , b0 );
			c_int32_1p3 = _mm512_sub_epi32( c_int32_1p3 , b0 );
			c_int32_2p3 = _mm512_sub_epi32( c_int32_2p3 , b0 );
			c_int32_3p3 = _mm512_sub_epi32( c_int32_3p3 , b0 );
			c_int32_4p3 = _mm512_sub_epi32( c_int32_4p3 , b0 );
			c_int32_5p3 = _mm512_sub_epi32( c_int32_5p3 , b0 );

			__m512 b_scl0, b_scl1, b_scl2, b_scl3;
			__mmask16 scl_mask = _cvtu32_mask16( 0xFFFF );
			if( grp_post_ops_attr.sf_stor_type == BF16 )
			{
				// load scales for B matrix
				bfloat16* b_scale_ptr = ((bfloat16*)(grp_post_ops_attr.b_scale_factor))
				+ ( group * grp_post_ops_attr.grp_post_op_ldb )
				+ grp_post_ops_attr.grp_post_op_j;

				SYM_QUANT_BF16_F32_SCL_LOAD(b_scl0, b_scale_ptr, scl_mask, 0)
				SYM_QUANT_BF16_F32_SCL_LOAD(b_scl1, b_scale_ptr, scl_mask, 1)
				SYM_QUANT_BF16_F32_SCL_LOAD(b_scl2, b_scale_ptr, scl_mask, 2)
				SYM_QUANT_BF16_F32_SCL_LOAD(b_scl3, b_scale_ptr, scl_mask, 3)
			}
			else
			{
				// load scales for B matrix
				float* b_scale_ptr = ((float*)(grp_post_ops_attr.b_scale_factor))
				+ ( group * grp_post_ops_attr.grp_post_op_ldb )
				+ grp_post_ops_attr.grp_post_op_j;

				SYM_QUANT_F32_F32_SCL_LOAD(b_scl0, b_scale_ptr, scl_mask, 0)
				SYM_QUANT_F32_F32_SCL_LOAD(b_scl1, b_scale_ptr, scl_mask, 1)
				SYM_QUANT_F32_F32_SCL_LOAD(b_scl2, b_scale_ptr, scl_mask, 2)
				SYM_QUANT_F32_F32_SCL_LOAD(b_scl3, b_scale_ptr, scl_mask, 3)
			}

			// Load and apply 2 scales of A matrix at a time to ensure
			// there is no register spillage.
			__m512 a_scl0, a_scl1;

			if( grp_post_ops_attr.sf_stor_type == BF16 )
			{
				bfloat16* a_scale_ptr = ((bfloat16*)(grp_post_ops_attr.a_scale_factor)) +
			                        ( grp_post_ops_attr.grp_post_op_i *
										grp_post_ops_attr.grp_post_op_lda )
			                        + group;

				// ----------- rows 0 & 1 -----------------------------
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl0, a_scale_ptr, 0)
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl1, a_scale_ptr, 1)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 0)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 1)

				// ----------- rows 2 & 3 -----------------------------
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl0, a_scale_ptr, 2)
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl1, a_scale_ptr, 3)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 2)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 3)

				// ----------- rows 4 & 5 -----------------------------
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl0, a_scale_ptr, 4)
				SYM_QUANT_BF16_F32_SCL_BCST(a_scl1, a_scale_ptr, 5)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 4)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 5)
			}
			else
			{
				float* a_scale_ptr = ((float*)(grp_post_ops_attr.a_scale_factor)) +
			                        ( grp_post_ops_attr.grp_post_op_i *
										grp_post_ops_attr.grp_post_op_lda )
			                        + group;

				// ----------- rows 0 & 1 -----------------------------
				SYM_QUANT_F32_F32_SCL_BCST(a_scl0, a_scale_ptr, 0)
				SYM_QUANT_F32_F32_SCL_BCST(a_scl1, a_scale_ptr, 1)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 0)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 1)

				// ----------- rows 2 & 3 -----------------------------
				SYM_QUANT_F32_F32_SCL_BCST(a_scl0, a_scale_ptr, 2)
				SYM_QUANT_F32_F32_SCL_BCST(a_scl1, a_scale_ptr, 3)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 2)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 3)

				// ----------- rows 4 & 5 -----------------------------
				SYM_QUANT_F32_F32_SCL_BCST(a_scl0, a_scale_ptr, 4)
				SYM_QUANT_F32_F32_SCL_BCST(a_scl1, a_scale_ptr, 5)

				// convert int32_t regs to float and apply scales and then add
				// to acc_ regs.
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 0, 4)
				CVT_ACCUM_REG_APPLY_SCALES_4COL(acc_, c_int32_, 1, 5)
			}
		} // group loop

		// Load alpha and beta
		__m512 selector1 = _mm512_cvtepi32_ps(_mm512_set1_epi32( alpha ));
		__m512 selector2 = _mm512_cvtepi32_ps(_mm512_set1_epi32( beta ));

		if ( alpha != 1 )
		{
			// Scale by alpha
			acc_00 = _mm512_mul_ps( selector1, acc_00 );
			acc_01 = _mm512_mul_ps( selector1, acc_01 );
			acc_02 = _mm512_mul_ps( selector1, acc_02 );
			acc_03 = _mm512_mul_ps( selector1, acc_03 );

			acc_10 = _mm512_mul_ps( selector1, acc_10 );
			acc_11 = _mm512_mul_ps( selector1, acc_11 );
			acc_12 = _mm512_mul_ps( selector1, acc_12 );
			acc_13 = _mm512_mul_ps( selector1, acc_13 );

			acc_20 = _mm512_mul_ps( selector1, acc_20 );
			acc_21 = _mm512_mul_ps( selector1, acc_21 );
			acc_22 = _mm512_mul_ps( selector1, acc_22 );
			acc_23 = _mm512_mul_ps( selector1, acc_23 );

			acc_30 = _mm512_mul_ps( selector1, acc_30 );
			acc_31 = _mm512_mul_ps( selector1, acc_31 );
			acc_32 = _mm512_mul_ps( selector1, acc_32 );
			acc_33 = _mm512_mul_ps( selector1, acc_33 );

			acc_40 = _mm512_mul_ps( selector1, acc_40 );
			acc_41 = _mm512_mul_ps( selector1, acc_41 );
			acc_42 = _mm512_mul_ps( selector1, acc_42 );
			acc_43 = _mm512_mul_ps( selector1, acc_43 );

			acc_50 = _mm512_mul_ps( selector1, acc_50 );
			acc_51 = _mm512_mul_ps( selector1, acc_51 );
			acc_52 = _mm512_mul_ps( selector1, acc_52 );
			acc_53 = _mm512_mul_ps( selector1, acc_53 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			// For the downscaled api (C-s8), the output C matrix values needs
			// to be upscaled to s32 to be used for beta scale.
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,0,selector1,selector2);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,1,selector1,selector2);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,2,selector1,selector2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,3,selector1,selector2);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,4,selector1,selector2);

					// c[5:0-15,16-31,32-47,48-63]
					BF16_F32_BETA_OP4(ir,5,selector1,selector2);
				}
			}
			else// if ( post_ops_attr.c_stor_type == F32 )
			{
				// c[0:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,0,selector1,selector2);

				// c[1:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,1,selector1,selector2);

				// c[2:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,2,selector1,selector2);

				// c[3:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,3,selector1,selector2);

				// c[4:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,4,selector1,selector2);

				// c[5:0-15,16-31,32-47,48-63]
				F32_F32_BETA_OP4(ir,5,selector1,selector2);
			}
		}

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x64:
		{
			__m512 b0,b1,b2,b3;
			__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
			if ( post_ops_list_temp->stor_type == S8 )
			{
				S8_F32_BIAS_LOAD(b0, bias_mask, 0);
				S8_F32_BIAS_LOAD(b1, bias_mask, 1);
				S8_F32_BIAS_LOAD(b2, bias_mask, 2);
				S8_F32_BIAS_LOAD(b3, bias_mask, 3);
			}
			else if ( post_ops_list_temp->stor_type == U8 )
			{
				U8_F32_BIAS_LOAD(b0, bias_mask, 0);
				U8_F32_BIAS_LOAD(b1, bias_mask, 1);
				U8_F32_BIAS_LOAD(b2, bias_mask, 2);
				U8_F32_BIAS_LOAD(b3, bias_mask, 3);
			}
			else if ( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_LOAD(b0, bias_mask, 0);
				BF16_F32_BIAS_LOAD(b1, bias_mask, 1);
				BF16_F32_BIAS_LOAD(b2, bias_mask, 2);
				BF16_F32_BIAS_LOAD(b3, bias_mask, 3);
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				S32_F32_BIAS_LOAD(b0, bias_mask, 0);
				S32_F32_BIAS_LOAD(b1, bias_mask, 1);
				S32_F32_BIAS_LOAD(b2, bias_mask, 2);
				S32_F32_BIAS_LOAD(b3, bias_mask, 3);
			}
			else /*(stor_type == F32 )*/
			{
				b0 =
					_mm512_loadu_ps( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				b1 =
					_mm512_loadu_ps( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				b2 =
					_mm512_loadu_ps( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				b3 =
					_mm512_loadu_ps( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}

			// c[0, 0-15]
			acc_00 = _mm512_add_ps( b0, acc_00 );

			// c[0, 16-31]
			acc_01 = _mm512_add_ps( b1, acc_01 );

			// c[0, 32-47]
			acc_02 = _mm512_add_ps( b2, acc_02 );

			// c[0, 48-63]
			acc_03 = _mm512_add_ps( b3, acc_03 );

			// c[1, 0-15]
			acc_10 = _mm512_add_ps( b0, acc_10 );

			// c[1, 16-31]
			acc_11 = _mm512_add_ps( b1, acc_11 );

			// c[1, 32-47]
			acc_12 = _mm512_add_ps( b2, acc_12 );

			// c[1, 48-63]
			acc_13 = _mm512_add_ps( b3, acc_13 );

			// c[2, 0-15]
			acc_20 = _mm512_add_ps( b0, acc_20 );

			// c[2, 16-31]
			acc_21 = _mm512_add_ps( b1, acc_21 );

			// c[2, 32-47]
			acc_22 = _mm512_add_ps( b2, acc_22 );

			// c[2, 48-63]
			acc_23 = _mm512_add_ps( b3, acc_23 );

			// c[3, 0-15]
			acc_30 = _mm512_add_ps( b0, acc_30 );

			// c[3, 16-31]
			acc_31 = _mm512_add_ps( b1, acc_31 );

			// c[3, 32-47]
			acc_32 = _mm512_add_ps( b2, acc_32 );

			// c[3, 48-63]
			acc_33 = _mm512_add_ps( b3, acc_33 );

			// c[4, 0-15]
			acc_40 = _mm512_add_ps( b0, acc_40 );

			// c[4, 16-31]
			acc_41 = _mm512_add_ps( b1, acc_41 );

			// c[4, 32-47]
			acc_42 = _mm512_add_ps( b2, acc_42 );

			// c[4, 48-63]
			acc_43 = _mm512_add_ps( b3, acc_43 );

			// c[5, 0-15]
			acc_50 = _mm512_add_ps( b0, acc_50 );

			// c[5, 16-31]
			acc_51 = _mm512_add_ps( b1, acc_51 );

			// c[5, 32-47]
			acc_52 = _mm512_add_ps( b2, acc_52 );

			// c[5, 48-63]
			acc_53 = _mm512_add_ps( b3, acc_53 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x64:
		{
			__m512 zero = _mm512_setzero_ps();

			// c[0, 0-15]
			acc_00 = _mm512_max_ps( zero, acc_00 );

			// c[0, 16-31]
			acc_01 = _mm512_max_ps( zero, acc_01 );

			// c[0, 32-47]
			acc_02 = _mm512_max_ps( zero, acc_02 );

			// c[0, 48-63]
			acc_03 = _mm512_max_ps( zero, acc_03 );

			// c[1, 0-15]
			acc_10 = _mm512_max_ps( zero, acc_10 );

			// c[1, 16-31]
			acc_11 = _mm512_max_ps( zero, acc_11 );

			// c[1, 32-47]
			acc_12 = _mm512_max_ps( zero, acc_12 );

			// c[1, 48-63]
			acc_13 = _mm512_max_ps( zero, acc_13 );

			// c[2, 0-15]
			acc_20 = _mm512_max_ps( zero, acc_20 );

			// c[2, 16-31]
			acc_21 = _mm512_max_ps( zero, acc_21 );

			// c[2, 32-47]
			acc_22 = _mm512_max_ps( zero, acc_22 );

			// c[2, 48-63]
			acc_23 = _mm512_max_ps( zero, acc_23 );

			// c[3, 0-15]
			acc_30 = _mm512_max_ps( zero, acc_30 );

			// c[3, 16-31]
			acc_31 = _mm512_max_ps( zero, acc_31 );

			// c[3, 32-47]
			acc_32 = _mm512_max_ps( zero, acc_32 );

			// c[3, 48-63]
			acc_33 = _mm512_max_ps( zero, acc_33 );

			// c[4, 0-15]
			acc_40 = _mm512_max_ps( zero, acc_40 );

			// c[4, 16-31]
			acc_41 = _mm512_max_ps( zero, acc_41 );

			// c[4, 32-47]
			acc_42 = _mm512_max_ps( zero, acc_42 );

			// c[4, 48-63]
			acc_43 = _mm512_max_ps( zero, acc_43 );

			// c[5, 0-15]
			acc_50 = _mm512_max_ps( zero, acc_50 );

			// c[5, 16-31]
			acc_51 = _mm512_max_ps( zero, acc_51 );

			// c[5, 32-47]
			acc_52 = _mm512_max_ps( zero, acc_52 );

			// c[5, 48-63]
			acc_53 = _mm512_max_ps( zero, acc_53 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x64:
		{
			__m512 zero = _mm512_setzero_ps();
			__m512 scale;

			if ( (post_ops_attr.c_stor_type == S32 ) ||
				 (post_ops_attr.c_stor_type == U8 ) ||
			     (post_ops_attr.c_stor_type == S8 ) )
			{
				scale = _mm512_cvtepi32_ps
						(_mm512_set1_epi32(
							*( ( int32_t* )post_ops_list_temp->op_args2 ) ));
			}
			else
			{
				scale = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
			}

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_00)

			// c[0, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_01)

			// c[0, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_02)

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_03)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_10)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_11)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_12)

			// c[1, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_13)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_20)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_21)

			// c[2, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_22)

			// c[2, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_23)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_30)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_31)

			// c[3, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_32)

			// c[3, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_33)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_40)

			// c[4, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_41)

			// c[4, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_42)

			// c[4, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_43)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_50)

			// c[5, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_51)

			// c[5, 32-47]
			RELU_SCALE_OP_F32_AVX512(acc_52)

			// c[5, 48-63]
			RELU_SCALE_OP_F32_AVX512(acc_53)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_00, y, r, r2, x, z, dn, tmpout)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_01, y, r, r2, x, z, dn, tmpout)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_02, y, r, r2, x, z, dn, tmpout)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_03, y, r, r2, x, z, dn, tmpout)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_10, y, r, r2, x, z, dn, tmpout)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_11, y, r, r2, x, z, dn, tmpout)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_12, y, r, r2, x, z, dn, tmpout)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_13, y, r, r2, x, z, dn, tmpout)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_20, y, r, r2, x, z, dn, tmpout)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_21, y, r, r2, x, z, dn, tmpout)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_22, y, r, r2, x, z, dn, tmpout)

			// c[2, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_23, y, r, r2, x, z, dn, tmpout)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_30, y, r, r2, x, z, dn, tmpout)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_31, y, r, r2, x, z, dn, tmpout)

			// c[3, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_32, y, r, r2, x, z, dn, tmpout)

			// c[3, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_33, y, r, r2, x, z, dn, tmpout)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_40, y, r, r2, x, z, dn, tmpout)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_41, y, r, r2, x, z, dn, tmpout)

			// c[4, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_42, y, r, r2, x, z, dn, tmpout)

			// c[4, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_43, y, r, r2, x, z, dn, tmpout)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_50, y, r, r2, x, z, dn, tmpout)

			// c[5, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_51, y, r, r2, x, z, dn, tmpout)

			// c[5, 32-47]
			GELU_TANH_F32_AVX512_DEF(acc_52, y, r, r2, x, z, dn, tmpout)

			// c[5, 48-63]
			GELU_TANH_F32_AVX512_DEF(acc_53, y, r, r2, x, z, dn, tmpout)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6x64:
		{
			__m512 y, r, r2;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_00, y, r, r2)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_01, y, r, r2)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_02, y, r, r2)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_03, y, r, r2)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_10, y, r, r2)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_11, y, r, r2)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_12, y, r, r2)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_13, y, r, r2)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_20, y, r, r2)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_21, y, r, r2)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_22, y, r, r2)

			// c[2, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_23, y, r, r2)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_30, y, r, r2)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_31, y, r, r2)

			// c[3, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_32, y, r, r2)

			// c[3, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_33, y, r, r2)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_40, y, r, r2)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_41, y, r, r2)

			// c[4, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_42, y, r, r2)

			// c[4, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_43, y, r, r2)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_50, y, r, r2)

			// c[5, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_51, y, r, r2)

			// c[5, 32-47]
			GELU_ERF_F32_AVX512_DEF(acc_52, y, r, r2)

			// c[5, 48-63]
			GELU_ERF_F32_AVX512_DEF(acc_53, y, r, r2)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_6x64:
		{
			__m512 min = _mm512_setzero_ps();
			__m512 max = _mm512_setzero_ps();


			if ( post_ops_attr.c_stor_type == S32 ||
				 post_ops_attr.c_stor_type == U8 ||
			     post_ops_attr.c_stor_type == S8 )
			{
				min = _mm512_cvtepi32_ps
						(_mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args2 ));
				max = _mm512_cvtepi32_ps
						(_mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args3 ));
			}else{
				min = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
				max = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args3 ) );
			}
			// c[0, 0-15]
			CLIP_F32_AVX512(acc_00, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(acc_01, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(acc_02, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(acc_03, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(acc_10, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(acc_11, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(acc_12, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(acc_13, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(acc_20, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(acc_21, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(acc_22, min, max)

			// c[2, 48-63]
			CLIP_F32_AVX512(acc_23, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(acc_30, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(acc_31, min, max)

			// c[3, 32-47]
			CLIP_F32_AVX512(acc_32, min, max)

			// c[3, 48-63]
			CLIP_F32_AVX512(acc_33, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(acc_40, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(acc_41, min, max)

			// c[4, 32-47]
			CLIP_F32_AVX512(acc_42, min, max)

			// c[4, 48-63]
			CLIP_F32_AVX512(acc_43, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(acc_50, min, max)

			// c[5, 16-31]
			CLIP_F32_AVX512(acc_51, min, max)

			// c[5, 32-47]
			CLIP_F32_AVX512(acc_52, min, max)

			// c[5, 48-63]
			CLIP_F32_AVX512(acc_53, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_6x64:
		{
			__m512 scale0, scale1, scale2, scale3;
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF );

			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				if( post_ops_list_temp->sf_stor_type == U8 )
				{
					U8_F32_SCALE_LOAD(scale0,load_mask, 0)
					U8_F32_SCALE_LOAD(scale1,load_mask, 1)
					U8_F32_SCALE_LOAD(scale2,load_mask, 2)
					U8_F32_SCALE_LOAD(scale3,load_mask, 3)
				}
				else if( post_ops_list_temp->sf_stor_type == S8 )
				{
					S8_F32_SCALE_LOAD(scale0,load_mask, 0)
					S8_F32_SCALE_LOAD(scale1,load_mask, 1)
					S8_F32_SCALE_LOAD(scale2,load_mask, 2)
					S8_F32_SCALE_LOAD(scale3,load_mask, 3)
				}
				else if( post_ops_list_temp->sf_stor_type == S32 )
				{
					S32_F32_SCALE_LOAD(scale0,load_mask, 0)
					S32_F32_SCALE_LOAD(scale1,load_mask, 1)
					S32_F32_SCALE_LOAD(scale2,load_mask, 2)
					S32_F32_SCALE_LOAD(scale3,load_mask, 3)
				}
				else if( post_ops_list_temp->sf_stor_type == BF16 )
				{
					BF16_F32_SCALE_LOAD(scale0,load_mask, 0)
					BF16_F32_SCALE_LOAD(scale1,load_mask, 1)
					BF16_F32_SCALE_LOAD(scale2,load_mask, 2)
					BF16_F32_SCALE_LOAD(scale3,load_mask, 3)
				}
				else
				{
					scale0 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scale1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scale2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scale3=
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
			}
			else /*if ( post_ops_list_temp->scale_factor_len == 1 )*/
			{
				if( post_ops_list_temp->sf_stor_type == U8 )
				{
					U8_F32_SCALE_BCST(scale0,0)
					U8_F32_SCALE_BCST(scale1,1)
					U8_F32_SCALE_BCST(scale2,2)
					U8_F32_SCALE_BCST(scale3,3)
				}
				else if( post_ops_list_temp->sf_stor_type == S8 )
				{
					S8_F32_SCALE_BCST(scale0,0)
					S8_F32_SCALE_BCST(scale1,1)
					S8_F32_SCALE_BCST(scale2,2)
					S8_F32_SCALE_BCST(scale3,3)
				}
				else if( post_ops_list_temp->sf_stor_type == S32 )
				{
					S32_F32_SCALE_BCST(scale0,0)
					S32_F32_SCALE_BCST(scale1,1)
					S32_F32_SCALE_BCST(scale2,2)
					S32_F32_SCALE_BCST(scale3,3)
				}
				else if( post_ops_list_temp->sf_stor_type == BF16 )
				{
					BF16_F32_SCALE_BCST(scale0,0)
					BF16_F32_SCALE_BCST(scale1,1)
					BF16_F32_SCALE_BCST(scale2,2)
					BF16_F32_SCALE_BCST(scale3,3)
				}
				else
				{
					scale0 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
					scale1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
					scale2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
					scale3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				}
			}

			// Need to ensure sse not used to avoid avx512 -> sse transition.
			__m128i zero_point0 = _mm512_castsi512_si128( _mm512_setzero_si512() );
			__m128i zero_point1 = _mm512_castsi512_si128( _mm512_setzero_si512() );
			__m128i zero_point2 = _mm512_castsi512_si128( _mm512_setzero_si512() );
			__m128i zero_point3 = _mm512_castsi512_si128( _mm512_setzero_si512() );

			// int8_t zero point value.
			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				if( post_ops_list_temp->zp_stor_type == S32 )
				{
					S32_S8_ZP_LOAD(zero_point0,load_mask, 0)
					S32_S8_ZP_LOAD(zero_point1,load_mask, 1)
					S32_S8_ZP_LOAD(zero_point2,load_mask, 2)
					S32_S8_ZP_LOAD(zero_point3,load_mask, 3)
				}
				else
				{
					zero_point0 = _mm_loadu_si128( ( __m128i const* )
									( ( int8_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = _mm_loadu_si128( ( __m128i const* )
									( ( int8_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = _mm_loadu_si128( ( __m128i const* )
									( ( int8_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = _mm_loadu_si128( ( __m128i const* )
									( ( int8_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}
			}
			else if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
			{
				if( post_ops_list_temp->zp_stor_type == S32 )
				{
					S32_S8_ZP_BCST(zero_point0)
					S32_S8_ZP_BCST(zero_point1)
					S32_S8_ZP_BCST(zero_point2)
					S32_S8_ZP_BCST(zero_point3)
				}
				else
				{
					zero_point0 = _mm_maskz_set1_epi8( 0xFFFF,
									*( ( int8_t* )post_ops_list_temp->op_args1 ) );
					zero_point1 = _mm_maskz_set1_epi8( 0xFFFF,
									*( ( int8_t* )post_ops_list_temp->op_args1 ) );
					zero_point2 = _mm_maskz_set1_epi8( 0xFFFF,
									*( ( int8_t* )post_ops_list_temp->op_args1 ) );
					zero_point3 = _mm_maskz_set1_epi8( 0xFFFF,
									*( ( int8_t* )post_ops_list_temp->op_args1 ) );
				}
			}

			// c[0, 0-15]
			CVT_MULRND_F32(acc_00,scale0,zero_point0);

			// c[0, 16-31]
			CVT_MULRND_F32(acc_01,scale1,zero_point1);

			// c[0, 32-47]
			CVT_MULRND_F32(acc_02,scale2,zero_point2);

			// c[0, 48-63]
			CVT_MULRND_F32(acc_03,scale3,zero_point3);

			// c[1, 0-15]
			CVT_MULRND_F32(acc_10,scale0,zero_point0);

			// c[1, 16-31]
			CVT_MULRND_F32(acc_11,scale1,zero_point1);

			// c[1, 32-47]
			CVT_MULRND_F32(acc_12,scale2,zero_point2);

			// c[1, 48-63]
			CVT_MULRND_F32(acc_13,scale3,zero_point3);

			// c[2, 0-15]
			CVT_MULRND_F32(acc_20,scale0,zero_point0);

			// c[2, 16-31]
			CVT_MULRND_F32(acc_21,scale1,zero_point1);

			// c[2, 32-47]
			CVT_MULRND_F32(acc_22,scale2,zero_point2);

			// c[2, 48-63]
			CVT_MULRND_F32(acc_23,scale3,zero_point3);

			// c[3, 0-15]
			CVT_MULRND_F32(acc_30,scale0,zero_point0);

			// c[3, 16-31]
			CVT_MULRND_F32(acc_31,scale1,zero_point1);

			// c[3, 32-47]
			CVT_MULRND_F32(acc_32,scale2,zero_point2);

			// c[3, 48-63]
			CVT_MULRND_F32(acc_33,scale3,zero_point3);

			// c[4, 0-15]
			CVT_MULRND_F32(acc_40,scale0,zero_point0);

			// c[4, 16-31]
			CVT_MULRND_F32(acc_41,scale1,zero_point1);

			// c[4, 32-47]
			CVT_MULRND_F32(acc_42,scale2,zero_point2);

			// c[4, 48-63]
			CVT_MULRND_F32(acc_43,scale3,zero_point3);

			// c[5, 0-15]
			CVT_MULRND_F32(acc_50,scale0,zero_point0);

			// c[5, 16-31]
			CVT_MULRND_F32(acc_51,scale1,zero_point1);

			// c[5, 32-47]
			CVT_MULRND_F32(acc_52,scale2,zero_point2);

			// c[5, 48-63]
			CVT_MULRND_F32(acc_53,scale3,zero_point3);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_6x64:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == S8 ) );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_f32 = ( post_ops_list_temp->stor_type == F32 );
			bool is_u8 = ( post_ops_list_temp->stor_type == U8 );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 t0, t1, t2, t3;

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					BF16_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					F32_ONLY_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_u8 == TRUE )
			{
				uint8_t* matptr = ( uint8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_6x64:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == S8 ) );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_f32 = ( post_ops_list_temp->stor_type == F32 );
			bool is_u8 = ( post_ops_list_temp->stor_type == U8 );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			__m512 t0,t1,t2,t3;

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					F32_U8S8_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else if ( is_u8 == TRUE )
			{
				uint8_t* matptr = ( uint8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					U8_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);

					// c[5:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL(t0,t1,t2,t3,\
							scl_fctr6,scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_6x64:
		{
			__m512 scale;

			if ( (post_ops_attr.c_stor_type == S32 ) ||
				 (post_ops_attr.c_stor_type == U8 ) ||
			     (post_ops_attr.c_stor_type == S8 ) )
			{
				scale = _mm512_cvtepi32_ps
						(_mm512_set1_epi32(
							*( ( int32_t* )post_ops_list_temp->op_args2 ) ));
			}
			else
			{
				scale = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
			}

			__m512 al_in, r, r2, z, dn;
			__m512i temp;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(acc_00, scale, al_in, r, r2, z, dn, temp);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(acc_01, scale, al_in, r, r2, z, dn, temp);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(acc_02, scale, al_in, r, r2, z, dn, temp);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(acc_03, scale, al_in, r, r2, z, dn, temp);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(acc_10, scale, al_in, r, r2, z, dn, temp);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(acc_11, scale, al_in, r, r2, z, dn, temp);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(acc_12, scale, al_in, r, r2, z, dn, temp);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(acc_13, scale, al_in, r, r2, z, dn, temp);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(acc_20, scale, al_in, r, r2, z, dn, temp);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(acc_21, scale, al_in, r, r2, z, dn, temp);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(acc_22, scale, al_in, r, r2, z, dn, temp);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(acc_23, scale, al_in, r, r2, z, dn, temp);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(acc_30, scale, al_in, r, r2, z, dn, temp);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(acc_31, scale, al_in, r, r2, z, dn, temp);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(acc_32, scale, al_in, r, r2, z, dn, temp);

			// c[3, 48-63]
			SWISH_F32_AVX512_DEF(acc_33, scale, al_in, r, r2, z, dn, temp);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(acc_40, scale, al_in, r, r2, z, dn, temp);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(acc_41, scale, al_in, r, r2, z, dn, temp);

			// c[4, 32-47]
			SWISH_F32_AVX512_DEF(acc_42, scale, al_in, r, r2, z, dn, temp);

			// c[4, 48-63]
			SWISH_F32_AVX512_DEF(acc_43, scale, al_in, r, r2, z, dn, temp);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(acc_50, scale, al_in, r, r2, z, dn, temp);

			// c[5, 16-31]
			SWISH_F32_AVX512_DEF(acc_51, scale, al_in, r, r2, z, dn, temp);

			// c[5, 32-47]
			SWISH_F32_AVX512_DEF(acc_52, scale, al_in, r, r2, z, dn, temp);

			// c[5, 48-63]
			SWISH_F32_AVX512_DEF(acc_53, scale, al_in, r, r2, z, dn, temp);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_6x64:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(acc_00, r, r2, x, z, dn, q)

			// c[0, 16-31]
			TANHF_AVX512(acc_01, r, r2, x, z, dn, q);

			// c[0, 32-47]
			TANHF_AVX512(acc_02, r, r2, x, z, dn, q);

			// c[0, 48-63]
			TANHF_AVX512(acc_03, r, r2, x, z, dn, q);

			// c[1, 0-15]
			TANHF_AVX512(acc_10, r, r2, x, z, dn, q);

			// c[1, 16-31]
			TANHF_AVX512(acc_11, r, r2, x, z, dn, q);

			// c[1, 32-47]
			TANHF_AVX512(acc_12, r, r2, x, z, dn, q);

			// c[1, 48-63]
			TANHF_AVX512(acc_13, r, r2, x, z, dn, q);

			// c[2, 0-15]
			TANHF_AVX512(acc_20, r, r2, x, z, dn, q);

			// c[2, 16-31]
			TANHF_AVX512(acc_21, r, r2, x, z, dn, q);

			// c[2, 32-47]
			TANHF_AVX512(acc_22, r, r2, x, z, dn, q);

			// c[2, 48-63]
			TANHF_AVX512(acc_23, r, r2, x, z, dn, q);

			// c[3, 0-15]
			TANHF_AVX512(acc_30, r, r2, x, z, dn, q);

			// c[3, 16-31]
			TANHF_AVX512(acc_31, r, r2, x, z, dn, q);

			// c[3, 32-47]
			TANHF_AVX512(acc_32, r, r2, x, z, dn, q);

			// c[3, 48-63]
			TANHF_AVX512(acc_33, r, r2, x, z, dn, q);

			// c[4, 0-15]
			TANHF_AVX512(acc_40, r, r2, x, z, dn, q);

			// c[4, 16-31]
			TANHF_AVX512(acc_41, r, r2, x, z, dn, q);

			// c[4, 32-47]
			TANHF_AVX512(acc_42, r, r2, x, z, dn, q);

			// c[4, 48-63]
			TANHF_AVX512(acc_43, r, r2, x, z, dn, q);

			// c[5, 0-15]
			TANHF_AVX512(acc_50, r, r2, x, z, dn, q);

			// c[5, 16-31]
			TANHF_AVX512(acc_51, r, r2, x, z, dn, q);

			// c[5, 32-47]
			TANHF_AVX512(acc_52, r, r2, x, z, dn, q);

			// c[5, 48-63]
			TANHF_AVX512(acc_53, r, r2, x, z, dn, q);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_6x64:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_00, al_in, r, r2, z, dn, tmpout);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_01, al_in, r, r2, z, dn, tmpout);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_02, al_in, r, r2, z, dn, tmpout);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_03, al_in, r, r2, z, dn, tmpout);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_10, al_in, r, r2, z, dn, tmpout);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_11, al_in, r, r2, z, dn, tmpout);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_12, al_in, r, r2, z, dn, tmpout);

			// c[1, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_13, al_in, r, r2, z, dn, tmpout);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_20, al_in, r, r2, z, dn, tmpout);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_21, al_in, r, r2, z, dn, tmpout);

			// c[2, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_22, al_in, r, r2, z, dn, tmpout);

			// c[2, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_23, al_in, r, r2, z, dn, tmpout);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_30, al_in, r, r2, z, dn, tmpout);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_31, al_in, r, r2, z, dn, tmpout);

			// c[3, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_32, al_in, r, r2, z, dn, tmpout);

			// c[3, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_33, al_in, r, r2, z, dn, tmpout);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_40, al_in, r, r2, z, dn, tmpout);

			// c[4, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_41, al_in, r, r2, z, dn, tmpout);

			// c[4, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_42, al_in, r, r2, z, dn, tmpout);

			// c[4, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_43, al_in, r, r2, z, dn, tmpout);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_50, al_in, r, r2, z, dn, tmpout);

			// c[5, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_51, al_in, r, r2, z, dn, tmpout);

			// c[5, 32-47]
			SIGMOID_F32_AVX512_DEF(acc_52, al_in, r, r2, z, dn, tmpout);

			// c[5, 48-63]
			SIGMOID_F32_AVX512_DEF(acc_53, al_in, r, r2, z, dn, tmpout);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_6x64_DISABLE:
		;

		// Case where the output C matrix is s8 (downscaled) and this is the
		// final write for a given block within C.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF );

			if ( post_ops_attr.c_stor_type == S8)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_S8(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_S8(acc_01,0,1);

				// c[0,32-47]
				CVT_STORE_F32_S8(acc_02,0,2);

				// c[0,48-63]
				CVT_STORE_F32_S8(acc_03,0,3);

				// c[1,0-15]
				CVT_STORE_F32_S8(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_S8(acc_11,1,1);

				// c[1,32-47]
				CVT_STORE_F32_S8(acc_12,1,2);

				// c[1,48-63]
				CVT_STORE_F32_S8(acc_13,1,3);

				// c[2,0-15]
				CVT_STORE_F32_S8(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_S8(acc_21,2,1);

				// c[2,32-47]
				CVT_STORE_F32_S8(acc_22,2,2);

				// c[2,48-63]
				CVT_STORE_F32_S8(acc_23,2,3);

				// c[3,0-15]
				CVT_STORE_F32_S8(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_S8(acc_31,3,1);

				// c[3,32-47]
				CVT_STORE_F32_S8(acc_32,3,2);

				// c[3,48-63]
				CVT_STORE_F32_S8(acc_33,3,3);

				// c[4,0-15]
				CVT_STORE_F32_S8(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_S8(acc_41,4,1);

				// c[4,32-47]
				CVT_STORE_F32_S8(acc_42,4,2);

				// c[4,48-63]
				CVT_STORE_F32_S8(acc_43,4,3);

				// c[5,0-15]
				CVT_STORE_F32_S8(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_S8(acc_51,5,1);

				// c[5,32-47]
				CVT_STORE_F32_S8(acc_52,5,2);

				// c[5,48-63]
				CVT_STORE_F32_S8(acc_53,5,3);
			}
			else if ( post_ops_attr.c_stor_type == U8)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_U8(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_U8(acc_01,0,1);

				// c[0,32-47]
				CVT_STORE_F32_U8(acc_02,0,2);

				// c[0,48-63]
				CVT_STORE_F32_U8(acc_03,0,3);

				// c[1,0-15]
				CVT_STORE_F32_U8(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_U8(acc_11,1,1);

				// c[1,32-47]
				CVT_STORE_F32_U8(acc_12,1,2);

				// c[1,48-63]
				CVT_STORE_F32_U8(acc_13,1,3);

				// c[2,0-15]
				CVT_STORE_F32_U8(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_U8(acc_21,2,1);

				// c[2,32-47]
				CVT_STORE_F32_U8(acc_22,2,2);

				// c[2,48-63]
				CVT_STORE_F32_U8(acc_23,2,3);

				// c[3,0-15]
				CVT_STORE_F32_U8(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_U8(acc_31,3,1);

				// c[3,32-47]
				CVT_STORE_F32_U8(acc_32,3,2);

				// c[3,48-63]
				CVT_STORE_F32_U8(acc_33,3,3);

				// c[4,0-15]
				CVT_STORE_F32_U8(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_U8(acc_41,4,1);

				// c[4,32-47]
				CVT_STORE_F32_U8(acc_42,4,2);

				// c[4,48-63]
				CVT_STORE_F32_U8(acc_43,4,3);

				// c[5,0-15]
				CVT_STORE_F32_U8(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_U8(acc_51,5,1);

				// c[5,32-47]
				CVT_STORE_F32_U8(acc_52,5,2);

				// c[5,48-63]
				CVT_STORE_F32_U8(acc_53,5,3);
			}
			else if ( post_ops_attr.c_stor_type == BF16)
			{
				// Store the results in downscaled type (bfloat16 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_BF16(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_BF16(acc_01,0,1);

				// c[0,32-47]
				CVT_STORE_F32_BF16(acc_02,0,2);

				// c[0,48-63]
				CVT_STORE_F32_BF16(acc_03,0,3);

				// c[1,0-15]
				CVT_STORE_F32_BF16(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_BF16(acc_11,1,1);

				// c[1,32-47]
				CVT_STORE_F32_BF16(acc_12,1,2);

				// c[1,48-63]
				CVT_STORE_F32_BF16(acc_13,1,3);

				// c[2,0-15]
				CVT_STORE_F32_BF16(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_BF16(acc_21,2,1);

				// c[2,32-47]
				CVT_STORE_F32_BF16(acc_22,2,2);

				// c[2,48-63]
				CVT_STORE_F32_BF16(acc_23,2,3);

				// c[3,0-15]
				CVT_STORE_F32_BF16(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_BF16(acc_31,3,1);

				// c[3,32-47]
				CVT_STORE_F32_BF16(acc_32,3,2);

				// c[3,48-63]
				CVT_STORE_F32_BF16(acc_33,3,3);

				// c[4,0-15]
				CVT_STORE_F32_BF16(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_BF16(acc_41,4,1);

				// c[4,32-47]
				CVT_STORE_F32_BF16(acc_42,4,2);

				// c[4,48-63]
				CVT_STORE_F32_BF16(acc_43,4,3);

				// c[5,0-15]
				CVT_STORE_F32_BF16(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_BF16(acc_51,5,1);

				// c[5,32-47]
				CVT_STORE_F32_BF16(acc_52,5,2);

				// c[5,48-63]
				CVT_STORE_F32_BF16(acc_53,5,3);
			}
		}
		// Case where the output C matrix is s32 or is the temp buffer used to
		// store intermediate s32 accumulated values for downscaled (C-s8) api.
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), acc_00 );

			// c[0, 16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), acc_01 );

			// c[0,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 2*16 ), acc_02 );

			// c[0,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 3*16 ), acc_03 );

			// c[1,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), acc_10 );

			// c[1,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), acc_11 );

			// c[1,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 2*16 ), acc_12 );

			// c[1,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 3*16 ), acc_13 );

			// c[2,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), acc_20 );

			// c[2,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), acc_21 );

			// c[2,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 2*16 ), acc_22 );

			// c[2,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 3*16 ), acc_23 );

			// c[3,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), acc_30 );

			// c[3,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), acc_31 );

			// c[3,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 2*16 ), acc_32 );

			// c[3,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 3*16 ), acc_33 );

			// c[4,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), acc_40 );

			// c[4,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), acc_41 );

			// c[4,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 2*16 ), acc_42 );

			// c[4,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 3*16 ), acc_43 );

			// c[5,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), acc_50 );

			// c[5,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), acc_51 );

			// c[5,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 2*16 ), acc_52 );

			// c[5,48-63]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 3*16 ), acc_53 );
		}

		a = a + ( MR * ps_a );
		grp_post_ops_attr.grp_post_op_i += MR;
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			// In cases where A matrix is packed cs_a is set to 24, since the
			// next column in a given row is accessed after 4*6 elements, where
			// 6 is MR and 4 elements are broadcasted each time from A (vnni).
			// In fringe case, where m < MR, the next column will be after m'*4
			// elements, and subsequently following adjustment of cs_a is
			// required before calling m fringe kernels.
			dim_t cs_a_use = ( cs_a == 4 ) ? 4 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_s8s8s32os32_5x64_sym_quant
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  grp_post_ops_attr ,post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			dim_t cs_a_use = ( cs_a == 4 ) ? 4 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_s8s8s32os32_4x64_sym_quant
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			dim_t cs_a_use = ( cs_a == 4 ) ? 4 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_s8s8s32os32_3x64_sym_quant
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			dim_t cs_a_use = ( cs_a == 4 ) ? 4 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_s8s8s32os32_2x64_sym_quant
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			dim_t cs_a_use = ( cs_a == 4 ) ? 4 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_s8s8s32os32_1x64_sym_quant
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  grp_post_ops_attr, post_ops_list, post_ops_attr
			);
		}
	}
}
#endif
