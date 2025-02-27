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

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32.h"

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_5x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_5x64_OPS,
						  &&POST_OPS_RELU_5x64_OPS,
						  &&POST_OPS_RELU_SCALE_5x64_OPS,
						  &&POST_OPS_GELU_TANH_5x64_OPS,
						  &&POST_OPS_GELU_ERF_5x64_OPS,
						  &&POST_OPS_CLIP_5x64_OPS,
						  &&POST_OPS_DOWNSCALE_5x64_OPS,
						  &&POST_OPS_MATRIX_ADD_5x64_OPS,
						  &&POST_OPS_SWISH_5x64_OPS,
						  &&POST_OPS_MATRIX_MUL_5x64_OPS,
						  &&POST_OPS_TANH_5x64_OPS,
						  &&POST_OPS_SIGMOID_5x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 zmm8 = _mm512_setzero_ps();
	__m512 zmm9 = _mm512_setzero_ps();
	__m512 zmm10 = _mm512_setzero_ps();
	__m512 zmm11 = _mm512_setzero_ps();

	__m512 zmm12 = _mm512_setzero_ps();
	__m512 zmm13 = _mm512_setzero_ps();
	__m512 zmm14 = _mm512_setzero_ps();
	__m512 zmm15 = _mm512_setzero_ps();

	__m512 zmm16 = _mm512_setzero_ps();
	__m512 zmm17 = _mm512_setzero_ps();
	__m512 zmm18 = _mm512_setzero_ps();
	__m512 zmm19 = _mm512_setzero_ps();

	__m512 zmm20 = _mm512_setzero_ps();
	__m512 zmm21 = _mm512_setzero_ps();
	__m512 zmm22 = _mm512_setzero_ps();
	__m512 zmm23 = _mm512_setzero_ps();

	__m512 zmm24 = _mm512_setzero_ps();
	__m512 zmm25 = _mm512_setzero_ps();
	__m512 zmm26 = _mm512_setzero_ps();
	__m512 zmm27 = _mm512_setzero_ps();

	__m512 zmm1 = _mm512_setzero_ps();
	__m512 zmm2 = _mm512_setzero_ps();
	__m512 zmm3 = _mm512_setzero_ps();
	__m512 zmm4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}
		// 1stx64 block.
		zmm8 = _mm512_maskz_loadu_ps( k0, a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm9 = _mm512_maskz_loadu_ps( k1, a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm10 = _mm512_maskz_loadu_ps( k2, a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm11 = _mm512_maskz_loadu_ps( k3, a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 2ndx64 block.
		zmm12 = _mm512_maskz_loadu_ps( k0, a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm13 = _mm512_maskz_loadu_ps( k1, a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm14 = _mm512_maskz_loadu_ps( k2, a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm15 = _mm512_maskz_loadu_ps( k3, a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 3rdx64 block.
		zmm16 = _mm512_maskz_loadu_ps( k0, a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm17 = _mm512_maskz_loadu_ps( k1, a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm18 = _mm512_maskz_loadu_ps( k2, a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm19 = _mm512_maskz_loadu_ps( k3, a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 4thx64 block.
		zmm20 = _mm512_maskz_loadu_ps( k0, a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm21 = _mm512_maskz_loadu_ps( k1, a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm22 = _mm512_maskz_loadu_ps( k2, a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm23 = _mm512_maskz_loadu_ps( k3, a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 5thx64 block.
		zmm24 = _mm512_maskz_loadu_ps( k0, a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm25 = _mm512_maskz_loadu_ps( k1, a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm26 = _mm512_maskz_loadu_ps( k2, a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm27 = _mm512_maskz_loadu_ps( k3, a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 48 ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x64_OPS:
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						BF16_F32_BIAS_LOAD(zmm1, k0, 0);
						BF16_F32_BIAS_LOAD(zmm2, k1, 1);
						BF16_F32_BIAS_LOAD(zmm3, k2, 2);
						BF16_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						S32_F32_BIAS_LOAD(zmm1, k0, 0);
						S32_F32_BIAS_LOAD(zmm2, k1, 1);
						S32_F32_BIAS_LOAD(zmm3, k2, 2);
						S32_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						S8_F32_BIAS_LOAD(zmm1, k0, 0);
						S8_F32_BIAS_LOAD(zmm2, k1, 1);
						S8_F32_BIAS_LOAD(zmm3, k2, 2);
						S8_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else
					{
						zmm1 =_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
						zmm2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
						zmm3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
						zmm4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm2, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm3, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm4, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm1, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm3, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm4, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm1, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm2, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm4, zmm19 );

					// c[3,0-15]
					zmm20 = _mm512_add_ps( zmm1, zmm20 );

					// c[3, 16-31]
					zmm21 = _mm512_add_ps( zmm2, zmm21 );

					// c[3,32-47]
					zmm22 = _mm512_add_ps( zmm3, zmm22 );

					// c[3,48-63]
					zmm23 = _mm512_add_ps( zmm4, zmm23 );

					// c[4,0-15]
					zmm24 = _mm512_add_ps( zmm1, zmm24 );

					// c[4, 16-31]
					zmm25 = _mm512_add_ps( zmm2, zmm25 );

					// c[4,32-47]
					zmm26 = _mm512_add_ps( zmm3, zmm26 );

					// c[4,48-63]
					zmm27 = _mm512_add_ps( zmm4, zmm27 );
				}
				else
				{
					// If original output was columns major, then by the time
					// kernel sees it, the matrix would be accessed as if it were
					// transposed. Due to this the bias array will be accessed by
					// the ic index, and each bias element corresponds to an
					// entire row of the transposed output array, instead of an
					// entire column.
					__m512 selector5;
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3);
						BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S32_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S32_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S32_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						S32_F32_BIAS_BCAST(zmm4, bias_mask, 3);
						S32_F32_BIAS_BCAST(selector5, bias_mask, 4);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S8_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S8_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S8_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						S8_F32_BIAS_BCAST(zmm4, bias_mask, 3);
						S8_F32_BIAS_BCAST(selector5, bias_mask, 4);
					}
					else
					{
						zmm1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 0 ) );
						zmm2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 1 ) );
						zmm3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 2 ) );
						zmm4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 3 ) );
						selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 4 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm1, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm1, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm1, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm2, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm2, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm2, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm3, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm3, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm3, zmm19 );

					// c[3,0-15]
					zmm20 = _mm512_add_ps( zmm4, zmm20 );

					// c[3, 16-31]
					zmm21 = _mm512_add_ps( zmm4, zmm21 );

					// c[3,32-47]
					zmm22 = _mm512_add_ps( zmm4, zmm22 );

					// c[3,48-63]
					zmm23 = _mm512_add_ps( zmm4, zmm23 );

					// c[4,0-15]
					zmm24 = _mm512_add_ps( selector5, zmm24 );

					// c[4, 16-31]
					zmm25 = _mm512_add_ps( selector5, zmm25 );

					// c[4,32-47]
					zmm26 = _mm512_add_ps( selector5, zmm26 );

					// c[4,48-63]
					zmm27 = _mm512_add_ps( selector5, zmm27 );
				}

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_RELU_5x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();

			// c[0,0-15]
			zmm8 = _mm512_max_ps( zmm1, zmm8 );

			// c[0, 16-31]
			zmm9 = _mm512_max_ps( zmm1, zmm9 );

			// c[0,32-47]
			zmm10 = _mm512_max_ps( zmm1, zmm10 );

			// c[0,48-63]
			zmm11 = _mm512_max_ps( zmm1, zmm11 );

			// c[1,0-15]
			zmm12 = _mm512_max_ps( zmm1, zmm12 );

			// c[1,16-31]
			zmm13 = _mm512_max_ps( zmm1, zmm13 );

			// c[1,32-47]
			zmm14 = _mm512_max_ps( zmm1, zmm14 );

			// c[1,48-63]
			zmm15 = _mm512_max_ps( zmm1, zmm15 );

			// c[2,0-15]
			zmm16 = _mm512_max_ps( zmm1, zmm16 );

			// c[2,16-31]
			zmm17 = _mm512_max_ps( zmm1, zmm17 );

			// c[2,32-47]
			zmm18 = _mm512_max_ps( zmm1, zmm18 );

			// c[2,48-63]
			zmm19 = _mm512_max_ps( zmm1, zmm19 );

			// c[3,0-15]
			zmm20 = _mm512_max_ps( zmm1, zmm20 );

			// c[3,16-31]
			zmm21 = _mm512_max_ps( zmm1, zmm21 );

			// c[3,32-47]
			zmm22 = _mm512_max_ps( zmm1, zmm22 );

			// c[3,48-63]
			zmm23 = _mm512_max_ps( zmm1, zmm23 );

			// c[4,0-15]
			zmm24 = _mm512_max_ps( zmm1, zmm24 );

			// c[4,16-31]
			zmm25 = _mm512_max_ps( zmm1, zmm25 );

			// c[4,32-47]
			zmm26 = _mm512_max_ps( zmm1, zmm26 );

			// c[4,48-63]
			zmm27 = _mm512_max_ps( zmm1, zmm27 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_5x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();
			zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm8)

			// c[0, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm9)

			// c[0, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm10)

			// c[0, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm11)

			// c[1, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm12)

			// c[1, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm13)

			// c[1, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm14)

			// c[1, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm15)

			// c[2, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm16)

			// c[2, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm17)

			// c[2, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm18)

			// c[2, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm19)

			// c[3, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm20)

			// c[3, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm21)

			// c[3, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm22)

			// c[3, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm23)

			// c[4, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm24)

			// c[4, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm25)

			// c[4, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm26)

			// c[4, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm27)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_5x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32S_AVX512(zmm20, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32S_AVX512(zmm21, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32S_AVX512(zmm22, r, r2, x, z, dn, x_tanh, q)

			// c[3, 48-63]
			GELU_TANH_F32S_AVX512(zmm23, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32S_AVX512(zmm24, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_F32S_AVX512(zmm25, r, r2, x, z, dn, x_tanh, q)

			// c[4, 32-47]
			GELU_TANH_F32S_AVX512(zmm26, r, r2, x, z, dn, x_tanh, q)

			// c[4, 48-63]
			GELU_TANH_F32S_AVX512(zmm27, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_5x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32S_AVX512(zmm8, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32S_AVX512(zmm9, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32S_AVX512(zmm10, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32S_AVX512(zmm11, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32S_AVX512(zmm12, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32S_AVX512(zmm13, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32S_AVX512(zmm14, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32S_AVX512(zmm15, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32S_AVX512(zmm16, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32S_AVX512(zmm17, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32S_AVX512(zmm18, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32S_AVX512(zmm19, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32S_AVX512(zmm20, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32S_AVX512(zmm21, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32S_AVX512(zmm22, r, x, x_erf)

			// c[3, 48-63]
			GELU_ERF_F32S_AVX512(zmm23, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32S_AVX512(zmm24, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_F32S_AVX512(zmm25, r, x, x_erf)

			// c[4, 32-47]
			GELU_ERF_F32S_AVX512(zmm26, r, x, x_erf)

			// c[4, 48-63]
			GELU_ERF_F32S_AVX512(zmm27, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_5x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32S_AVX512(zmm8, min, max)

			// c[0, 16-31]
			CLIP_F32S_AVX512(zmm9, min, max)

			// c[0, 32-47]
			CLIP_F32S_AVX512(zmm10, min, max)

			// c[0, 48-63]
			CLIP_F32S_AVX512(zmm11, min, max)

			// c[1, 0-15]
			CLIP_F32S_AVX512(zmm12, min, max)

			// c[1, 16-31]
			CLIP_F32S_AVX512(zmm13, min, max)

			// c[1, 32-47]
			CLIP_F32S_AVX512(zmm14, min, max)

			// c[1, 48-63]
			CLIP_F32S_AVX512(zmm15, min, max)

			// c[2, 0-15]
			CLIP_F32S_AVX512(zmm16, min, max)

			// c[2, 16-31]
			CLIP_F32S_AVX512(zmm17, min, max)

			// c[2, 32-47]
			CLIP_F32S_AVX512(zmm18, min, max)

			// c[2, 48-63]
			CLIP_F32S_AVX512(zmm19, min, max)

			// c[3, 0-15]
			CLIP_F32S_AVX512(zmm20, min, max)

			// c[3, 16-31]
			CLIP_F32S_AVX512(zmm21, min, max)

			// c[3, 32-47]
			CLIP_F32S_AVX512(zmm22, min, max)

			// c[3, 48-63]
			CLIP_F32S_AVX512(zmm23, min, max)

			// c[4, 0-15]
			CLIP_F32S_AVX512(zmm24, min, max)

			// c[4, 16-31]
			CLIP_F32S_AVX512(zmm25, min, max)

			// c[4, 32-47]
			CLIP_F32S_AVX512(zmm26, min, max)

			// c[4, 48-63]
			CLIP_F32S_AVX512(zmm27, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_5x64_OPS:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( post_ops_list_temp->zp_stor_type == BF16 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S32 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S32_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
		else if ( post_ops_list_temp->zp_stor_type == U8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          U8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            BF16_F32_ZP_LOAD(zero_point0, k0, 0);
            BF16_F32_ZP_LOAD(zero_point1, k1, 1);
            BF16_F32_ZP_LOAD(zero_point2, k2, 2);
            BF16_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            S32_F32_ZP_LOAD(zero_point0, k0, 0);
            S32_F32_ZP_LOAD(zero_point1, k1, 1);
            S32_F32_ZP_LOAD(zero_point2, k2, 2);
            S32_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            S8_F32_ZP_LOAD(zero_point0, k0, 0);
            S8_F32_ZP_LOAD(zero_point1, k1, 1);
            S8_F32_ZP_LOAD(zero_point2, k2, 2);
            S8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            U8_F32_ZP_LOAD(zero_point0, k0, 0);
            U8_F32_ZP_LOAD(zero_point1, k1, 1);
            U8_F32_ZP_LOAD(zero_point2, k2, 2);
            U8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps( k0, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_maskz_loadu_ps( k1, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_maskz_loadu_ps( k2, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_maskz_loadu_ps( k3, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector2, zero_point1);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector3, zero_point2);

        //c[4, 48-63]
        F32_SCL_MULRND(zmm27, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            BF16_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            BF16_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            BF16_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S32_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S32_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            S32_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            S8_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            U8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            U8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            U8_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);

        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector1, zero_point0);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector1, zero_point0);

        //c[4, 48-63]
        F32_SCL_MULRND(zmm27, selector1, zero_point0);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
				bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
				bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
				scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27, \
					scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,zmm1,zmm2,zmm3,zmm4,4);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_5x64_OPS:
		{
			zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(zmm8, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(zmm9, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(zmm10, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(zmm11, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(zmm12, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(zmm13, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(zmm14, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(zmm15, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(zmm16, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(zmm17, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(zmm18, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(zmm19, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(zmm20, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(zmm21, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(zmm22, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SWISH_F32_AVX512_DEF(zmm23, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(zmm24, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(zmm25, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[4, 32-47]
			SWISH_F32_AVX512_DEF(zmm26, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[4, 48-63]
			SWISH_F32_AVX512_DEF(zmm27, zmm1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_5x64_OPS:
			{
				__m512 dn, z, x, r2, r;
				__m512i q;

				// c[0, 0-15]
				TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, q)

				// c[0, 16-31]
				TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, q)

				// c[0, 32-47]
				TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, q)

				// c[0, 48-63]
				TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, q)

				// c[1, 0-15]
				TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, q)

				// c[1, 16-31]
				TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, q)

				// c[1, 32-47]
				TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, q)

				// c[1, 48-63]
				TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, q)

				// c[2, 0-15]
				TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, q)

				// c[2, 16-31]
				TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, q)

				// c[2, 32-47]
				TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, q)

				// c[2, 48-63]
				TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, q)

				// c[3, 0-15]
				TANH_F32S_AVX512(zmm20, r, r2, x, z, dn, q)

				// c[3, 16-31]
				TANH_F32S_AVX512(zmm21, r, r2, x, z, dn, q)

				// c[3, 32-47]
				TANH_F32S_AVX512(zmm22, r, r2, x, z, dn, q)

				// c[3, 48-63]
				TANH_F32S_AVX512(zmm23, r, r2, x, z, dn, q)

				// c[4, 0-15]
				TANH_F32S_AVX512(zmm24, r, r2, x, z, dn, q)

				// c[4, 16-31]
				TANH_F32S_AVX512(zmm25, r, r2, x, z, dn, q)

				// c[4, 32-47]
				TANH_F32S_AVX512(zmm26, r, r2, x, z, dn, q)

				// c[4, 48-63]
				TANH_F32S_AVX512(zmm27, r, r2, x, z, dn, q)

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_SIGMOID_5x64_OPS:
			{
				__m512 al_in, r, r2, z, dn;
				__m512i ex_out;

				// c[0, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);

				// c[0, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm9, al_in, r, r2, z, dn, ex_out);

				// c[0, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm10, al_in, r, r2, z, dn, ex_out);

				// c[0, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm11, al_in, r, r2, z, dn, ex_out);

				// c[1, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm12, al_in, r, r2, z, dn, ex_out);

				// c[1, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm13, al_in, r, r2, z, dn, ex_out);

				// c[1, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm14, al_in, r, r2, z, dn, ex_out);

				// c[1, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm15, al_in, r, r2, z, dn, ex_out);

				// c[2, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm16, al_in, r, r2, z, dn, ex_out);

				// c[2, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm17, al_in, r, r2, z, dn, ex_out);

				// c[2, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm18, al_in, r, r2, z, dn, ex_out);

				// c[2, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm19, al_in, r, r2, z, dn, ex_out);

				// c[3, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm20, al_in, r, r2, z, dn, ex_out);

				// c[3, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm21, al_in, r, r2, z, dn, ex_out);

				// c[3, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm22, al_in, r, r2, z, dn, ex_out);

				// c[3, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm23, al_in, r, r2, z, dn, ex_out);

				// c[4, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm24, al_in, r, r2, z, dn, ex_out);

				// c[4, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm25, al_in, r, r2, z, dn, ex_out);

				// c[4, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm26, al_in, r, r2, z, dn, ex_out);

				// c[4, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm27, al_in, r, r2, z, dn, ex_out);

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_5x64_OPS_DISABLE:
			;

			// Case where the output C matrix is bf16 (downscaled) and this is the
			// final write for a given block within C.
			if ( post_ops_attr.c_stor_type == BF16 )
			{

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm23,k3,3,48);

				// c[4, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm24,k0,4,0);
				// c[4, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm25,k1,4,16);
				// c[4, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm26,k2,4,32);
				// c[4, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm27,k3,4,48);
			}
			else if ( post_ops_attr.c_stor_type == S32 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int32_t* b_q = ( int32_t* )b;
				dim_t ir = 0;
				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm23,k3,3,48);

				// c[4, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm24,k0,4,0);
				// c[4, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm25,k1,4,16);
				// c[4, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm26,k2,4,32);
				// c[4, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm27,k3,4,48);
			}
			else if ( post_ops_attr.c_stor_type == S8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int8_t* b_q = ( int8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm23,k3,3,48);

				// c[4, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm24,k0,4,0);
				// c[4, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm25,k1,4,16);
				// c[4, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm26,k2,4,32);
				// c[4, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm27,k3,4,48);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				uint8_t* b_q = ( uint8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm23,k3,3,48);

				// c[4, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm24,k0,4,0);
				// c[4, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm25,k1,4,16);
				// c[4, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm26,k2,4,32);
				// c[4, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm27,k3,4,48);
			}
			else // Case where the output C matrix is float
			{
				// Store the results.
				// c[0,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm8 );
				// c[0,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm9 );
				// c[0,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm10 );
				// c[0,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm11 );

				// c[1,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm12 );
				// c[1,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm13 );
				// c[1,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm14 );
				// c[1,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm15 );

				// c[2,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm16 );
				// c[2,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm17 );
				// c[2,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm18 );
				// c[2,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm19 );

				// c[3,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm20 );
				// c[3,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm21 );
				// c[3,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm22 );
				// c[3,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm23 );

				// c[4,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm24 );
				// c[4,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm25 );
				// c[4,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm26 );
				// c[4,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm27 );
			}
            post_ops_attr.post_op_c_j += NR_L;
		}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_4x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_4x64_OPS,
						  &&POST_OPS_RELU_4x64_OPS,
						  &&POST_OPS_RELU_SCALE_4x64_OPS,
						  &&POST_OPS_GELU_TANH_4x64_OPS,
						  &&POST_OPS_GELU_ERF_4x64_OPS,
						  &&POST_OPS_CLIP_4x64_OPS,
						  &&POST_OPS_DOWNSCALE_4x64_OPS,
						  &&POST_OPS_MATRIX_ADD_4x64_OPS,
						  &&POST_OPS_SWISH_4x64_OPS,
						  &&POST_OPS_MATRIX_MUL_4x64_OPS,
						  &&POST_OPS_TANH_4x64_OPS,
						  &&POST_OPS_SIGMOID_4x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 zmm8 = _mm512_setzero_ps();
	__m512 zmm9 = _mm512_setzero_ps();
	__m512 zmm10 = _mm512_setzero_ps();
	__m512 zmm11 = _mm512_setzero_ps();

	__m512 zmm12 = _mm512_setzero_ps();
	__m512 zmm13 = _mm512_setzero_ps();
	__m512 zmm14 = _mm512_setzero_ps();
	__m512 zmm15 = _mm512_setzero_ps();

	__m512 zmm16 = _mm512_setzero_ps();
	__m512 zmm17 = _mm512_setzero_ps();
	__m512 zmm18 = _mm512_setzero_ps();
	__m512 zmm19 = _mm512_setzero_ps();

	__m512 zmm20 = _mm512_setzero_ps();
	__m512 zmm21 = _mm512_setzero_ps();
	__m512 zmm22 = _mm512_setzero_ps();
	__m512 zmm23 = _mm512_setzero_ps();

	__m512 zmm1 = _mm512_setzero_ps();
	__m512 zmm2 = _mm512_setzero_ps();
	__m512 zmm3 = _mm512_setzero_ps();
	__m512 zmm4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		zmm8 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm9 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm10 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm11 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 2ndx64 block.
		zmm12 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm13 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm14 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm15 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 3rdx64 block.
		zmm16 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm17 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm18 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm19 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 4thx64 block.
		zmm20 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm21 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm22 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm23 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 48 ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x64_OPS:
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						BF16_F32_BIAS_LOAD(zmm1, k0, 0);
						BF16_F32_BIAS_LOAD(zmm2, k1, 1);
						BF16_F32_BIAS_LOAD(zmm3, k2, 2);
						BF16_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						S32_F32_BIAS_LOAD(zmm1, k0, 0);
						S32_F32_BIAS_LOAD(zmm2, k1, 1);
						S32_F32_BIAS_LOAD(zmm3, k2, 2);
						S32_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						S8_F32_BIAS_LOAD(zmm1, k0, 0);
						S8_F32_BIAS_LOAD(zmm2, k1, 1);
						S8_F32_BIAS_LOAD(zmm3, k2, 2);
						S8_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else
					{
						zmm1 =_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
						zmm2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
						zmm3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
						zmm4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm2, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm3, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm4, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm1, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm3, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm4, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm1, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm2, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm4, zmm19 );

					// c[3,0-15]
					zmm20 = _mm512_add_ps( zmm1, zmm20 );

					// c[3, 16-31]
					zmm21 = _mm512_add_ps( zmm2, zmm21 );

					// c[3,32-47]
					zmm22 = _mm512_add_ps( zmm3, zmm22 );

					// c[3,48-63]
					zmm23 = _mm512_add_ps( zmm4, zmm23 );
				}
				else
				{
					// If original output was columns major, then by the time
					// kernel sees it, the matrix would be accessed as if it were
					// transposed. Due to this the bias array will be accessed by
					// the ic index, and each bias element corresponds to an
					// entire row of the transposed output array, instead of an
					// entire column.
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S32_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S32_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S32_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						S32_F32_BIAS_BCAST(zmm4, bias_mask, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S8_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S8_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S8_F32_BIAS_BCAST(zmm3, bias_mask, 2);
						S8_F32_BIAS_BCAST(zmm4, bias_mask, 3);
					}
					else
					{
						zmm1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 0 ) );
						zmm2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 1 ) );
						zmm3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 2 ) );
						zmm4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 3 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm1, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm1, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm1, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm2, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm2, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm2, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm3, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm3, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm3, zmm19 );

					// c[3,0-15]
					zmm20 = _mm512_add_ps( zmm4, zmm20 );

					// c[3, 16-31]
					zmm21 = _mm512_add_ps( zmm4, zmm21 );

					// c[3,32-47]
					zmm22 = _mm512_add_ps( zmm4, zmm22 );

					// c[3,48-63]
					zmm23 = _mm512_add_ps( zmm4, zmm23 );
				}

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_RELU_4x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();

			// c[0,0-15]
			zmm8 = _mm512_max_ps( zmm1, zmm8 );

			// c[0, 16-31]
			zmm9 = _mm512_max_ps( zmm1, zmm9 );

			// c[0,32-47]
			zmm10 = _mm512_max_ps( zmm1, zmm10 );

			// c[0,48-63]
			zmm11 = _mm512_max_ps( zmm1, zmm11 );

			// c[1,0-15]
			zmm12 = _mm512_max_ps( zmm1, zmm12 );

			// c[1,16-31]
			zmm13 = _mm512_max_ps( zmm1, zmm13 );

			// c[1,32-47]
			zmm14 = _mm512_max_ps( zmm1, zmm14 );

			// c[1,48-63]
			zmm15 = _mm512_max_ps( zmm1, zmm15 );

			// c[2,0-15]
			zmm16 = _mm512_max_ps( zmm1, zmm16 );

			// c[2,16-31]
			zmm17 = _mm512_max_ps( zmm1, zmm17 );

			// c[2,32-47]
			zmm18 = _mm512_max_ps( zmm1, zmm18 );

			// c[2,48-63]
			zmm19 = _mm512_max_ps( zmm1, zmm19 );

			// c[3,0-15]
			zmm20 = _mm512_max_ps( zmm1, zmm20 );

			// c[3,16-31]
			zmm21 = _mm512_max_ps( zmm1, zmm21 );

			// c[3,32-47]
			zmm22 = _mm512_max_ps( zmm1, zmm22 );

			// c[3,48-63]
			zmm23 = _mm512_max_ps( zmm1, zmm23 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_4x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();
			zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm8)

			// c[0, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm9)

			// c[0, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm10)

			// c[0, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm11)

			// c[1, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm12)

			// c[1, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm13)

			// c[1, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm14)

			// c[1, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm15)

			// c[2, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm16)

			// c[2, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm17)

			// c[2, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm18)

			// c[2, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm19)

			// c[3, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm20)

			// c[3, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm21)

			// c[3, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm22)

			// c[3, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm23)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_4x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32S_AVX512(zmm20, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32S_AVX512(zmm21, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32S_AVX512(zmm22, r, r2, x, z, dn, x_tanh, q)

			// c[3, 48-63]
			GELU_TANH_F32S_AVX512(zmm23, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_4x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32S_AVX512(zmm8, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32S_AVX512(zmm9, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32S_AVX512(zmm10, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32S_AVX512(zmm11, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32S_AVX512(zmm12, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32S_AVX512(zmm13, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32S_AVX512(zmm14, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32S_AVX512(zmm15, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32S_AVX512(zmm16, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32S_AVX512(zmm17, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32S_AVX512(zmm18, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32S_AVX512(zmm19, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32S_AVX512(zmm20, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32S_AVX512(zmm21, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32S_AVX512(zmm22, r, x, x_erf)

			// c[3, 48-63]
			GELU_ERF_F32S_AVX512(zmm23, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_4x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32S_AVX512(zmm8, min, max)

			// c[0, 16-31]
			CLIP_F32S_AVX512(zmm9, min, max)

			// c[0, 32-47]
			CLIP_F32S_AVX512(zmm10, min, max)

			// c[0, 48-63]
			CLIP_F32S_AVX512(zmm11, min, max)

			// c[1, 0-15]
			CLIP_F32S_AVX512(zmm12, min, max)

			// c[1, 16-31]
			CLIP_F32S_AVX512(zmm13, min, max)

			// c[1, 32-47]
			CLIP_F32S_AVX512(zmm14, min, max)

			// c[1, 48-63]
			CLIP_F32S_AVX512(zmm15, min, max)

			// c[2, 0-15]
			CLIP_F32S_AVX512(zmm16, min, max)

			// c[2, 16-31]
			CLIP_F32S_AVX512(zmm17, min, max)

			// c[2, 32-47]
			CLIP_F32S_AVX512(zmm18, min, max)

			// c[2, 48-63]
			CLIP_F32S_AVX512(zmm19, min, max)

			// c[3, 0-15]
			CLIP_F32S_AVX512(zmm20, min, max)

			// c[3, 16-31]
			CLIP_F32S_AVX512(zmm21, min, max)

			// c[3, 32-47]
			CLIP_F32S_AVX512(zmm22, min, max)

			// c[3, 48-63]
			CLIP_F32S_AVX512(zmm23, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_4x64_OPS:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( post_ops_list_temp->zp_stor_type == BF16 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S32 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S32_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
		else if ( post_ops_list_temp->zp_stor_type == U8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          U8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            BF16_F32_ZP_LOAD(zero_point0, k0, 0);
            BF16_F32_ZP_LOAD(zero_point1, k1, 1);
            BF16_F32_ZP_LOAD(zero_point2, k2, 2);
            BF16_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            S32_F32_ZP_LOAD(zero_point0, k0, 0);
            S32_F32_ZP_LOAD(zero_point1, k1, 1);
            S32_F32_ZP_LOAD(zero_point2, k2, 2);
            S32_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            S8_F32_ZP_LOAD(zero_point0, k0, 0);
            S8_F32_ZP_LOAD(zero_point1, k1, 1);
            S8_F32_ZP_LOAD(zero_point2, k2, 2);
            S8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            U8_F32_ZP_LOAD(zero_point0, k0, 0);
            U8_F32_ZP_LOAD(zero_point1, k1, 1);
            U8_F32_ZP_LOAD(zero_point2, k2, 2);
            U8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps( k0, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_maskz_loadu_ps( k1, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_maskz_loadu_ps( k2, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_maskz_loadu_ps( k3, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            BF16_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            BF16_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            BF16_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S32_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S32_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            S32_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            S8_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            U8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            U8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
            U8_F32_ZP_BCAST(zero_point3, zp_mask, 3);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
				bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
				bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
				scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23, \
					scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,zmm1,zmm2,zmm3,zmm4,3);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_4x64_OPS:
		{
			zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(zmm8, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(zmm9, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(zmm10, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(zmm11, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(zmm12, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(zmm13, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(zmm14, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(zmm15, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(zmm16, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(zmm17, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(zmm18, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(zmm19, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(zmm20, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(zmm21, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(zmm22, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SWISH_F32_AVX512_DEF(zmm23, zmm1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_4x64_OPS:
			{
				__m512 dn, z, x, r2, r;
				__m512i q;

				// c[0, 0-15]
				TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, q)

				// c[0, 16-31]
				TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, q)

				// c[0, 32-47]
				TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, q)

				// c[0, 48-63]
				TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, q)

				// c[1, 0-15]
				TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, q)

				// c[1, 16-31]
				TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, q)

				// c[1, 32-47]
				TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, q)

				// c[1, 48-63]
				TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, q)

				// c[2, 0-15]
				TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, q)

				// c[2, 16-31]
				TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, q)

				// c[2, 32-47]
				TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, q)

				// c[2, 48-63]
				TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, q)

				// c[3, 0-15]
				TANH_F32S_AVX512(zmm20, r, r2, x, z, dn, q)

				// c[3, 16-31]
				TANH_F32S_AVX512(zmm21, r, r2, x, z, dn, q)

				// c[3, 32-47]
				TANH_F32S_AVX512(zmm22, r, r2, x, z, dn, q)

				// c[3, 48-63]
				TANH_F32S_AVX512(zmm23, r, r2, x, z, dn, q)

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_SIGMOID_4x64_OPS:
			{
				__m512 al_in, r, r2, z, dn;
				__m512i ex_out;

				// c[0, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);

				// c[0, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm9, al_in, r, r2, z, dn, ex_out);

				// c[0, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm10, al_in, r, r2, z, dn, ex_out);

				// c[0, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm11, al_in, r, r2, z, dn, ex_out);

				// c[1, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm12, al_in, r, r2, z, dn, ex_out);

				// c[1, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm13, al_in, r, r2, z, dn, ex_out);

				// c[1, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm14, al_in, r, r2, z, dn, ex_out);

				// c[1, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm15, al_in, r, r2, z, dn, ex_out);

				// c[2, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm16, al_in, r, r2, z, dn, ex_out);

				// c[2, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm17, al_in, r, r2, z, dn, ex_out);

				// c[2, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm18, al_in, r, r2, z, dn, ex_out);

				// c[2, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm19, al_in, r, r2, z, dn, ex_out);

				// c[3, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm20, al_in, r, r2, z, dn, ex_out);

				// c[3, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm21, al_in, r, r2, z, dn, ex_out);

				// c[3, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm22, al_in, r, r2, z, dn, ex_out);

				// c[3, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm23, al_in, r, r2, z, dn, ex_out);

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_4x64_OPS_DISABLE:
			;

			// Case where the output C matrix is bf16 (downscaled) and this is the
			// final write for a given block within C.
			if ( post_ops_attr.c_stor_type == BF16 )
			{

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm23,k3,3,48);
			}
			else if ( post_ops_attr.c_stor_type == S32 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int32_t* b_q = ( int32_t* )b;
				dim_t ir = 0;
				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm23,k3,3,48);
			}
			else if ( post_ops_attr.c_stor_type == S8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int8_t* b_q = ( int8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm23,k3,3,48);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				uint8_t* b_q = ( uint8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm19,k3,2,48);

				// c[3, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm20,k0,3,0);
				// c[3, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm21,k1,3,16);
				// c[3, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm22,k2,3,32);
				// c[3, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm23,k3,3,48);
			}
			else // Case where the output C matrix is float
			{
				// Store the results.
				// c[0,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm8 );
				// c[0,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm9 );
				// c[0,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm10 );
				// c[0,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm11 );

				// c[1,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm12 );
				// c[1,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm13 );
				// c[1,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm14 );
				// c[1,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm15 );

				// c[2,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm16 );
				// c[2,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm17 );
				// c[2,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm18 );
				// c[2,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm19 );

				// c[3,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm20 );
				// c[3,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm21 );
				// c[3,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm22 );
				// c[3,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm23 );
			}
            post_ops_attr.post_op_c_j += NR_L;
		}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_3x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_3x64_OPS,
						  &&POST_OPS_RELU_3x64_OPS,
						  &&POST_OPS_RELU_SCALE_3x64_OPS,
						  &&POST_OPS_GELU_TANH_3x64_OPS,
						  &&POST_OPS_GELU_ERF_3x64_OPS,
						  &&POST_OPS_CLIP_3x64_OPS,
						  &&POST_OPS_DOWNSCALE_3x64_OPS,
						  &&POST_OPS_MATRIX_ADD_3x64_OPS,
						  &&POST_OPS_SWISH_3x64_OPS,
						  &&POST_OPS_MATRIX_MUL_3x64_OPS,
						  &&POST_OPS_TANH_3x64_OPS,
						  &&POST_OPS_SIGMOID_3x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 zmm8 = _mm512_setzero_ps();
	__m512 zmm9 = _mm512_setzero_ps();
	__m512 zmm10 = _mm512_setzero_ps();
	__m512 zmm11 = _mm512_setzero_ps();

	__m512 zmm12 = _mm512_setzero_ps();
	__m512 zmm13 = _mm512_setzero_ps();
	__m512 zmm14 = _mm512_setzero_ps();
	__m512 zmm15 = _mm512_setzero_ps();

	__m512 zmm16 = _mm512_setzero_ps();
	__m512 zmm17 = _mm512_setzero_ps();
	__m512 zmm18 = _mm512_setzero_ps();
	__m512 zmm19 = _mm512_setzero_ps();

	__m512 zmm1 = _mm512_setzero_ps();
	__m512 zmm2 = _mm512_setzero_ps();
	__m512 zmm3 = _mm512_setzero_ps();
	__m512 zmm4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		zmm8 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm9 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) ;
		zmm10 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) ;
		zmm11 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) ;

		// 2ndx64 block.
		zmm12 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm13 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) ) ;
		zmm14 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) ) ;
		zmm15 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) ) ;

		// 3rdx64 block.
		zmm16 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm17 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) ) ;
		zmm18 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) ) ;
		zmm19 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) ) ;

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x64_OPS:
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						BF16_F32_BIAS_LOAD(zmm1, k0, 0);
						BF16_F32_BIAS_LOAD(zmm2, k1, 1);
						BF16_F32_BIAS_LOAD(zmm3, k2, 2);
						BF16_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						S32_F32_BIAS_LOAD(zmm1, k0, 0);
						S32_F32_BIAS_LOAD(zmm2, k1, 1);
						S32_F32_BIAS_LOAD(zmm3, k2, 2);
						S32_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						S8_F32_BIAS_LOAD(zmm1, k0, 0);
						S8_F32_BIAS_LOAD(zmm2, k1, 1);
						S8_F32_BIAS_LOAD(zmm3, k2, 2);
						S8_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else
					{
						zmm1 =_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
						zmm2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
						zmm3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
						zmm4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm2, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm3, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm4, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm1, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm3, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm4, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm1, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm2, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm4, zmm19 );
				}
				else
				{
					// If original output was columns major, then by the time
					// kernel sees it, the matrix would be accessed as if it were
					// transposed. Due to this the bias array will be accessed by
					// the ic index, and each bias element corresponds to an
					// entire row of the transposed output array, instead of an
					// entire column.
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S32_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S32_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S32_F32_BIAS_BCAST(zmm3, bias_mask, 2);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S8_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S8_F32_BIAS_BCAST(zmm2, bias_mask, 1);
						S8_F32_BIAS_BCAST(zmm3, bias_mask, 2);
					}
					else
					{
						zmm1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 0 ) );
						zmm2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 1 ) );
						zmm3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 2 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm1, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm1, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm1, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm2, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm2, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm2, zmm15 );

					// c[2,0-15]
					zmm16 = _mm512_add_ps( zmm3, zmm16 );

					// c[2, 16-31]
					zmm17 = _mm512_add_ps( zmm3, zmm17 );

					// c[2,32-47]
					zmm18 = _mm512_add_ps( zmm3, zmm18 );

					// c[2,48-63]
					zmm19 = _mm512_add_ps( zmm3, zmm19 );
				}

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_RELU_3x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();

			// c[0,0-15]
			zmm8 = _mm512_max_ps( zmm1, zmm8 );

			// c[0, 16-31]
			zmm9 = _mm512_max_ps( zmm1, zmm9 );

			// c[0,32-47]
			zmm10 = _mm512_max_ps( zmm1, zmm10 );

			// c[0,48-63]
			zmm11 = _mm512_max_ps( zmm1, zmm11 );

			// c[1,0-15]
			zmm12 = _mm512_max_ps( zmm1, zmm12 );

			// c[1,16-31]
			zmm13 = _mm512_max_ps( zmm1, zmm13 );

			// c[1,32-47]
			zmm14 = _mm512_max_ps( zmm1, zmm14 );

			// c[1,48-63]
			zmm15 = _mm512_max_ps( zmm1, zmm15 );

			// c[2,0-15]
			zmm16 = _mm512_max_ps( zmm1, zmm16 );

			// c[2,16-31]
			zmm17 = _mm512_max_ps( zmm1, zmm17 );

			// c[2,32-47]
			zmm18 = _mm512_max_ps( zmm1, zmm18 );

			// c[2,48-63]
			zmm19 = _mm512_max_ps( zmm1, zmm19 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_3x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();
			zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm8)

			// c[0, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm9)

			// c[0, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm10)

			// c[0, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm11)

			// c[1, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm12)

			// c[1, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm13)

			// c[1, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm14)

			// c[1, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm15)

			// c[2, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm16)

			// c[2, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm17)

			// c[2, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm18)

			// c[2, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm19)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_3x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_3x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32S_AVX512(zmm8, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32S_AVX512(zmm9, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32S_AVX512(zmm10, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32S_AVX512(zmm11, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32S_AVX512(zmm12, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32S_AVX512(zmm13, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32S_AVX512(zmm14, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32S_AVX512(zmm15, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32S_AVX512(zmm16, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32S_AVX512(zmm17, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32S_AVX512(zmm18, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32S_AVX512(zmm19, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_3x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32S_AVX512(zmm8, min, max)

			// c[0, 16-31]
			CLIP_F32S_AVX512(zmm9, min, max)

			// c[0, 32-47]
			CLIP_F32S_AVX512(zmm10, min, max)

			// c[0, 48-63]
			CLIP_F32S_AVX512(zmm11, min, max)

			// c[1, 0-15]
			CLIP_F32S_AVX512(zmm12, min, max)

			// c[1, 16-31]
			CLIP_F32S_AVX512(zmm13, min, max)

			// c[1, 32-47]
			CLIP_F32S_AVX512(zmm14, min, max)

			// c[1, 48-63]
			CLIP_F32S_AVX512(zmm15, min, max)

			// c[2, 0-15]
			CLIP_F32S_AVX512(zmm16, min, max)

			// c[2, 16-31]
			CLIP_F32S_AVX512(zmm17, min, max)

			// c[2, 32-47]
			CLIP_F32S_AVX512(zmm18, min, max)

			// c[2, 48-63]
			CLIP_F32S_AVX512(zmm19, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_3x64_OPS:
{
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( post_ops_list_temp->zp_stor_type == BF16 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S32 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S32_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
		else if ( post_ops_list_temp->zp_stor_type == U8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          U8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            BF16_F32_ZP_LOAD(zero_point0, k0, 0);
            BF16_F32_ZP_LOAD(zero_point1, k1, 1);
            BF16_F32_ZP_LOAD(zero_point2, k2, 2);
            BF16_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            S32_F32_ZP_LOAD(zero_point0, k0, 0);
            S32_F32_ZP_LOAD(zero_point1, k1, 1);
            S32_F32_ZP_LOAD(zero_point2, k2, 2);
            S32_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            S8_F32_ZP_LOAD(zero_point0, k0, 0);
            S8_F32_ZP_LOAD(zero_point1, k1, 1);
            S8_F32_ZP_LOAD(zero_point2, k2, 2);
            S8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            U8_F32_ZP_LOAD(zero_point0, k0, 0);
            U8_F32_ZP_LOAD(zero_point1, k1, 1);
            U8_F32_ZP_LOAD(zero_point2, k2, 2);
            U8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps( k0, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_maskz_loadu_ps( k1, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_maskz_loadu_ps( k2, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_maskz_loadu_ps( k3, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            BF16_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            BF16_F32_ZP_BCAST(zero_point2, zp_mask, 2);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S32_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S32_F32_ZP_BCAST(zero_point2, zp_mask, 2);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            S8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            U8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
            U8_F32_ZP_BCAST(zero_point2, zp_mask, 2);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
}
POST_OPS_MATRIX_ADD_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
				bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
				bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
				scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19, \
					scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,zmm1,zmm2,zmm3,zmm4,2);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_3x64_OPS:
		{
			zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(zmm8, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(zmm9, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(zmm10, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(zmm11, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(zmm12, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(zmm13, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(zmm14, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(zmm15, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(zmm16, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(zmm17, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(zmm18, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(zmm19, zmm1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_3x64_OPS:
			{
				__m512 dn, z, x, r2, r;
				__m512i q;

				// c[0, 0-15]
				TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, q)

				// c[0, 16-31]
				TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, q)

				// c[0, 32-47]
				TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, q)

				// c[0, 48-63]
				TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, q)

				// c[1, 0-15]
				TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, q)

				// c[1, 16-31]
				TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, q)

				// c[1, 32-47]
				TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, q)

				// c[1, 48-63]
				TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, q)

				// c[2, 0-15]
				TANH_F32S_AVX512(zmm16, r, r2, x, z, dn, q)

				// c[2, 16-31]
				TANH_F32S_AVX512(zmm17, r, r2, x, z, dn, q)

				// c[2, 32-47]
				TANH_F32S_AVX512(zmm18, r, r2, x, z, dn, q)

				// c[2, 48-63]
				TANH_F32S_AVX512(zmm19, r, r2, x, z, dn, q)

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_SIGMOID_3x64_OPS:
			{
				__m512 al_in, r, r2, z, dn;
				__m512i ex_out;

				// c[0, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);

				// c[0, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm9, al_in, r, r2, z, dn, ex_out);

				// c[0, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm10, al_in, r, r2, z, dn, ex_out);

				// c[0, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm11, al_in, r, r2, z, dn, ex_out);

				// c[1, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm12, al_in, r, r2, z, dn, ex_out);

				// c[1, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm13, al_in, r, r2, z, dn, ex_out);

				// c[1, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm14, al_in, r, r2, z, dn, ex_out);

				// c[1, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm15, al_in, r, r2, z, dn, ex_out);

				// c[2, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm16, al_in, r, r2, z, dn, ex_out);

				// c[2, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm17, al_in, r, r2, z, dn, ex_out);

				// c[2, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm18, al_in, r, r2, z, dn, ex_out);

				// c[2, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm19, al_in, r, r2, z, dn, ex_out);

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_3x64_OPS_DISABLE:
			;

			// Case where the output C matrix is bf16 (downscaled) and this is the
			// final write for a given block within C.
			if ( post_ops_attr.c_stor_type == BF16 )
			{

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm19,k3,2,48);
			}
			else if ( post_ops_attr.c_stor_type == S32 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int32_t* b_q = ( int32_t* )b;
				dim_t ir = 0;
				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm19,k3,2,48);
			}
			else if ( post_ops_attr.c_stor_type == S8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int8_t* b_q = ( int8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm19,k3,2,48);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				uint8_t* b_q = ( uint8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm15,k3,1,48);

				// c[2, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm16,k0,2,0);
				// c[2, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm17,k1,2,16);
				// c[2, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm18,k2,2,32);
				// c[2, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm19,k3,2,48);
			}
			else // Case where the output C matrix is float
			{
				// Store the results.
				// c[0,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm8 );
				// c[0,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm9 );
				// c[0,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm10 );
				// c[0,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm11 );

				// c[1,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm12 );
				// c[1,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm13 );
				// c[1,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm14 );
				// c[1,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm15 );

				// c[2,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm16 );
				// c[2,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm17 );
				// c[2,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm18 );
				// c[2,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm19 );
			}
            post_ops_attr.post_op_c_j += NR_L;
		}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_2x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_2x64_OPS,
						  &&POST_OPS_RELU_2x64_OPS,
						  &&POST_OPS_RELU_SCALE_2x64_OPS,
						  &&POST_OPS_GELU_TANH_2x64_OPS,
						  &&POST_OPS_GELU_ERF_2x64_OPS,
						  &&POST_OPS_CLIP_2x64_OPS,
						  &&POST_OPS_DOWNSCALE_2x64_OPS,
						  &&POST_OPS_MATRIX_ADD_2x64_OPS,
						  &&POST_OPS_SWISH_2x64_OPS,
						  &&POST_OPS_MATRIX_MUL_2x64_OPS,
						  &&POST_OPS_TANH_2x64_OPS,
						  &&POST_OPS_SIGMOID_2x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 zmm8 = _mm512_setzero_ps();
	__m512 zmm9 = _mm512_setzero_ps();
	__m512 zmm10 = _mm512_setzero_ps();
	__m512 zmm11 = _mm512_setzero_ps();

	__m512 zmm12 = _mm512_setzero_ps();
	__m512 zmm13 = _mm512_setzero_ps();
	__m512 zmm14 = _mm512_setzero_ps();
	__m512 zmm15 = _mm512_setzero_ps();

	__m512 zmm1 = _mm512_setzero_ps();
	__m512 zmm2 = _mm512_setzero_ps();
	__m512 zmm3 = _mm512_setzero_ps();
	__m512 zmm4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		zmm8 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ));
		zmm9 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm10 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm11 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) );

		// 2ndx64 block.
		zmm12 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ));
		zmm13 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm14 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm15 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x64_OPS:
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						BF16_F32_BIAS_LOAD(zmm1, k0, 0);
						BF16_F32_BIAS_LOAD(zmm2, k1, 1);
						BF16_F32_BIAS_LOAD(zmm3, k2, 2);
						BF16_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						S32_F32_BIAS_LOAD(zmm1, k0, 0);
						S32_F32_BIAS_LOAD(zmm2, k1, 1);
						S32_F32_BIAS_LOAD(zmm3, k2, 2);
						S32_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						S8_F32_BIAS_LOAD(zmm1, k0, 0);
						S8_F32_BIAS_LOAD(zmm2, k1, 1);
						S8_F32_BIAS_LOAD(zmm3, k2, 2);
						S8_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else
					{
						zmm1 =_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
						zmm2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
						zmm3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
						zmm4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm2, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm3, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm4, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm1, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm3, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm4, zmm15 );
				}
				else
				{
					// If original output was columns major, then by the time
					// kernel sees it, the matrix would be accessed as if it were
					// transposed. Due to this the bias array will be accessed by
					// the ic index, and each bias element corresponds to an
					// entire row of the transposed output array, instead of an
					// entire column.
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S32_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S32_F32_BIAS_BCAST(zmm2, bias_mask, 1);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S8_F32_BIAS_BCAST(zmm1, bias_mask, 0);
						S8_F32_BIAS_BCAST(zmm2, bias_mask, 1);
					}
					else
					{
						zmm1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 0 ) );
						zmm2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 1 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm1, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm1, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm1, zmm11 );

					// c[1,0-15]
					zmm12 = _mm512_add_ps( zmm2, zmm12 );

					// c[1, 16-31]
					zmm13 = _mm512_add_ps( zmm2, zmm13 );

					// c[1,32-47]
					zmm14 = _mm512_add_ps( zmm2, zmm14 );

					// c[1,48-63]
					zmm15 = _mm512_add_ps( zmm2, zmm15 );
				}

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_RELU_2x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();

			// c[0,0-15]
			zmm8 = _mm512_max_ps( zmm1, zmm8 );

			// c[0, 16-31]
			zmm9 = _mm512_max_ps( zmm1, zmm9 );

			// c[0,32-47]
			zmm10 = _mm512_max_ps( zmm1, zmm10 );

			// c[0,48-63]
			zmm11 = _mm512_max_ps( zmm1, zmm11 );

			// c[1,0-15]
			zmm12 = _mm512_max_ps( zmm1, zmm12 );

			// c[1,16-31]
			zmm13 = _mm512_max_ps( zmm1, zmm13 );

			// c[1,32-47]
			zmm14 = _mm512_max_ps( zmm1, zmm14 );

			// c[1,48-63]
			zmm15 = _mm512_max_ps( zmm1, zmm15 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_2x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();
			zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm8)

			// c[0, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm9)

			// c[0, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm10)

			// c[0, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm11)

			// c[1, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm12)

			// c[1, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm13)

			// c[1, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm14)

			// c[1, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm15)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_2x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_2x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32S_AVX512(zmm8, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32S_AVX512(zmm9, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32S_AVX512(zmm10, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32S_AVX512(zmm11, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32S_AVX512(zmm12, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32S_AVX512(zmm13, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32S_AVX512(zmm14, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32S_AVX512(zmm15, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_2x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32S_AVX512(zmm8, min, max)

			// c[0, 16-31]
			CLIP_F32S_AVX512(zmm9, min, max)

			// c[0, 32-47]
			CLIP_F32S_AVX512(zmm10, min, max)

			// c[0, 48-63]
			CLIP_F32S_AVX512(zmm11, min, max)

			// c[1, 0-15]
			CLIP_F32S_AVX512(zmm12, min, max)

			// c[1, 16-31]
			CLIP_F32S_AVX512(zmm13, min, max)

			// c[1, 32-47]
			CLIP_F32S_AVX512(zmm14, min, max)

			// c[1, 48-63]
			CLIP_F32S_AVX512(zmm15, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_2x64_OPS:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( post_ops_list_temp->zp_stor_type == BF16 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S32 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S32_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
		else if ( post_ops_list_temp->zp_stor_type == U8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          U8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            BF16_F32_ZP_LOAD(zero_point0, k0, 0);
            BF16_F32_ZP_LOAD(zero_point1, k1, 1);
            BF16_F32_ZP_LOAD(zero_point2, k2, 2);
            BF16_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            S32_F32_ZP_LOAD(zero_point0, k0, 0);
            S32_F32_ZP_LOAD(zero_point1, k1, 1);
            S32_F32_ZP_LOAD(zero_point2, k2, 2);
            S32_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            S8_F32_ZP_LOAD(zero_point0, k0, 0);
            S8_F32_ZP_LOAD(zero_point1, k1, 1);
            S8_F32_ZP_LOAD(zero_point2, k2, 2);
            S8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            U8_F32_ZP_LOAD(zero_point0, k0, 0);
            U8_F32_ZP_LOAD(zero_point1, k1, 1);
            U8_F32_ZP_LOAD(zero_point2, k2, 2);
            U8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps( k0, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_maskz_loadu_ps( k1, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_maskz_loadu_ps( k2, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_maskz_loadu_ps( k3, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            BF16_F32_ZP_BCAST(zero_point1, zp_mask, 1);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S32_F32_ZP_BCAST(zero_point1, zp_mask, 1);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            S8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
            U8_F32_ZP_BCAST(zero_point1, zp_mask, 1);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);
      }
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
}
POST_OPS_MATRIX_ADD_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
				bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
				bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
				scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
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
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15, \
					scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,zmm1,zmm2,zmm3,zmm4,1);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_2x64_OPS:
		{
			zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(zmm8, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(zmm9, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(zmm10, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(zmm11, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(zmm12, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(zmm13, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(zmm14, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(zmm15, zmm1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_2x64_OPS:
			{
				__m512 dn, z, x, r2, r;
				__m512i q;

				// c[0, 0-15]
				TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, q)

				// c[0, 16-31]
				TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, q)

				// c[0, 32-47]
				TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, q)

				// c[0, 48-63]
				TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, q)

				// c[1, 0-15]
				TANH_F32S_AVX512(zmm12, r, r2, x, z, dn, q)

				// c[1, 16-31]
				TANH_F32S_AVX512(zmm13, r, r2, x, z, dn, q)

				// c[1, 32-47]
				TANH_F32S_AVX512(zmm14, r, r2, x, z, dn, q)

				// c[1, 48-63]
				TANH_F32S_AVX512(zmm15, r, r2, x, z, dn, q)

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_SIGMOID_2x64_OPS:
			{
				__m512 al_in, r, r2, z, dn;
				__m512i ex_out;

				// c[0, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);

				// c[0, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm9, al_in, r, r2, z, dn, ex_out);

				// c[0, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm10, al_in, r, r2, z, dn, ex_out);

				// c[0, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm11, al_in, r, r2, z, dn, ex_out);

				// c[1, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm12, al_in, r, r2, z, dn, ex_out);

				// c[1, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm13, al_in, r, r2, z, dn, ex_out);

				// c[1, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm14, al_in, r, r2, z, dn, ex_out);

				// c[1, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm15, al_in, r, r2, z, dn, ex_out);

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_2x64_OPS_DISABLE:
			;

			// Case where the output C matrix is bf16 (downscaled) and this is the
			// final write for a given block within C.
			if ( post_ops_attr.c_stor_type == BF16 )
			{

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm15,k3,1,48);
			}
			else if ( post_ops_attr.c_stor_type == S32 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int32_t* b_q = ( int32_t* )b;
				dim_t ir = 0;
				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm15,k3,1,48);
			}
			else if ( post_ops_attr.c_stor_type == S8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int8_t* b_q = ( int8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm15,k3,1,48);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				uint8_t* b_q = ( uint8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm11,k3,0,48);

				// c[1, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm12,k0,1,0);
				// c[1, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm13,k1,1,16);
				// c[1, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm14,k2,1,32);
				// c[1, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm15,k3,1,48);
			}
			else // Case where the output C matrix is float
			{
				// Store the results.
				// c[0,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm8 );
				// c[0,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm9 );
				// c[0,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm10 );
				// c[0,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm11 );

				// c[1,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm12 );
				// c[1,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm13 );
				// c[1,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm14 );
				// c[1,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm15 );
			}
            post_ops_attr.post_op_c_j += NR_L;
		}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_1x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_1x64_OPS,
						  &&POST_OPS_RELU_1x64_OPS,
						  &&POST_OPS_RELU_SCALE_1x64_OPS,
						  &&POST_OPS_GELU_TANH_1x64_OPS,
						  &&POST_OPS_GELU_ERF_1x64_OPS,
						  &&POST_OPS_CLIP_1x64_OPS,
						  &&POST_OPS_DOWNSCALE_1x64_OPS,
						  &&POST_OPS_MATRIX_ADD_1x64_OPS,
						  &&POST_OPS_SWISH_1x64_OPS,
						  &&POST_OPS_MATRIX_MUL_1x64_OPS,
						  &&POST_OPS_TANH_1x64_OPS,
						  &&POST_OPS_SIGMOID_1x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 zmm8 = _mm512_setzero_ps();
	__m512 zmm9 = _mm512_setzero_ps();
	__m512 zmm10 = _mm512_setzero_ps();
	__m512 zmm11 = _mm512_setzero_ps();

	__m512 zmm1 = _mm512_setzero_ps();
	__m512 zmm2 = _mm512_setzero_ps();
	__m512 zmm3 = _mm512_setzero_ps();
	__m512 zmm4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		zmm8 = _mm512_maskz_loadu_ps( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) );
		zmm9 = _mm512_maskz_loadu_ps( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) );
		zmm10 = _mm512_maskz_loadu_ps( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) );
		zmm11 = _mm512_maskz_loadu_ps( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64_OPS:
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						BF16_F32_BIAS_LOAD(zmm1, k0, 0);
						BF16_F32_BIAS_LOAD(zmm2, k1, 1);
						BF16_F32_BIAS_LOAD(zmm3, k2, 2);
						BF16_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						S32_F32_BIAS_LOAD(zmm1, k0, 0);
						S32_F32_BIAS_LOAD(zmm2, k1, 1);
						S32_F32_BIAS_LOAD(zmm3, k2, 2);
						S32_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						S8_F32_BIAS_LOAD(zmm1, k0, 0);
						S8_F32_BIAS_LOAD(zmm2, k1, 1);
						S8_F32_BIAS_LOAD(zmm3, k2, 2);
						S8_F32_BIAS_LOAD(zmm4, k3, 3);
					}
					else
					{
						zmm1 =_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
						zmm2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
						zmm3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
						zmm4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm2, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm3, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm4, zmm11 );
				}
				else
				{
					// If original output was columns major, then by the time
					// kernel sees it, the matrix would be accessed as if it were
					// transposed. Due to this the bias array will be accessed by
					// the ic index, and each bias element corresponds to an
					// entire row of the transposed output array, instead of an
					// entire column.
					if ( post_ops_list_temp->stor_type == BF16 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0);
					}
					else if ( post_ops_list_temp->stor_type == S32 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S32_F32_BIAS_BCAST(zmm1, bias_mask, 0);
					}
					else if ( post_ops_list_temp->stor_type == S8 )
					{
						__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
						S8_F32_BIAS_BCAST(zmm1, bias_mask, 0);
					}
					else
					{
						zmm1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 0 ) );
					}

					// c[0,0-15]
					zmm8 = _mm512_add_ps( zmm1, zmm8 );

					// c[0, 16-31]
					zmm9 = _mm512_add_ps( zmm1, zmm9 );

					// c[0,32-47]
					zmm10 = _mm512_add_ps( zmm1, zmm10 );

					// c[0,48-63]
					zmm11 = _mm512_add_ps( zmm1, zmm11 );
				}

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_RELU_1x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();

			// c[0,0-15]
			zmm8 = _mm512_max_ps( zmm1, zmm8 );

			// c[0, 16-31]
			zmm9 = _mm512_max_ps( zmm1, zmm9 );

			// c[0,32-47]
			zmm10 = _mm512_max_ps( zmm1, zmm10 );

			// c[0,48-63]
			zmm11 = _mm512_max_ps( zmm1, zmm11 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_1x64_OPS:
		{
			zmm1 = _mm512_setzero_ps();
			zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32S_AVX512(zmm8)

			// c[0, 16-31]
			RELU_SCALE_OP_F32S_AVX512(zmm9)

			// c[0, 32-47]
			RELU_SCALE_OP_F32S_AVX512(zmm10)

			// c[0, 48-63]
			RELU_SCALE_OP_F32S_AVX512(zmm11)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_1x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_1x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32S_AVX512(zmm8, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32S_AVX512(zmm9, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32S_AVX512(zmm10, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32S_AVX512(zmm11, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_1x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32S_AVX512(zmm8, min, max)

			// c[0, 16-31]
			CLIP_F32S_AVX512(zmm9, min, max)

			// c[0, 32-47]
			CLIP_F32S_AVX512(zmm10, min, max)

			// c[0, 48-63]
			CLIP_F32S_AVX512(zmm11, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_1x64_OPS:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( post_ops_list_temp->zp_stor_type == BF16 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          BF16_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S32 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S32_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S32_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else if ( post_ops_list_temp->zp_stor_type == S8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          S8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          S8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
		else if ( post_ops_list_temp->zp_stor_type == U8 )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          U8_F32_SCALAR_ZP_BCAST(zero_point0, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point1, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point2, zp_mask);
          U8_F32_SCALAR_ZP_BCAST(zero_point3, zp_mask);
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            BF16_F32_ZP_LOAD(zero_point0, k0, 0);
            BF16_F32_ZP_LOAD(zero_point1, k1, 1);
            BF16_F32_ZP_LOAD(zero_point2, k2, 2);
            BF16_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            S32_F32_ZP_LOAD(zero_point0, k0, 0);
            S32_F32_ZP_LOAD(zero_point1, k1, 1);
            S32_F32_ZP_LOAD(zero_point2, k2, 2);
            S32_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            S8_F32_ZP_LOAD(zero_point0, k0, 0);
            S8_F32_ZP_LOAD(zero_point1, k1, 1);
            S8_F32_ZP_LOAD(zero_point2, k2, 2);
            S8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            U8_F32_ZP_LOAD(zero_point0, k0, 0);
            U8_F32_ZP_LOAD(zero_point1, k1, 1);
            U8_F32_ZP_LOAD(zero_point2, k2, 2);
            U8_F32_ZP_LOAD(zero_point3, k3, 3);
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps( k0, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_maskz_loadu_ps( k1, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_maskz_loadu_ps( k2, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_maskz_loadu_ps( k3, (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( post_ops_list_temp->zp_stor_type == BF16 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else if ( post_ops_list_temp->zp_stor_type == S32 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S32_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else if ( post_ops_list_temp->zp_stor_type == S8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            S8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
		  else if ( post_ops_list_temp->zp_stor_type == U8 )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            U8_F32_ZP_BCAST(zero_point0, zp_mask, 0);
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_i + 0 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 );
				bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
				bool is_s32 = ( post_ops_list_temp->stor_type == S32 );

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

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
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
				scl_fctr1 =
					_mm512_maskz_loadu_ps( k0, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k1, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k2, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k3, ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_i + 0 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else if ( is_s32 == TRUE )
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,zmm1,zmm2,zmm3,zmm4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11, \
					scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,zmm1,zmm2,zmm3,zmm4,0);
				}
			}
			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_1x64_OPS:
		{
			zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(zmm8, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(zmm9, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(zmm10, zmm1, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(zmm11, zmm1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_1x64_OPS:
			{
				__m512 dn, z, x, r2, r;
				__m512i q;

				// c[0, 0-15]
				TANH_F32S_AVX512(zmm8, r, r2, x, z, dn, q)

				// c[0, 16-31]
				TANH_F32S_AVX512(zmm9, r, r2, x, z, dn, q)

				// c[0, 32-47]
				TANH_F32S_AVX512(zmm10, r, r2, x, z, dn, q)

				// c[0, 48-63]
				TANH_F32S_AVX512(zmm11, r, r2, x, z, dn, q)

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_SIGMOID_1x64_OPS:
			{
				__m512 al_in, r, r2, z, dn;
				__m512i ex_out;

				// c[0, 0-15]
				SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);

				// c[0, 16-31]
				SIGMOID_F32_AVX512_DEF(zmm9, al_in, r, r2, z, dn, ex_out);

				// c[0, 32-47]
				SIGMOID_F32_AVX512_DEF(zmm10, al_in, r, r2, z, dn, ex_out);

				// c[0, 48-63]
				SIGMOID_F32_AVX512_DEF(zmm11, al_in, r, r2, z, dn, ex_out);

				POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
			}
POST_OPS_1x64_OPS_DISABLE:
			;

			// Case where the output C matrix is bf16 (downscaled) and this is the
			// final write for a given block within C.
			if ( post_ops_attr.c_stor_type == BF16 )
			{

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_BF16_POST_OPS_MASK(0,jr,zmm11,k3,0,48);
			}
			else if ( post_ops_attr.c_stor_type == S32 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int32_t* b_q = ( int32_t* )b;
				dim_t ir = 0;
				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S32_POST_OPS_MASK(zmm11,k3,0,48);
			}
			else if ( post_ops_attr.c_stor_type == S8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				int8_t* b_q = ( int8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_S8_POST_OPS_MASK(zmm11,k3,0,48);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Actually the b matrix is of type bfloat16. However
				// in order to reuse this kernel for f32, the output
				// matrix type in kernel function signature is set to
				// f32 irrespective of original output matrix type.
				uint8_t* b_q = ( uint8_t* )b;
				dim_t ir = 0;

				// Store the results in downscaled type (bf16 instead of float).
				// c[0, 0-15]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm8,k0,0,0);
				// c[0, 16-31]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm9,k1,0,16);
				// c[0, 32-47]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm10,k2,0,32);
				// c[0, 48-63]
				CVT_STORE_F32_U8_POST_OPS_MASK(zmm11,k3,0,48);
			}
			else // Case where the output C matrix is float
			{
				// Store the results.
				// c[0,0-15]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 0 ) ), k0, zmm8 );
				// c[0,16-31]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 16 ) ), k1, zmm9 );
				// c[0,32-47]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 32 ) ), k2, zmm10 );
				// c[0,48-63]
				_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) +
					( cs_b * ( jr + 48 ) ), k3, zmm11 );
			}
            post_ops_attr.post_op_c_j += NR_L;
		}
}

#endif
