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
						  NULL,// Virtual node for downscale, else segfault
						  &&POST_OPS_MATRIX_ADD_5x64_OPS,
						  &&POST_OPS_SWISH_5x64_OPS,
						  &&POST_OPS_MATRIX_MUL_5x64_OPS
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
				zmm1 =_mm512_maskz_loadu_ps( k0,
					( float* )post_ops_list_temp->op_args1 +
				post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				zmm2 =_mm512_maskz_loadu_ps( k1,
					( float* )post_ops_list_temp->op_args1 +
				post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				zmm3 =_mm512_maskz_loadu_ps( k2,
					( float* )post_ops_list_temp->op_args1 +
				post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				zmm4 =_mm512_maskz_loadu_ps( k3,
					( float* )post_ops_list_temp->op_args1 +
				post_ops_attr.post_op_c_j + ( 3 * 16 ) );

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
POST_OPS_MATRIX_ADD_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23,zmm1,zmm2,zmm3,zmm4,3);

			// c[4:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,24,25,26,27,zmm1,zmm2,zmm3,zmm4,4);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23,zmm1,zmm2,zmm3,zmm4,3);

			// c[4:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,24,25,26,27,zmm1,zmm2,zmm3,zmm4,4);

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
POST_OPS_5x64_OPS_DISABLE:
		;

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
						  NULL,// Virtual node for downscale, else segfault
						  &&POST_OPS_MATRIX_ADD_4x64_OPS,
						  &&POST_OPS_SWISH_4x64_OPS,
						  &&POST_OPS_MATRIX_MUL_4x64_OPS
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
				zmm1 =
				_mm512_maskz_loadu_ps( k0,
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
POST_OPS_MATRIX_ADD_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,20,21,22,23,zmm1,zmm2,zmm3,zmm4,3);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,20,21,22,23,zmm1,zmm2,zmm3,zmm4,3);

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
POST_OPS_4x64_OPS_DISABLE:
		;

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
						  NULL,// Virtual node for downscale, else segfault
						  &&POST_OPS_MATRIX_ADD_3x64_OPS,
						  &&POST_OPS_SWISH_3x64_OPS,
						  &&POST_OPS_MATRIX_MUL_3x64_OPS
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
				zmm1 =
				_mm512_maskz_loadu_ps( k0,
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
				zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 0 ) );
				zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 1 ) );
				zmm3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 2 ) );

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
POST_OPS_MATRIX_ADD_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.

			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,16,17,18,19,zmm1,zmm2,zmm3,zmm4,2);

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
POST_OPS_3x64_OPS_DISABLE:
		;

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
						  NULL,// Virtual node for downscale, else segfault
						  &&POST_OPS_MATRIX_ADD_2x64_OPS,
						  &&POST_OPS_SWISH_2x64_OPS,
						  &&POST_OPS_MATRIX_MUL_2x64_OPS
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
				zmm1 =
				_mm512_maskz_loadu_ps( k0,
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
				zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 0 ) );
				zmm2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 1 ) );

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
POST_OPS_MATRIX_ADD_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,12,13,14,15,zmm1,zmm2,zmm3,zmm4,1);

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
POST_OPS_2x64_OPS_DISABLE:
		;

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
						  NULL,// Virtual node for downscale, else segfault
						  &&POST_OPS_MATRIX_ADD_1x64_OPS,
						  &&POST_OPS_SWISH_1x64_OPS,
						  &&POST_OPS_MATRIX_MUL_1x64_OPS
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
				zmm1 =
				_mm512_maskz_loadu_ps( k0,
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
				zmm1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
					post_ops_attr.post_op_c_i + 0 ) );

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
POST_OPS_MATRIX_ADD_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,8,9,10,11,zmm1,zmm2,zmm3,zmm4,0);

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
POST_OPS_1x64_OPS_DISABLE:
		;

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

		post_ops_attr.post_op_c_j += NR_L;
	}
}

#endif
