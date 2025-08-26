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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "../../../zen/lpgemm/f32f32f32/lpgemm_kernel_macros_f32_avx2.h"

// Zero-out the given YMM accumulator registers
#define ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm2, ymm3) \
	ymm0 = _mm256_setzero_ps(); \
	ymm1 = _mm256_setzero_ps(); \
	ymm2 = _mm256_setzero_ps(); \
	ymm3 = _mm256_setzero_ps();

LPGEMV_M_EQ1_KERN( float, float, float, f32f32f32of32_avx512_256 )
{
	static void *post_ops_labels[] =
		{
			&&POST_OPS_1x64F_DISABLE,
			&&POST_OPS_BIAS_1x64F,
			&&POST_OPS_RELU_1x64F,
			&&POST_OPS_RELU_SCALE_1x64F,
			&&POST_OPS_GELU_TANH_1x64F,
			&&POST_OPS_GELU_ERF_1x64F,
			&&POST_OPS_CLIP_1x64F,
			&&POST_OPS_DOWNSCALE_1x64F,
			&&POST_OPS_MATRIX_ADD_1x64F,
			&&POST_OPS_SWISH_1x64F,
			&&POST_OPS_MATRIX_MUL_1x64F,
			&&POST_OPS_TANH_1x64F,
			&&POST_OPS_SIGMOID_1x64F
		};

	// Strides are updated based on matrix packing/reordering.
	const float *a_use = NULL;
	const float *b_use = NULL;
	float *c_use = NULL;
	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	// Predefined masks for handling edge cases
	__m256i masks[9] = {
		_mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
		_mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
		_mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
		_mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
		_mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
		_mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
		_mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
		_mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),    // 7 elements
		_mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),   // 8 elements
	};

	for( dim_t jr = 0; jr < n0; jr += NR )
	{
		dim_t nr0 = bli_min((n0 - jr), NR);
		c_use = c + jr;

		// masks to handle edge cases.
		__m256i k1 = masks[8], k2 = masks[8], k3 = masks[8], k4 = masks[8];
		__m256i k5 = masks[8], k6 = masks[8], k7 = masks[8], k8 = masks[8];

		// Adjust masks based on the number of remaining columns
		if (nr0 < NR)
		{
			dim_t n_left = n0 % 8;
			if (nr0 >= 56)
			{
				k8 = masks[n_left];
			}
			else if (nr0 >= 48)
			{
				k7 =  masks[n_left];
				k8 = masks[0];
			}
			else if (nr0 >= 40)
			{
				k6 =  masks[n_left];
				k7 = k8 = masks[0];
			}
			else if (nr0 >= 32)
			{
				k5 = masks[n_left];
				k6 = k7 = k8 = masks[0];
			}
			else if (nr0 >= 24)
			{
				k4 = masks[n_left];
				k5 = k6 = k7 = k8 = masks[0];
			}
			else if (nr0 >= 16)
			{
				k3 =  masks[n_left];
				k4 = k5 = k6 = k7 = k8 = masks[0];
			}
			else if (nr0 >= 8)
			{
				k2 = masks[n_left];
				k3 = k4 = k5 = k6 = k7 = k8 = masks[0];
			}
			else
			{
				k2 = k3 = k4 = k5 = k6 = k7 = k8 = masks[0];
				k1 = masks[n_left];
			}
		}

		 // Declare YMM registers for computation
		__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
		__m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14;
		__m256 ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21;
		__m256 ymm22, ymm23, ymm24, ymm25, ymm26, ymm27, ymm28;
		__m256 ymm29, ymm30, ymm31;

		// zero out the registers
		ZERO_ACC_YMM_4_REG(ymm8, ymm9, ymm10, ymm11);
		ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);
		ZERO_ACC_YMM_4_REG(ymm16, ymm17, ymm18, ymm19);
		ZERO_ACC_YMM_4_REG(ymm20, ymm21, ymm22, ymm23);
		ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm2, ymm3);
		ZERO_ACC_YMM_4_REG(ymm24, ymm25, ymm26, ymm27);
		ZERO_ACC_YMM_4_REG(ymm28, ymm29, ymm30, ymm31);

		//_mm_prefetch( (MR X NR) from C
		_mm_prefetch((c_use + 0 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 8 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 16 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 24 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 32 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 40 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 48 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 56 * rs_c), _MM_HINT_T0);

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			dim_t kc0 = bli_min((k - pc), KC);
			uint64_t k_iter = kc0 / 2;
			uint64_t k_rem = kc0 % 2;
			dim_t ps_b_use = 0;
			dim_t rs_b_use = NR;
			// No parallelization in k dim, k always starts at 0.
			if (mtag_b == REORDERED||mtag_b == PACK)
			{
				// In multi-threaded scenarios, an extra offset into a given
				// packed B panel is required, since the jc loop split can
				// result in per thread start offset inside the panel, instead
				// of panel boundaries.
				b_use = b + (n_sub_updated * pc) + (jc_cur_loop_rem * kc0);
				ps_b_use = kc0;
			}
			else
			{
				b_use = b + (pc * rs_b);
				ps_b_use = 1;
				rs_b_use = rs_b;
			}

			a_use = a + pc;
			b_use = b_use + jr * ps_b_use;

			for (dim_t k = 0; k < k_iter; k++)
			{
				 _mm_prefetch((b_use + 2 * rs_b_use), _MM_HINT_T0);
				//Using mask loads to avoid writing fringe kernels

				//Broadcast col0 - col1 element of A
				ymm0 = _mm256_broadcast_ss( a_use );	 // broadcast c0
				ymm1 = _mm256_broadcast_ss( a_use + 1 ); // broadcast c1

				//Load first 4x8 tile from row 0-1
				ymm2 = _mm256_maskload_ps(b_use, k1);
				ymm3 = _mm256_maskload_ps(b_use + rs_b_use, k1);
				b_use += 8;

				ymm4 = _mm256_maskload_ps(b_use, k2);
				ymm5 = _mm256_maskload_ps(b_use + rs_b_use, k2);
				b_use += 8;

				ymm6 = _mm256_maskload_ps(b_use, k3);
				ymm7 = _mm256_maskload_ps(b_use + rs_b_use, k3);
				b_use += 8;

				ymm8 = _mm256_maskload_ps(b_use, k4);
				ymm9 = _mm256_maskload_ps(b_use + rs_b_use, k4);
				b_use += 8;

				ymm10 = _mm256_maskload_ps(b_use, k5);
				ymm11 = _mm256_maskload_ps(b_use + rs_b_use, k5);
				b_use += 8;

				ymm12 = _mm256_maskload_ps(b_use, k6);
				ymm13 = _mm256_maskload_ps(b_use + rs_b_use, k6);
				b_use += 8;

				ymm16 = _mm256_fmadd_ps( ymm2, ymm0, ymm16);
				ymm17 = _mm256_fmadd_ps( ymm3, ymm1, ymm17);
				ymm18 = _mm256_fmadd_ps( ymm4, ymm0, ymm18);
				ymm19 = _mm256_fmadd_ps( ymm5, ymm1, ymm19);
				ymm20 = _mm256_fmadd_ps( ymm6, ymm0, ymm20);
				ymm21 = _mm256_fmadd_ps( ymm7, ymm1, ymm21);
				ymm22 = _mm256_fmadd_ps( ymm8, ymm0, ymm22);
				ymm23 = _mm256_fmadd_ps( ymm9, ymm1, ymm23);
				ymm24 = _mm256_fmadd_ps( ymm10,ymm0, ymm24);
				ymm25 = _mm256_fmadd_ps( ymm11,ymm1, ymm25);
				ymm26 = _mm256_fmadd_ps( ymm12,ymm0, ymm26);
				ymm27 = _mm256_fmadd_ps( ymm13,ymm1, ymm27);

				ymm14 = _mm256_maskload_ps(b_use, k7);
				ymm15 = _mm256_maskload_ps(b_use + rs_b_use, k7);
				b_use += 8;

				ymm2 = _mm256_maskload_ps(b_use, k8);
				ymm3 = _mm256_maskload_ps(b_use + rs_b_use, k8);

				ymm28 = _mm256_fmadd_ps(ymm0, ymm14, ymm28);
				ymm29 = _mm256_fmadd_ps(ymm1, ymm15, ymm29);
				ymm30 = _mm256_fmadd_ps(ymm0, ymm2, ymm30);
				ymm31 = _mm256_fmadd_ps(ymm1, ymm3, ymm31);


				b_use -= 56; // move b point back to start of KCXNR
				b_use += (2 * rs_b_use);
				a_use += 2; // move a pointer to next col
			}				// kloop

			for (dim_t kr = 0; kr < k_rem; kr++)
			{
				//Load 64 elements from a row of B
				ymm0 = _mm256_maskload_ps( b_use, k1 );
				ymm1 = _mm256_maskload_ps( b_use + 8, k2 );
				ymm2 = _mm256_maskload_ps( b_use + 16, k3 );
				ymm3 = _mm256_maskload_ps( b_use + 24, k4 );
				ymm4 = _mm256_maskload_ps( b_use + 32, k5 );
				ymm5 = _mm256_maskload_ps( b_use + 40, k6 );
				ymm6 = _mm256_maskload_ps( b_use + 48, k7 );
				ymm7 = _mm256_maskload_ps( b_use + 56, k7 );

				//Broadcast col0 elements of 12 rows of A
				ymm8 = _mm256_set1_ps( * ( a_use ) ); // broadcast c0r0

				ymm16 = _mm256_fmadd_ps(ymm8, ymm0, ymm16);
				ymm18 = _mm256_fmadd_ps(ymm8, ymm1, ymm18);
				ymm20 = _mm256_fmadd_ps(ymm8, ymm2, ymm20);
				ymm22 = _mm256_fmadd_ps(ymm8, ymm3, ymm22);
				ymm24 = _mm256_fmadd_ps(ymm8, ymm4, ymm24);
				ymm26 = _mm256_fmadd_ps(ymm8, ymm5, ymm26);
				ymm28 = _mm256_fmadd_ps(ymm8, ymm6, ymm28);
				ymm30 = _mm256_fmadd_ps(ymm8, ymm7, ymm30);

				b_use += rs_b_use; // move b pointer to next row
				a_use++;		 // move a pointer to next col
			}					// kloop
		}						// kc loop

		//SUMUP K untoll output
		ymm16 = _mm256_add_ps(ymm16, ymm17);
		ymm18 = _mm256_add_ps(ymm18, ymm19);
		ymm20 = _mm256_add_ps(ymm20, ymm21);

		ymm22 = _mm256_add_ps(ymm22, ymm23);
		ymm24 = _mm256_add_ps(ymm24, ymm25);
		ymm26 = _mm256_add_ps(ymm26, ymm27);

		ymm28 = _mm256_add_ps(ymm28, ymm29);
		ymm30 = _mm256_add_ps(ymm30, ymm31);

		//Mulitply A*B output with alpha
		ymm0 = _mm256_set1_ps( alpha );
		ymm16 = _mm256_mul_ps(ymm16, ymm0);
		ymm18 = _mm256_mul_ps(ymm18, ymm0);
		ymm20 = _mm256_mul_ps(ymm20, ymm0);

		ymm22 = _mm256_mul_ps(ymm22, ymm0);
		ymm24 = _mm256_mul_ps(ymm24, ymm0);
		ymm26 = _mm256_mul_ps(ymm26, ymm0);

		ymm28 = _mm256_mul_ps(ymm28, ymm0);
		ymm30 = _mm256_mul_ps(ymm30, ymm0);

		if (beta != 0)
		{
			ymm0 = _mm256_set1_ps( beta );

			if( post_ops_attr.buf_downscale != NULL )
			{
				BF16_F32_C_BNZ_8_MASK(0,0,ymm1, ymm0,ymm16, k1)
				BF16_F32_C_BNZ_8_MASK(0,1,ymm2, ymm0,ymm18, k2)
				BF16_F32_C_BNZ_8_MASK(0,2,ymm3, ymm0,ymm20, k3)
				BF16_F32_C_BNZ_8_MASK(0,3,ymm4, ymm0,ymm22, k4)
				BF16_F32_C_BNZ_8_MASK(0,4,ymm5, ymm0,ymm24, k5)
				BF16_F32_C_BNZ_8_MASK(0,5,ymm6, ymm0,ymm26, k6)
				BF16_F32_C_BNZ_8_MASK(0,6,ymm7, ymm0,ymm28, k7)
				BF16_F32_C_BNZ_8_MASK(0,7,ymm8, ymm0,ymm30, k8)
			}
			else
			{
				const float *_cbuf = c_use;
				// load c and multiply with beta and
				// add to accumulator and store back

				ymm1 = _mm256_maskload_ps( _cbuf, k1 );
				ymm16 = _mm256_fmadd_ps(ymm1, ymm0, ymm16);

				ymm2 = _mm256_maskload_ps( (_cbuf + 8), k2 );
				ymm18 = _mm256_fmadd_ps(ymm2, ymm0, ymm18 );

				ymm3 = _mm256_maskload_ps( (_cbuf + 16), k3 );
				ymm20 = _mm256_fmadd_ps(ymm3, ymm0, ymm20 );

				ymm4 = _mm256_maskload_ps( (_cbuf + 24), k4 );
				ymm22 = _mm256_fmadd_ps(ymm4, ymm0, ymm22 );

				ymm5 = _mm256_maskload_ps( (_cbuf + 32), k5 );
				ymm24 = _mm256_fmadd_ps(ymm5, ymm0, ymm24 );

				ymm6 = _mm256_maskload_ps( (_cbuf + 40), k6 );
				ymm26 = _mm256_fmadd_ps(ymm6, ymm0, ymm26 );

				ymm7 = _mm256_maskload_ps( (_cbuf + 48), k7 );
				ymm28 = _mm256_fmadd_ps(ymm7, ymm0, ymm28 );

				ymm8 = _mm256_maskload_ps( (_cbuf + 56), k8 );
				ymm30 = _mm256_fmadd_ps(ymm8, ymm0, ymm30 );
			}
		}

		// Post Ops
		post_ops_attr.is_last_k = TRUE;
		lpgemm_post_op *post_ops_list_temp = post_op;
		POST_OP_LABEL_LASTK_SAFE_JUMP

	POST_OPS_BIAS_1x64F:
	{
		if ((*(char *)post_ops_list_temp->op_args2 == 'r') ||
			(*(char *)post_ops_list_temp->op_args2 == 'R'))
		{
			if( post_ops_list_temp->stor_type == BF16 )
			{
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm9, 0, k1 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm10, 1, k2 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm11, 2, k3 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm12, 3, k4 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm13, 4, k5 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm14, 5, k6 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm15, 6, k7 );
			  BF16_F32_BIAS_LOAD_AVX2_MASK( ymm8, 7, k8 );
			}
			else
			{
				float* bias_ptr = (float *)post_ops_list_temp->op_args1 +
										post_ops_attr.post_op_c_j;
				ymm9 = _mm256_maskload_ps( bias_ptr + (0 * 8), k1 );

				ymm10 =	_mm256_maskload_ps( bias_ptr + (1 * 8), k2 );

				ymm11 =	_mm256_maskload_ps( bias_ptr + (2 * 8), k3 );

				ymm12 =	_mm256_maskload_ps( bias_ptr + (3 * 8), k4 );

				ymm13 =	_mm256_maskload_ps( bias_ptr + (4 * 8), k5 );

				ymm14 =	_mm256_maskload_ps( bias_ptr + (5 * 8), k6 );

				ymm15 =	_mm256_maskload_ps( bias_ptr + (6 * 8), k7 );

				ymm8 =	_mm256_maskload_ps( bias_ptr + (7 * 8), k8 );
			}
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the bias array will be accessed by
			// the ic index, and each bias element corresponds to an
			// entire row of the transposed output array, instead of an
			// entire column.
			if( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_BCAST_AVX2(ymm9,0);
			}
			else
			{
				float bias = (*((float *)post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0));

				ymm9 =	_mm256_set1_ps(bias);
			}
			ymm10 = ymm11 = ymm12 = ymm13 = ymm14 = ymm15 = ymm8 = ymm9;
		}
		// c[0,0-7]

		ymm16 = _mm256_add_ps(ymm16, ymm9);
		ymm18 = _mm256_add_ps(ymm18, ymm10);
		ymm20 = _mm256_add_ps(ymm20, ymm11);
		ymm22 = _mm256_add_ps(ymm22, ymm12);
		ymm24 = _mm256_add_ps(ymm24, ymm13);
		ymm26 = _mm256_add_ps(ymm26, ymm14);
		ymm28 = _mm256_add_ps(ymm28, ymm15);
		ymm30 = _mm256_add_ps(ymm30, ymm8);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_RELU_1x64F:
	{
		ymm1 = _mm256_setzero_ps();

		ymm16 = _mm256_max_ps( ymm1, ymm16 );
		ymm18 = _mm256_max_ps( ymm1, ymm18 );
		ymm20 = _mm256_max_ps( ymm1, ymm20 );
		ymm22 = _mm256_max_ps( ymm1, ymm22 );
		ymm24 = _mm256_max_ps( ymm1, ymm24 );
		ymm26 = _mm256_max_ps( ymm1, ymm26 );
		ymm28 = _mm256_max_ps( ymm1, ymm28 );
		ymm30 = _mm256_max_ps( ymm1, ymm30 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_RELU_SCALE_1x64F:
	{
		ymm1 = _mm256_setzero_ps();
		ymm0 =
			_mm256_set1_ps(*((float *)post_ops_list_temp->op_args2));

		RELU_SCALE_OP_F32S_AVX2(ymm16, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm18, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm20, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm22, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm24, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm26, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm28, ymm0, ymm1, ymm2)
		RELU_SCALE_OP_F32S_AVX2(ymm30, ymm0, ymm1, ymm2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_GELU_TANH_1x64F:
	{
		__m256 dn, x_tanh;
		__m256i q;

		GELU_TANH_F32S_AVX2(ymm16, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm18, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm20, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm22, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm24, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm26, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm28, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
		GELU_TANH_F32S_AVX2(ymm30, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_GELU_ERF_1x64F:
	{
		GELU_ERF_F32S_AVX2(ymm16, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm18, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm20, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm22, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm24, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm26, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm28, ymm0, ymm1, ymm2)
		GELU_ERF_F32S_AVX2(ymm30, ymm0, ymm1, ymm2)


		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_CLIP_1x64F:
	{
		ymm0 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args2);
		ymm1 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args3);

		CLIP_F32S_AVX2(ymm16,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm18,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm20,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm22,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm24,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm26,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm28,  ymm0, ymm1)
		CLIP_F32S_AVX2(ymm30,  ymm0, ymm1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_DOWNSCALE_1x64F:
	{
		__m256 selector1 = _mm256_setzero_ps();
		__m256 selector2 = _mm256_setzero_ps();
		__m256 selector3 = _mm256_setzero_ps();
		__m256 selector4 = _mm256_setzero_ps();
		__m256 selector5 = _mm256_setzero_ps();
		__m256 selector6 = _mm256_setzero_ps();
		__m256 selector7 = _mm256_setzero_ps();
		__m256 selector8 = _mm256_setzero_ps();

		__m256 zero_point0 = _mm256_setzero_ps();
		__m256 zero_point1 = _mm256_setzero_ps();
		__m256 zero_point2 = _mm256_setzero_ps();
		__m256 zero_point3 = _mm256_setzero_ps();
		__m256 zero_point4 = _mm256_setzero_ps();
		__m256 zero_point5 = _mm256_setzero_ps();
		__m256 zero_point6 = _mm256_setzero_ps();
		__m256 zero_point7 = _mm256_setzero_ps();

		bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector6 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector7 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector8 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}
		if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			if ( is_bf16 == TRUE )
			{
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point4)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point5)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point6)
				BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point7)
			}
			else
			{
				zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point6 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
				zero_point7 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
			}
		}
		if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 8 ), k1 );
				selector2 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 8 ), k2 );
				selector3 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 8 ), k3 );
				selector4 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 8 ), k4 );
				selector5 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 4 * 8 ), k5 );
				selector6 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 5 * 8 ), k6 );
				selector7 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 6 * 8 ), k7 );
				selector8 = _mm256_maskload_ps( ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 7 * 8 ), k8 );
			}
			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				if ( is_bf16 == TRUE )
				{
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point0, 0, k1)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point1, 1, k2)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point2, 2, k3)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point3, 3, k4)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point4, 4, k5)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point5, 5, k6)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point6, 6, k7)
					BF16_F32_ZP_VECTOR_LOAD_AVX2_MASK(zero_point7, 7, k8)
				}
				else
				{
					zero_point0 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 8 ), k1 );
					zero_point1 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 1 * 8 ), k2 );
					zero_point2 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 2 * 8 ), k3 );
					zero_point3 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 3 * 8 ), k4 );
					zero_point4 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 4 * 8 ), k5 );
					zero_point5 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 5 * 8 ), k6 );
					zero_point6 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 6 * 8 ), k7 );
					zero_point7 = _mm256_maskload_ps( (float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 7 * 8 ), k8 );
				}
			}

			F32_SCL_MULRND_AVX2( ymm16,  selector1, zero_point0 )
			F32_SCL_MULRND_AVX2( ymm18,  selector2, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm20,  selector3, zero_point2 )
			F32_SCL_MULRND_AVX2( ymm22,  selector4, zero_point3 )
			F32_SCL_MULRND_AVX2( ymm24,  selector5, zero_point4 )
			F32_SCL_MULRND_AVX2( ymm26,  selector6, zero_point5 )
			F32_SCL_MULRND_AVX2( ymm28,  selector7, zero_point6 )
			F32_SCL_MULRND_AVX2( ymm30,  selector8, zero_point7 )
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.

			F32_SCL_MULRND_AVX2( ymm16,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm18,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm20,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm22,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm24,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm26,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm28,  selector1, zero_point1 )
			F32_SCL_MULRND_AVX2( ymm30,  selector1, zero_point1 )
		}
		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_MATRIX_ADD_1x64F:
	{
		__m256 selector1;
		__m256 selector2;
		__m256 selector3;
		__m256 selector4;
		__m256 selector5;
		__m256 selector6;
		__m256 selector7;
		__m256 selector8;

		__m256 scl_fctr1 = _mm256_setzero_ps();
		__m256 scl_fctr2 = _mm256_setzero_ps();
		__m256 scl_fctr3 = _mm256_setzero_ps();
		__m256 scl_fctr4 = _mm256_setzero_ps();
		__m256 scl_fctr5 = _mm256_setzero_ps();
		__m256 scl_fctr6 = _mm256_setzero_ps();
		__m256 scl_fctr7 = _mm256_setzero_ps();
		__m256 scl_fctr8 = _mm256_setzero_ps();

		bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

		// Even though different registers are used for scalar in column and
		// row major case, all those registers will contain the same value.
		// For column major, if m==1, then it means n=1 and scale_factor_len=1.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			scl_fctr1 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr2 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr3 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr4 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr5 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr6 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr7 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr8 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}
		else
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				scl_fctr1 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 8 ), k1 );
				scl_fctr2 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 8 ), k2 );
				scl_fctr3 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 8 ), k3 );
				scl_fctr4 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 8 ), k4 );
				scl_fctr5 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 4 * 8 ), k5 );
				scl_fctr6 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 5 * 8 ), k6 );
				scl_fctr7 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 6 * 8 ), k7 );
				scl_fctr8 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 7 * 8 ), k8 );
			}
		}
		if ( is_bf16 == TRUE )
        {
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector1,scl_fctr1,0, k1)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector2,scl_fctr2,1, k2)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector3,scl_fctr3,2, k3)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector4,scl_fctr4,3, k4)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector5,scl_fctr5,4, k5)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector6,scl_fctr6,5, k6)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector7,scl_fctr7,6, k7)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector8,scl_fctr8,7, k8)
			}
			else
			{
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector1,scl_fctr1,0, k1)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector2,scl_fctr1,1, k2)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector3,scl_fctr1,2, k3)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector4,scl_fctr1,3, k4)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector5,scl_fctr1,4, k5)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector6,scl_fctr1,5, k6)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector7,scl_fctr1,6, k7)
				BF16_F32_MATRIX_ADD_GEMV_MASK(matptr,selector8,scl_fctr1,7, k8)
			}
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				selector1 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j ), k1 );
				selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 8 ), k2 );
				selector2 = _mm256_mul_ps( selector2, scl_fctr2 );
				selector3 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 16 ), k3 );
				selector3 = _mm256_mul_ps( selector3, scl_fctr3 );
				selector4 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 24 ), k4 );
				selector4 = _mm256_mul_ps( selector4, scl_fctr4 );
				selector5 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 32 ) , k5 );
				selector5 = _mm256_mul_ps( selector5, scl_fctr5 );
				selector6 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 40 ), k6 );
				selector6 = _mm256_mul_ps( selector6, scl_fctr6 );
				selector7 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 48 ), k7 );
				selector7 = _mm256_mul_ps( selector7, scl_fctr7 );
				selector8 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 56 ), k8 );
				selector8 = _mm256_mul_ps( selector8, scl_fctr8 );
			}
			else
			{
				selector1 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j ), k1 );
				selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 8 ), k2 );
				selector2 = _mm256_mul_ps( selector2, scl_fctr1 );
				selector3 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 16 ), k3 );
				selector3 = _mm256_mul_ps( selector3, scl_fctr1 );
				selector4 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 24 ), k4 );
				selector4 = _mm256_mul_ps( selector4, scl_fctr1 );
				selector5 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 32 ) , k5 );
				selector5 = _mm256_mul_ps( selector5, scl_fctr1 );
				selector6 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 40 ), k6 );
				selector6 = _mm256_mul_ps( selector6, scl_fctr1 );
				selector7 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 48 ), k7 );
				selector7 = _mm256_mul_ps( selector7, scl_fctr1 );
				selector8 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 56 ), k8 );
				selector8 = _mm256_mul_ps( selector8, scl_fctr1 );
			}
		}
		ymm16 = _mm256_add_ps( selector1, ymm16 );
		ymm18 = _mm256_add_ps( selector2, ymm18 );
		ymm20 = _mm256_add_ps( selector3, ymm20 );
		ymm22 = _mm256_add_ps( selector4, ymm22 );
		ymm24 = _mm256_add_ps( selector5, ymm24 );
		ymm26 = _mm256_add_ps( selector6, ymm26 );
		ymm28 = _mm256_add_ps( selector7, ymm28 );
		ymm30 = _mm256_add_ps( selector8, ymm30 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_MATRIX_MUL_1x64F:
	{
		__m256 selector1;
		__m256 selector2;
		__m256 selector3;
		__m256 selector4;
		__m256 selector5;
		__m256 selector6;
		__m256 selector7;
		__m256 selector8;


		__m256 scl_fctr1 = _mm256_setzero_ps();
		__m256 scl_fctr2 = _mm256_setzero_ps();
		__m256 scl_fctr3 = _mm256_setzero_ps();
		__m256 scl_fctr4 = _mm256_setzero_ps();
		__m256 scl_fctr5 = _mm256_setzero_ps();
		__m256 scl_fctr6 = _mm256_setzero_ps();
		__m256 scl_fctr7 = _mm256_setzero_ps();
		__m256 scl_fctr8 = _mm256_setzero_ps();

		bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

		// Even though different registers are used for scalar in column and
		// row major case, all those registers will contain the same value.
		// For column major, if m==1, then it means n=1 and scale_factor_len=1.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			scl_fctr1 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr2 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr3 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr4 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr5 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr6 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr7 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scl_fctr8 =
				_mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}
		else
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				scl_fctr1 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 8 ), k1 );
				scl_fctr2 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 8 ), k2 );
				scl_fctr3 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 8 ), k3 );
				scl_fctr4 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 8 ), k4 );
				scl_fctr5 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 4 * 8 ), k5 );
				scl_fctr6 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 5 * 8 ), k6 );
				scl_fctr7 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 6 * 8 ), k7 );
				scl_fctr8 =
					_mm256_maskload_ps(
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 7 * 8 ), k8 );
			}
		}
		if ( is_bf16 == TRUE )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1; \

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector1,scl_fctr1,0, k1)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector2,scl_fctr2,1, k2)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector3,scl_fctr3,2, k3)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector4,scl_fctr4,3, k4)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector5,scl_fctr5,4, k5)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector6,scl_fctr6,5, k6)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector7,scl_fctr7,6, k7)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector8,scl_fctr8,7, k8)
			}
			else
			{
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector1,scl_fctr1,0, k1)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector2,scl_fctr1,1, k2)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector3,scl_fctr1,2, k3)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector4,scl_fctr1,3, k4)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector5,scl_fctr1,4, k5)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector6,scl_fctr1,5, k6)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector7,scl_fctr1,6, k7)
				BF16_F32_MATRIX_MUL_GEMV_MASK(matptr,selector8,scl_fctr1,7, k8)
			}
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				selector1 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j ), k1 );
				selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 8 ), k2 );
				selector2 = _mm256_mul_ps( selector2, scl_fctr2 );
				selector3 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 16 ), k3 );
				selector3 = _mm256_mul_ps( selector3, scl_fctr3 );
				selector4 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 24 ), k4 );
				selector4 = _mm256_mul_ps( selector4, scl_fctr4 );
				selector5 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 32 ) , k5 );
				selector5 = _mm256_mul_ps( selector5, scl_fctr5 );
				selector6 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 40 ), k6 );
				selector6 = _mm256_mul_ps( selector6, scl_fctr6 );
				selector7 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 48 ), k7 );
				selector7 = _mm256_mul_ps( selector7, scl_fctr7 );
				selector8 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 56 ), k8 );
				selector8 = _mm256_mul_ps( selector8, scl_fctr8 );
			}
			else
			{
				selector1 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j ), k1 );
				selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 8 ), k2 );
				selector2 = _mm256_mul_ps( selector2, scl_fctr1 );
				selector3 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 16 ), k3 );
				selector3 = _mm256_mul_ps( selector3, scl_fctr1 );
				selector4 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 24 ), k4 );
				selector4 = _mm256_mul_ps( selector4, scl_fctr1 );
				selector5 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 32 ) , k5 );
				selector5 = _mm256_mul_ps( selector5, scl_fctr1 );
				selector6 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 40 ), k6 );
				selector6 = _mm256_mul_ps( selector6, scl_fctr1 );
				selector7 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 48 ), k7 );
				selector7 = _mm256_mul_ps( selector7, scl_fctr1 );
				selector8 =
					_mm256_maskload_ps( (matptr + post_ops_attr.post_op_c_j + 56 ), k8 );
				selector8 = _mm256_mul_ps( selector8, scl_fctr1 );
			}
		}

		ymm16 = _mm256_mul_ps( selector1, ymm16 );
		ymm18 = _mm256_mul_ps( selector2, ymm18 );
		ymm20 = _mm256_mul_ps( selector3, ymm20 );
		ymm22 = _mm256_mul_ps( selector4, ymm22 );
		ymm24 = _mm256_mul_ps( selector5, ymm24 );
		ymm26 = _mm256_mul_ps( selector6, ymm26 );
		ymm28 = _mm256_mul_ps( selector7, ymm28 );
		ymm30 = _mm256_mul_ps( selector8, ymm30 );


		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_SWISH_1x64F:
	{
		ymm7 =
			_mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
	__m256 z, dn;
	__m256i ex_out;

		SWISH_F32_AVX2_DEF(ymm16, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm18, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm20, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm22, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm24, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm26, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm28, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
		SWISH_F32_AVX2_DEF(ymm30, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_TANH_1x64F:
	{
		__m256 dn;
		__m256i q;

		TANH_F32S_AVX2(ymm16, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm18, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm20, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm22, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm24, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm26, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm28, ymm0, ymm1, ymm2, ymm3, dn, q)
		TANH_F32S_AVX2(ymm30, ymm0, ymm1, ymm2, ymm3, dn, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_SIGMOID_1x64F:
	{
		__m256 z, dn;
		__m256i ex_out;

		SIGMOID_F32_AVX2_DEF(ymm16, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm18, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm20, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm22, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm24, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm26, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm28, ymm1, ymm2, ymm3, z, dn, ex_out)
		SIGMOID_F32_AVX2_DEF(ymm30, ymm1, ymm2, ymm3, z, dn, ex_out)


		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_1x64F_DISABLE:
	{
		uint32_t tlsb, rounded;
		int i;
		bfloat16* dest;

		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		{
			MASK_STORE_F32_BF16_YMM(ymm16, 0, 0, k1);
			MASK_STORE_F32_BF16_YMM(ymm18, 0, 1, k2);
			MASK_STORE_F32_BF16_YMM(ymm20, 0, 2, k3);
			MASK_STORE_F32_BF16_YMM(ymm22, 0, 3, k4);
			MASK_STORE_F32_BF16_YMM(ymm24, 0, 4, k5);
			MASK_STORE_F32_BF16_YMM(ymm26, 0, 5, k6);
			MASK_STORE_F32_BF16_YMM(ymm28, 0, 6, k7);
			MASK_STORE_F32_BF16_YMM(ymm30, 0, 7, k8);
		}
		else
		{
			_mm256_maskstore_ps(c_use, k1, ymm16);
			_mm256_maskstore_ps((c_use + 8), k2, ymm18);
			_mm256_maskstore_ps((c_use + 16), k3, ymm20);
			_mm256_maskstore_ps((c_use + 24), k4, ymm22);
			_mm256_maskstore_ps((c_use + 32), k5, ymm24);
			_mm256_maskstore_ps((c_use + 40), k6, ymm26);
			_mm256_maskstore_ps((c_use + 48), k7, ymm28);
			_mm256_maskstore_ps((c_use + 56), k8, ymm30);

		}
		post_ops_attr.post_op_c_j += NR;
	}
	} // jr loop
}

#endif // BLIS_ADDON_LPGEMM
