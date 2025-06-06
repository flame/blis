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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32.h"

LPGEMV_M_EQ1_KERN( float, float, float, f32f32f32of32 )
{
	static void *post_ops_labels[] =
		{
			&&POST_OPS_6x64F_DISABLE,
			&&POST_OPS_BIAS_6x64F,
			&&POST_OPS_RELU_6x64F,
			&&POST_OPS_RELU_SCALE_6x64F,
			&&POST_OPS_GELU_TANH_6x64F,
			&&POST_OPS_GELU_ERF_6x64F,
			&&POST_OPS_CLIP_6x64F,
			&&POST_OPS_DOWNSCALE_6x64F,
			&&POST_OPS_MATRIX_ADD_6x64F,
			&&POST_OPS_SWISH_6x64F,
			&&POST_OPS_MATRIX_MUL_6x64F,
			&&POST_OPS_TANH_6x64F,
			&&POST_OPS_SIGMOID_6x64F
		};

	// Strides are updated based on matrix packing/reordering.
	const float *a_use = NULL;
	const float *b_use = NULL;
	float *c_use = NULL;
	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	for (dim_t jr = 0; jr < n0; jr += NR)
	{
		dim_t nr0 = bli_min((n0 - jr), NR);
		c_use = c + jr;
		__mmask16 k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF, k4 = 0xFFFF;

		if (nr0 < NR)
		{
			__mmask16 k = (0xFFFF >> (16 - (nr0 & 0x0F)));
			if (nr0 >= 48)
			{
				k4 = k;
			}
			else if (nr0 >= 32)
			{
				k3 = k;
				k4 = 0;
			}
			else if (nr0 >= 16)
			{
				k2 = k;
				k3 = k4 = 0;
			}
			else
			{
				k1 = k;
				k2 = k3 = k4 = 0;
			}
		}

		__m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
		__m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14;
		__m512 zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
		__m512 zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28;
		__m512 zmm29, zmm30, zmm31;

		// zero the accumulator registers
		ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
		ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
		ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
		ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);

		//Zero out registers used for mask load to avoid warnings
		ZERO_ACC_ZMM_4_REG(zmm0, zmm1, zmm2, zmm3);
		ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm27);
		ZERO_ACC_ZMM_4_REG(zmm28, zmm29, zmm30, zmm31);

		//_mm_prefetch( (MR X NR) from C
		_mm_prefetch((c_use + 0 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 16 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 32 * rs_c), _MM_HINT_T0);
		_mm_prefetch((c_use + 64 * rs_c), _MM_HINT_T0);

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			dim_t kc0 = bli_min((k - pc), KC);
			uint64_t k_iter = kc0 / 4;
			uint64_t k_rem = kc0 % 4;
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
				_mm_prefetch((b_use + 4 * rs_b_use), _MM_HINT_T0);
				//Using mask loads to avoid writing fringe kernels

				//Load first 4x16 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_ps(k1, b_use);
				zmm1 = _mm512_maskz_loadu_ps(k1, b_use + rs_b_use);
				zmm2 = _mm512_maskz_loadu_ps(k1, b_use + 2 * rs_b_use);
				zmm3 = _mm512_maskz_loadu_ps(k1, b_use + 3 * rs_b_use);
				b_use += 16;

				//Broadcast col0 - col3 element of A
				zmm4 = _mm512_set1_ps(*(a_use));	 // broadcast c0
				zmm5 = _mm512_set1_ps(*(a_use + 1)); // broadcast c1
				zmm6 = _mm512_set1_ps(*(a_use + 2)); // broadcast c2
				zmm7 = _mm512_set1_ps(*(a_use + 3)); // broadcast c3

				//Load second 4x16 tile from row 0-3
				zmm24 = _mm512_maskz_loadu_ps(k2, b_use);
				zmm25 = _mm512_maskz_loadu_ps(k2, b_use + rs_b_use);
				zmm26 = _mm512_maskz_loadu_ps(k2, b_use + 2 * rs_b_use);
				zmm27 = _mm512_maskz_loadu_ps(k2, b_use + 3 * rs_b_use);
				b_use += 16;

				zmm8 = _mm512_fmadd_ps(zmm0, zmm4, zmm8);
				zmm9 = _mm512_fmadd_ps(zmm1, zmm5, zmm9);
				zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
				zmm11 = _mm512_fmadd_ps(zmm3, zmm7, zmm11);

				//Load third 4x16 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_ps(k3, b_use);
				zmm1 = _mm512_maskz_loadu_ps(k3, b_use + rs_b_use);
				zmm2 = _mm512_maskz_loadu_ps(k3, b_use + 2 * rs_b_use);
				zmm3 = _mm512_maskz_loadu_ps(k3, b_use + 3 * rs_b_use);
				b_use += 16;

				zmm12 = _mm512_fmadd_ps(zmm24, zmm4, zmm12);
				zmm13 = _mm512_fmadd_ps(zmm25, zmm5, zmm13);
				zmm14 = _mm512_fmadd_ps(zmm26, zmm6, zmm14);
				zmm15 = _mm512_fmadd_ps(zmm27, zmm7, zmm15);

				//Load fourth 4x16 tile from row 0-3
				zmm28 = _mm512_maskz_loadu_ps(k4, b_use);
				zmm29 = _mm512_maskz_loadu_ps(k4, b_use + rs_b_use);
				zmm30 = _mm512_maskz_loadu_ps(k4, b_use + 2 * rs_b_use);
				zmm31 = _mm512_maskz_loadu_ps(k4, b_use + 3 * rs_b_use);

				zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
				zmm17 = _mm512_fmadd_ps(zmm1, zmm5, zmm17);
				zmm18 = _mm512_fmadd_ps(zmm2, zmm6, zmm18);
				zmm19 = _mm512_fmadd_ps(zmm3, zmm7, zmm19);

				zmm20 = _mm512_fmadd_ps(zmm28, zmm4, zmm20);
				zmm21 = _mm512_fmadd_ps(zmm29, zmm5, zmm21);
				zmm22 = _mm512_fmadd_ps(zmm30, zmm6, zmm22);
				zmm23 = _mm512_fmadd_ps(zmm31, zmm7, zmm23);

				b_use -= 48; // move b point back to start of KCXNR
				b_use += (4 * rs_b_use);
				a_use += 4; // move a pointer to next col
			}				// kloop

			for (dim_t kr = 0; kr < k_rem; kr++)
			{
				//Load 64 elements from a row of B
				zmm0 = _mm512_maskz_loadu_ps(k1, b_use);
				zmm1 = _mm512_maskz_loadu_ps(k2, b_use + 16);
				zmm2 = _mm512_maskz_loadu_ps(k3, b_use + 32);
				zmm3 = _mm512_maskz_loadu_ps(k4, b_use + 48);

				//Broadcast col0 elements of 12 rows of A
				zmm4 = _mm512_set1_ps(*(a_use)); // broadcast c0r0

				zmm8 = _mm512_fmadd_ps(zmm0, zmm4, zmm8);
				zmm12 = _mm512_fmadd_ps(zmm1, zmm4, zmm12);
				zmm16 = _mm512_fmadd_ps(zmm2, zmm4, zmm16);
				zmm20 = _mm512_fmadd_ps(zmm3, zmm4, zmm20);

				b_use += rs_b_use; // move b pointer to next row
				a_use++;		 // move a pointer to next col
			}					// kloop
		}						// kc loop

		//SUMUP K untoll output
		zmm8 = _mm512_add_ps(zmm9, zmm8);
		zmm10 = _mm512_add_ps(zmm11, zmm10);
		zmm8 = _mm512_add_ps(zmm10, zmm8); // 16 outputs

		zmm12 = _mm512_add_ps(zmm13, zmm12);
		zmm14 = _mm512_add_ps(zmm15, zmm14);
		zmm12 = _mm512_add_ps(zmm14, zmm12); // 16 outputs

		zmm16 = _mm512_add_ps(zmm17, zmm16);
		zmm18 = _mm512_add_ps(zmm19, zmm18);
		zmm16 = _mm512_add_ps(zmm18, zmm16); // 16 outputs

		zmm20 = _mm512_add_ps(zmm21, zmm20);
		zmm22 = _mm512_add_ps(zmm23, zmm22);
		zmm20 = _mm512_add_ps(zmm22, zmm20); // 16 outputs

		//Mulitply A*B output with alpha
		zmm0 = _mm512_set1_ps(alpha);
		zmm8 = _mm512_mul_ps(zmm0, zmm8);
		zmm12 = _mm512_mul_ps(zmm0, zmm12);
		zmm16 = _mm512_mul_ps(zmm0, zmm16);
		zmm20 = _mm512_mul_ps(zmm0, zmm20);

		if (beta != 0)
		{
			zmm3 = _mm512_set1_ps(beta);

			if ( ( post_ops_attr.buf_downscale != NULL ) &&
			     ( post_ops_attr.is_first_k == TRUE ) )
		    {
              BF16_F32_BETA_OP_NLT16F_MASK(k1, zmm8, 0, 0,  zmm0,zmm3);
              BF16_F32_BETA_OP_NLT16F_MASK(k2, zmm12, 0, 1,  zmm1,zmm3);
              BF16_F32_BETA_OP_NLT16F_MASK(k3, zmm16, 0, 2,  zmm2,zmm3);
              BF16_F32_BETA_OP_NLT16F_MASK(k4, zmm20, 0, 3,  zmm3,zmm3);
            }
			else
			{
			  const float *_cbuf = c_use;
			  // load c and multiply with beta and
			  // add to accumulator and store back

			  zmm0 = _mm512_maskz_loadu_ps(k1, _cbuf);
			  zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);

			  zmm1 = _mm512_maskz_loadu_ps(k2, (_cbuf + 16));
			  zmm12 = _mm512_fmadd_ps(zmm1, zmm3, zmm12);

			  zmm2 = _mm512_maskz_loadu_ps(k3, (_cbuf + 32));
			  zmm16 = _mm512_fmadd_ps(zmm2, zmm3, zmm16);

			  zmm4 = _mm512_maskz_loadu_ps(k4, (_cbuf + 48));
			  zmm20 = _mm512_fmadd_ps(zmm4, zmm3, zmm20);

			}

		}

		// Post Ops
		post_ops_attr.is_last_k = TRUE;
		lpgemm_post_op *post_ops_list_temp = post_op;
		POST_OP_LABEL_LASTK_SAFE_JUMP

	POST_OPS_BIAS_6x64F:
	{
		if ((*(char *)post_ops_list_temp->op_args2 == 'r') ||
			(*(char *)post_ops_list_temp->op_args2 == 'R'))
		{
			if( post_ops_list_temp->stor_type == BF16 )
			{
			  BF16_F32_BIAS_LOAD(zmm9, k1, 0)
			  BF16_F32_BIAS_LOAD(zmm10, k2, 1)
			  BF16_F32_BIAS_LOAD(zmm13, k3, 2)
			  BF16_F32_BIAS_LOAD(zmm14, k3, 3)
			}
			else
			{
				float* bias_ptr = (float *)post_ops_list_temp->op_args1 +
										post_ops_attr.post_op_c_j;
				zmm9 = _mm512_maskz_loadu_ps(k1, bias_ptr + (0 * 16));

				zmm10 =	_mm512_maskz_loadu_ps(k2, bias_ptr + (1 * 16));

				zmm13 =	_mm512_maskz_loadu_ps(k3, bias_ptr + (2 * 16));

				zmm14 =	_mm512_maskz_loadu_ps(k4, bias_ptr + (3 * 16));
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
			if ( post_ops_list_temp->stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(zmm9, bias_mask, 0)
			}
			else
			{
				float bias = (*((float *)post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0));

				zmm9 =	_mm512_set1_ps(bias);
			}
				zmm10 = zmm13 = zmm14 = zmm9;
		}
		// c[0,0-15]
		zmm8 = _mm512_add_ps(zmm9, zmm8);
		zmm12 = _mm512_add_ps(zmm10, zmm12);
		zmm16 = _mm512_add_ps(zmm13, zmm16);
		zmm20 = _mm512_add_ps(zmm14, zmm20);
		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_RELU_6x64F:
	{
		zmm1 = _mm512_setzero_ps();

		// c[0,0-15]
		zmm8 = _mm512_max_ps(zmm1, zmm8);
		zmm12 = _mm512_max_ps(zmm1, zmm12);
		zmm16 = _mm512_max_ps(zmm1, zmm16);
		zmm20 = _mm512_max_ps(zmm1, zmm20);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_RELU_SCALE_6x64F:
	{
		zmm1 = _mm512_setzero_ps();
		zmm2 =
			_mm512_set1_ps(*((float *)post_ops_list_temp->op_args2));

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32S_AVX512(zmm8)
		RELU_SCALE_OP_F32S_AVX512(zmm12)
		RELU_SCALE_OP_F32S_AVX512(zmm16)
		RELU_SCALE_OP_F32S_AVX512(zmm20)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_GELU_TANH_6x64F:
	{
		__m512i zmm6;
		// c[0, 0-15]
		GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)
		GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)
		GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)
		GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_GELU_ERF_6x64F:
	{
		// c[0, 0-15]
		GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)
		GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)
		GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)
		GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_CLIP_6x64F:
	{
		zmm0 = _mm512_set1_ps(*(float *)post_ops_list_temp->op_args2);
		zmm1 = _mm512_set1_ps(*(float *)post_ops_list_temp->op_args3);

		// c[0, 0-15]
		CLIP_F32S_AVX512(zmm8, zmm0, zmm1)
		CLIP_F32S_AVX512(zmm12, zmm0, zmm1)
		CLIP_F32S_AVX512(zmm16, zmm0, zmm1)
		CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_DOWNSCALE_6x64F:
	{
		__m512 selector1 = _mm512_setzero_ps();
		__m512 selector2 = _mm512_setzero_ps();
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

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
			if ( is_bf16 == TRUE )
            {
              BF16_F32_ZP_BCST(zero_point0,0, k1)
              BF16_F32_ZP_BCST(zero_point1,1, k2)
              BF16_F32_ZP_BCST(zero_point2,2, k3)
              BF16_F32_ZP_BCST(zero_point3,3, k4)
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
				selector1 = _mm512_maskz_loadu_ps( k1, ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 = _mm512_maskz_loadu_ps( k2, ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 = _mm512_maskz_loadu_ps( k3, ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				selector4 = _mm512_maskz_loadu_ps( k4, ( float* )
								post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}
			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				if ( is_bf16 == TRUE )
                {
                  BF16_F32_ZP_LOAD(zero_point0, k1, 0)
                  BF16_F32_ZP_LOAD(zero_point1, k2, 1)
                  BF16_F32_ZP_LOAD(zero_point2, k3, 2)
                  BF16_F32_ZP_LOAD(zero_point3, k4, 3)
                }
                else
                {
                  zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
                  zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
                  zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
                  zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_j + ( 3 * 16 ) );
                }
			}
			//c[0, 0-15]
			F32_SCL_MULRND(zmm8, selector1, zero_point0);

			//c[0, 16-31]
			F32_SCL_MULRND(zmm12, selector2, zero_point1);

			//c[0, 32-47]
			F32_SCL_MULRND(zmm16, selector3, zero_point2);

			//c[0, 48-63]
			F32_SCL_MULRND(zmm20, selector4, zero_point3);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.

			//c[0, 0-15]
			F32_SCL_MULRND(zmm8, selector1, zero_point0);

			//c[0, 16-31]
			F32_SCL_MULRND(zmm12, selector1, zero_point0);

			//c[0, 32-47]
			F32_SCL_MULRND(zmm16, selector1, zero_point0);

			//c[0, 48-63]
			F32_SCL_MULRND(zmm20, selector1, zero_point0);
		}
		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_MATRIX_ADD_6x64F:
	{
		__m512 selector1;
		__m512 selector2;
		__m512 selector3;
		__m512 selector4;

		__m512 scl_fctr1 = _mm512_setzero_ps();
		__m512 scl_fctr2 = _mm512_setzero_ps();
		__m512 scl_fctr3 = _mm512_setzero_ps();
		__m512 scl_fctr4 = _mm512_setzero_ps();

		bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

		// Even though different registers are used for scalar in column and
		// row major case, all those registers will contain the same value.
		// For column major, if m==1, then it means n=1 and scale_factor_len=1.
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
					_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k4,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}
		}
		if ( is_bf16 == TRUE )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k1,selector1,scl_fctr1,post_ops_attr.post_op_c_j,0)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k2,selector2,scl_fctr2,post_ops_attr.post_op_c_j,1)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k3,selector3,scl_fctr3,post_ops_attr.post_op_c_j,2)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k4,selector4,scl_fctr4,post_ops_attr.post_op_c_j,3)
			}
			else
			{
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k1,selector1,scl_fctr1,post_ops_attr.post_op_c_j,0)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k2,selector2,scl_fctr1,post_ops_attr.post_op_c_j,1)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k3,selector3,scl_fctr1,post_ops_attr.post_op_c_j,2)
				BF16_F32_MATRIX_ADD_LOAD_GEMV(
					k4,selector4,scl_fctr1,post_ops_attr.post_op_c_j,3)
			}
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				selector1 =
					_mm512_maskz_loadu_ps(k1, (matptr + post_ops_attr.post_op_c_j ) );
				selector1 = _mm512_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_j + 16 ) );
				selector2 = _mm512_mul_ps( selector2, scl_fctr2 );
				selector3 =
					_mm512_maskz_loadu_ps(k3, (matptr + post_ops_attr.post_op_c_j + 32 ) );
				selector3 = _mm512_mul_ps( selector3, scl_fctr3 );
				selector4 =
					_mm512_maskz_loadu_ps(k4, (matptr + post_ops_attr.post_op_c_j + 48 ) );
				selector4 = _mm512_mul_ps( selector4, scl_fctr4 );
			}
			else
			{
				selector1 =
					_mm512_maskz_loadu_ps(k1, (matptr + post_ops_attr.post_op_c_j ) );
				selector1 = _mm512_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_j + 16 ) );
				selector2 = _mm512_mul_ps( selector2, scl_fctr1 );
				selector3 =
					_mm512_maskz_loadu_ps(k3, (matptr + post_ops_attr.post_op_c_j + 32 ) );
				selector3 = _mm512_mul_ps( selector3, scl_fctr1 );
				selector4 =
					_mm512_maskz_loadu_ps(k4, (matptr + post_ops_attr.post_op_c_j + 48 ) );
				selector4 = _mm512_mul_ps( selector4, scl_fctr1 );
			}
		}
		zmm8  = _mm512_add_ps( selector1, zmm8  );
		zmm12 = _mm512_add_ps( selector2, zmm12 );
		zmm16 = _mm512_add_ps( selector3, zmm16 );
		zmm20 = _mm512_add_ps( selector4, zmm20 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_MATRIX_MUL_6x64F:
	{
		__m512 selector1;
		__m512 selector2;
		__m512 selector3;
		__m512 selector4;


		__m512 scl_fctr1 = _mm512_setzero_ps();
		__m512 scl_fctr2 = _mm512_setzero_ps();
		__m512 scl_fctr3 = _mm512_setzero_ps();
		__m512 scl_fctr4 = _mm512_setzero_ps();

		bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

		// Even though different registers are used for scalar in column and
		// row major case, all those registers will contain the same value.
		// For column major, if m==1, then it means n=1 and scale_factor_len=1.
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
					_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scl_fctr2 =
					_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scl_fctr3 =
					_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scl_fctr4 =
					_mm512_maskz_loadu_ps( k4,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}
		}
		if( is_bf16 == TRUE )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
			  	bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k1, selector1, scl_fctr1, post_ops_attr.post_op_c_j,0)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k2, selector2, scl_fctr2, post_ops_attr.post_op_c_j,1)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k3, selector3, scl_fctr3, post_ops_attr.post_op_c_j,2)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k4, selector4, scl_fctr4, post_ops_attr.post_op_c_j,3)
			}
			else
			{
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k1, selector1, scl_fctr1, post_ops_attr.post_op_c_j,0)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k2, selector2, scl_fctr1, post_ops_attr.post_op_c_j,1)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k3, selector3, scl_fctr1, post_ops_attr.post_op_c_j,2)
				BF16_F32_MATRIX_MUL_LOAD_GEMV(
					k4, selector4, scl_fctr1, post_ops_attr.post_op_c_j,3)
			}
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				selector1 =
					_mm512_maskz_loadu_ps(k1, (matptr + post_ops_attr.post_op_c_j ) );
				selector1 = _mm512_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_j + 16 ) );
				selector2 = _mm512_mul_ps( selector2, scl_fctr2 );
				selector3 =
					_mm512_maskz_loadu_ps(k3, (matptr + post_ops_attr.post_op_c_j + 32 ) );
				selector3 = _mm512_mul_ps( selector3, scl_fctr3 );
				selector4 =
					_mm512_maskz_loadu_ps(k4, (matptr + post_ops_attr.post_op_c_j + 48 ) );
				selector4 = _mm512_mul_ps( selector4, scl_fctr4 );
			}
			else
			{
				selector1 =
					_mm512_maskz_loadu_ps(k1, (matptr + post_ops_attr.post_op_c_j ) );
				selector1 = _mm512_mul_ps( selector1, scl_fctr1 );
				selector2 =
					_mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_j + 16 ) );
				selector2 = _mm512_mul_ps( selector2, scl_fctr1 );
				selector3 =
					_mm512_maskz_loadu_ps(k3, (matptr + post_ops_attr.post_op_c_j + 32 ) );
				selector3 = _mm512_mul_ps( selector3, scl_fctr1 );
				selector4 =
					_mm512_maskz_loadu_ps(k4, (matptr + post_ops_attr.post_op_c_j + 48 ) );
				selector4 = _mm512_mul_ps( selector4, scl_fctr1 );
			}
		}

		zmm8  = _mm512_mul_ps( selector1, zmm8  );
		zmm12 = _mm512_mul_ps( selector2, zmm12 );
		zmm16 = _mm512_mul_ps( selector3, zmm16 );
		zmm20 = _mm512_mul_ps( selector4, zmm20 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_SWISH_6x64F:
	{
		zmm7 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_TANH_6x64F:
	{
		__m512i zmm6;
		// c[0, 0-15]
		TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)
		TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)
		TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)
		TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_SIGMOID_6x64F:
	{
		__m512i ex_out;

		// c[0, 0-15]
		SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[1, 0-15]
		SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[2, 0-15]
		SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		// c[3, 0-15]
		SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
	POST_OPS_6x64F_DISABLE:
	{
		if ( post_ops_attr.buf_downscale != NULL )
		{
			_mm256_mask_storeu_epi16
			(
			( bfloat16* )post_ops_attr.buf_downscale +
			post_ops_attr.post_op_c_j + ( 0 * 16 ),
			k1, (__m256i) _mm512_cvtneps_pbh( zmm8 )
			);

			_mm256_mask_storeu_epi16
			(
			( bfloat16* )post_ops_attr.buf_downscale +
			post_ops_attr.post_op_c_j + ( 1 * 16 ),
			k2, (__m256i) _mm512_cvtneps_pbh( zmm12 )
			);

			_mm256_mask_storeu_epi16
			(
			( bfloat16* )post_ops_attr.buf_downscale +
			post_ops_attr.post_op_c_j + ( 2 * 16 ),
			k3, (__m256i) _mm512_cvtneps_pbh( zmm16 )
			);

			_mm256_mask_storeu_epi16
			(
			( bfloat16* )post_ops_attr.buf_downscale +
			post_ops_attr.post_op_c_j + ( 3 * 16 ),
			k4, (__m256i) _mm512_cvtneps_pbh( zmm20 )
			);
		}
		else
		{
		  _mm512_mask_storeu_ps(c_use, k1, zmm8);
		  _mm512_mask_storeu_ps((c_use + 16), k2, zmm12);
		  _mm512_mask_storeu_ps((c_use + 32), k3, zmm16);
		  _mm512_mask_storeu_ps((c_use + 48), k4, zmm20);

		}
		post_ops_attr.post_op_c_j += NR;
	}
	} // jr loop
}

#endif // BLIS_ADDON_LPGEMM
