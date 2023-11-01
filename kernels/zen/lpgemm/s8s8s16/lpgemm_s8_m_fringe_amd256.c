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

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "../u8s8s16/lpgemm_s16_kern_macros.h"

// 4x32 int8o16 kernel
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int16_t,s8s8s16o16_4x32)
{
	dim_t NR = 32;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_4x32_DISABLE,
			&&POST_OPS_BIAS_4x32,
			&&POST_OPS_RELU_4x32,
			&&POST_OPS_RELU_SCALE_4x32,
			&&POST_OPS_GELU_TANH_4x32,
			&&POST_OPS_GELU_ERF_4x32,
			&&POST_OPS_CLIP_4x32,
			&&POST_OPS_DOWNSCALE_4x32
		};

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;
	__m256i b1;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i a_int32_1;
	__m256i inter_vec[4];

    uint8_t cvt_uint8 = 128;
	__m256i vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();
	__m256i c_int16_0p1 = _mm256_setzero_si256();

	__m256i c_int16_1p0 = _mm256_setzero_si256();
	__m256i c_int16_1p1 = _mm256_setzero_si256();

	__m256i c_int16_2p0 = _mm256_setzero_si256();
	__m256i c_int16_2p1 = _mm256_setzero_si256();

	__m256i c_int16_3p0 = _mm256_setzero_si256();
	__m256i c_int16_3p1 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 0) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 1)));

		// Broadcast a[1,kr:kr+2].
		a_int32_1 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 1) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

		// Broadcast a[2,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 2) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
		c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);

		// Broadcast a[3,kr:kr+2].
		a_int32_1 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 3) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec[0], c_int16_2p0);
		c_int16_2p1 = _mm256_add_epi16(inter_vec[1], c_int16_2p1);

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec[2], c_int16_3p0);
		c_int16_3p1 = _mm256_add_epi16(inter_vec[3], c_int16_3p1);
	}

	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		int8_t a_kfringe;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 1)));

		a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

		a_kfringe = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_1 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
		c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);

		a_kfringe = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 4.
		// c[2,0-63] = a[2,kr:kr+4]*b[kr:kr+4,0-63]
		c_int16_2p0 = _mm256_add_epi16(inter_vec[0], c_int16_2p0);
		c_int16_2p1 = _mm256_add_epi16(inter_vec[1], c_int16_2p1);

		a_kfringe = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
		a_int32_1 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec[2], c_int16_3p0);
		c_int16_3p1 = _mm256_add_epi16(inter_vec[3], c_int16_3p1);
	}
    if ( post_ops_attr.is_last_k == 1 )
    {
        //Subtract B matrix sum column values to compensate 
        //for addition of 128 to A matrix elements

        int16_t* bsumptr = post_ops_attr.b_col_sum_vec_s16 + post_ops_attr.b_sum_offset;

        __m256i b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr) );

        c_int16_0p0 = _mm256_sub_epi16( c_int16_0p0 , b0 );
        c_int16_1p0 = _mm256_sub_epi16( c_int16_1p0 , b0 );
        c_int16_2p0 = _mm256_sub_epi16( c_int16_2p0 , b0 );
        c_int16_3p0 = _mm256_sub_epi16( c_int16_3p0 , b0 );

        b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr + 16) );

        c_int16_0p1 = _mm256_sub_epi16( c_int16_0p1 , b0 );
        c_int16_1p1 = _mm256_sub_epi16( c_int16_1p1 , b0 );
        c_int16_2p1 = _mm256_sub_epi16( c_int16_2p1 , b0 );
        c_int16_3p1 = _mm256_sub_epi16( c_int16_3p1 , b0 );
    }

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
		c_int16_0p1 = _mm256_mullo_epi16(selector1, c_int16_0p1);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);
		c_int16_1p1 = _mm256_mullo_epi16(selector1, c_int16_1p1);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);
		c_int16_2p1 = _mm256_mullo_epi16(selector1, c_int16_2p1);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);
		c_int16_3p1 = _mm256_mullo_epi16(selector1, c_int16_3p1);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S8_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			S8_S16_BETA_OP(c_int16_1p1,0,1,1,selector1,selector2)

			// c[2,0-15]
			S8_S16_BETA_OP(c_int16_2p0,0,2,0,selector1,selector2)

			// c[2,16-31]
			S8_S16_BETA_OP(c_int16_2p1,0,2,1,selector1,selector2)

			// c[3,0-15]
			S8_S16_BETA_OP(c_int16_3p0,0,3,0,selector1,selector2)

			// c[3,16-31]
			S8_S16_BETA_OP(c_int16_3p1,0,3,1,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S16_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			S16_S16_BETA_OP(c_int16_1p1,0,1,1,selector1,selector2)

			// c[2,0-15]
			S16_S16_BETA_OP(c_int16_2p0,0,2,0,selector1,selector2)

			// c[2,16-31]
			S16_S16_BETA_OP(c_int16_2p1,0,2,1,selector1,selector2)

			// c[3,0-15]
			S16_S16_BETA_OP(c_int16_3p0,0,3,0,selector1,selector2)

			// c[3,16-31]
			S16_S16_BETA_OP(c_int16_3p1,0,3,1,selector1,selector2)
		}
	}
	
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x32:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );
		selector2 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 )) );
		
		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_add_epi16( selector2, c_int16_0p1 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		// c[1, 16-31]
		c_int16_1p1 = _mm256_add_epi16( selector2, c_int16_1p1 );

		// c[2,0-15]
		c_int16_2p0 = _mm256_add_epi16( selector1, c_int16_2p0 );

		// c[2, 16-31]
		c_int16_2p1 = _mm256_add_epi16( selector2, c_int16_2p1 );

		// c[3,0-15]
		c_int16_3p0 = _mm256_add_epi16( selector1, c_int16_3p0 );

		// c[3, 16-31]
		c_int16_3p1 = _mm256_add_epi16( selector2, c_int16_3p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4x32:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_max_epi16( selector1, c_int16_0p1 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

		// c[1,16-31]
		c_int16_1p1 = _mm256_max_epi16( selector1, c_int16_1p1 );

		// c[2,0-15]
		c_int16_2p0 = _mm256_max_epi16( selector1, c_int16_2p0 );

		// c[2,16-31]
		c_int16_2p1 = _mm256_max_epi16( selector1, c_int16_2p1 );

		// c[3,0-15]
		c_int16_3p0 = _mm256_max_epi16( selector1, c_int16_3p0 );

		// c[3,16-31]
		c_int16_3p1 = _mm256_max_epi16( selector1, c_int16_3p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4x32:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		// c[0,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p1)

		// c[1,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

		// c[1,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p1)

		// c[2,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_2p0)

		// c[2,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_2p1)

		// c[3,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_3p0)

		// c[3,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_3p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4x32:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[0,16-31]
		GELU_TANH_S16_AVX2(c_int16_0p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,16-31]
		GELU_TANH_S16_AVX2(c_int16_1p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[2,0-15]
		GELU_TANH_S16_AVX2(c_int16_2p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[2,16-31]
		GELU_TANH_S16_AVX2(c_int16_2p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[3,0-15]
		GELU_TANH_S16_AVX2(c_int16_3p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[3,16-31]
		GELU_TANH_S16_AVX2(c_int16_3p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4x32:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[0,16-31]
		GELU_ERF_S16_AVX2(c_int16_0p1, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		// c[1,16-31]
		GELU_ERF_S16_AVX2(c_int16_1p1, y1, y2, r, x, x_erf)

		// c[2,0-15]
		GELU_ERF_S16_AVX2(c_int16_2p0, y1, y2, r, x, x_erf)

		// c[2,16-31]
		GELU_ERF_S16_AVX2(c_int16_2p1, y1, y2, r, x, x_erf)

		// c[3,0-15]
		GELU_ERF_S16_AVX2(c_int16_3p0, y1, y2, r, x, x_erf)

		// c[3,16-31]
		GELU_ERF_S16_AVX2(c_int16_3p1, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4x32:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[0,16-31]
		CLIP_S16_AVX2(c_int16_0p1, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		// c[1,16-31]
		CLIP_S16_AVX2(c_int16_1p1, min, max)

		// c[2,0-15]
		CLIP_S16_AVX2(c_int16_2p0, min, max)

		// c[2,16-31]
		CLIP_S16_AVX2(c_int16_2p1, min, max)

		// c[3,0-15]
		CLIP_S16_AVX2(c_int16_3p0, min, max)

		// c[3,16-31]
		CLIP_S16_AVX2(c_int16_3p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_4x32:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m128i _zero_point_0;
		__m256i zero_point_0 = _mm256_setzero_si256();
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Load zero points (2 byte values).
		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale first 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_2p0, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_3p0, scale_1, scale_2, zero_point_0)

		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (2 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (3 * 8));

		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale next 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p1, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_1p1, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_2p1, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_3p1, scale_1, scale_2, zero_point_0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4x32_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-31]
		CVT_STORE_S16_S8(c_int16_0p0, c_int16_0p1, 0, 0);

		// c[1,0-31]
		CVT_STORE_S16_S8(c_int16_1p0, c_int16_1p1, 1, 0);

		// c[2,0-31]
		CVT_STORE_S16_S8(c_int16_2p0, c_int16_2p1, 2, 0);

		// c[3,0-31]
		CVT_STORE_S16_S8(c_int16_3p0, c_int16_3p1, 3, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0*16 )), c_int16_0p0 );

		// c[0, 16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 1*16 )), c_int16_0p1 );

		// c[1,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 0*16 )), c_int16_1p0 );

		// c[1,16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 1*16 )), c_int16_1p1 );

		// c[2,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 2 ) + ( 0*16 )), c_int16_2p0 );

		// c[2,16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 2 ) + ( 1*16 )), c_int16_2p1 );

		// c[3,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 3 ) + ( 0*16 )), c_int16_3p0 );

		// c[3,16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 3 ) + ( 1*16 )), c_int16_3p1 );
	}
}


// 2x32 int8o16 kernel
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int16_t,s8s8s16o16_2x32)
{
	dim_t NR = 32;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_2x32_DISABLE,
			&&POST_OPS_BIAS_2x32,
			&&POST_OPS_RELU_2x32,
			&&POST_OPS_RELU_SCALE_2x32,
			&&POST_OPS_GELU_TANH_2x32,
			&&POST_OPS_GELU_ERF_2x32,
			&&POST_OPS_CLIP_2x32,
			&&POST_OPS_DOWNSCALE_2x32
		};

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;
	__m256i b1;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i a_int32_1;
	__m256i inter_vec[4];

    uint8_t cvt_uint8 = 128;
	__m256i vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();
	__m256i c_int16_0p1 = _mm256_setzero_si256();

	__m256i c_int16_1p0 = _mm256_setzero_si256();
	__m256i c_int16_1p1 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 0) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 1)));

		// Broadcast a[1,kr:kr+2].
		a_int32_1 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 1) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
		c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		int8_t a_kfringe;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 1)));

		a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

		a_kfringe = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_1 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_1 = _mm256_add_epi8( a_int32_1, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
		inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
		c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);
	}
    if ( post_ops_attr.is_last_k == 1 )
    {
        //Subtract B matrix sum column values to compensate 
        //for addition of 128 to A matrix elements

        int16_t* bsumptr = post_ops_attr.b_col_sum_vec_s16 + post_ops_attr.b_sum_offset;

        __m256i b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr) );

        c_int16_0p0 = _mm256_sub_epi16( c_int16_0p0 , b0 );
        c_int16_1p0 = _mm256_sub_epi16( c_int16_1p0 , b0 );

        b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr + 16) );

        c_int16_0p1 = _mm256_sub_epi16( c_int16_0p1 , b0 );
        c_int16_1p1 = _mm256_sub_epi16( c_int16_1p1 , b0 );
    }

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
		c_int16_0p1 = _mm256_mullo_epi16(selector1, c_int16_0p1);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);
		c_int16_1p1 = _mm256_mullo_epi16(selector1, c_int16_1p1);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S8_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			S8_S16_BETA_OP(c_int16_1p1,0,1,1,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S16_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			S16_S16_BETA_OP(c_int16_1p1,0,1,1,selector1,selector2)
		}
	}

		// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x32:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );
		selector2 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 )) );
		
		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_add_epi16( selector2, c_int16_0p1 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		// c[1, 16-31]
		c_int16_1p1 = _mm256_add_epi16( selector2, c_int16_1p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x32:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_max_epi16( selector1, c_int16_0p1 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

		// c[1,16-31]
		c_int16_1p1 = _mm256_max_epi16( selector1, c_int16_1p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x32:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		// c[0,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p1)

		// c[1,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

		// c[1,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2x32:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[0,16-31]
		GELU_TANH_S16_AVX2(c_int16_0p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,16-31]
		GELU_TANH_S16_AVX2(c_int16_1p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2x32:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[0,16-31]
		GELU_ERF_S16_AVX2(c_int16_0p1, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		// c[1,16-31]
		GELU_ERF_S16_AVX2(c_int16_1p1, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2x32:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[0,16-31]
		CLIP_S16_AVX2(c_int16_0p1, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		// c[1,16-31]
		CLIP_S16_AVX2(c_int16_1p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_2x32:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m128i _zero_point_0;
		__m256i zero_point_0 = _mm256_setzero_si256();
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Load zero points (2 byte values).
		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale first 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2, zero_point_0)

		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (2 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (3 * 8));

		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale next 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p1, scale_1, scale_2, zero_point_0)
		CVT_MULRND_CVT16(c_int16_1p1, scale_1, scale_2, zero_point_0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x32_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-31]
		CVT_STORE_S16_S8(c_int16_0p0, c_int16_0p1, 0, 0);

		// c[1,0-31]
		CVT_STORE_S16_S8(c_int16_1p0, c_int16_1p1, 1, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0*16 )), c_int16_0p0 );

		// c[0, 16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 1*16 )), c_int16_0p1 );

		// c[1,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 0*16 )), c_int16_1p0 );

		// c[1,16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 1*16 )), c_int16_1p1 );
	}
}

// 1x32 int8o16 kernel
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int16_t,s8s8s16o16_1x32)
{
	dim_t NR = 32;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_1x32_DISABLE,
			&&POST_OPS_BIAS_1x32,
			&&POST_OPS_RELU_1x32,
			&&POST_OPS_RELU_SCALE_1x32,
			&&POST_OPS_GELU_TANH_1x32,
			&&POST_OPS_GELU_ERF_1x32,
			&&POST_OPS_CLIP_1x32,
			&&POST_OPS_DOWNSCALE_1x32
		};

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;
	__m256i b1;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec[2];

    uint8_t cvt_uint8 = 128;
	__m256i vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();
	__m256i c_int16_0p1 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(int16_t *)(a + (rs_a * 0) + (cs_a * offset)));

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 1)));

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		int8_t a_kfringe;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 0)));
		b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 1)));

		a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

        //convert signed int8 to uint8 for u8s8s16 FMA ops
        a_int32_0 = _mm256_add_epi8( a_int32_0, vec_uint8 );

		// Separate register for intermediate op
		inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
		inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
		c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);
	}
    if ( post_ops_attr.is_last_k == 1 )
    {
        //Subtract B matrix sum column values to compensate 
        //for addition of 128 to A matrix elements

        int16_t* bsumptr = post_ops_attr.b_col_sum_vec_s16 + post_ops_attr.b_sum_offset;

        __m256i b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr) );

        c_int16_0p0 = _mm256_sub_epi16( c_int16_0p0 , b0 );

        b0 = _mm256_loadu_si256( (__m256i const *)(bsumptr + 16) );

        c_int16_0p1 = _mm256_sub_epi16( c_int16_0p1 , b0 );
    }

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
		c_int16_0p1 = _mm256_mullo_epi16(selector1, c_int16_0p1);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S8_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			S16_S16_BETA_OP(c_int16_0p1,0,0,1,selector1,selector2)
		}
	}

		// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x32:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );
		selector2 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 )) );
		
		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_add_epi16( selector2, c_int16_0p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x32:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		// c[0, 16-31]
		c_int16_0p1 = _mm256_max_epi16( selector1, c_int16_0p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x32:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		// c[0,16-31]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1x32:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[0,16-31]
		GELU_TANH_S16_AVX2(c_int16_0p1, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1x32:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[0,16-31]
		GELU_ERF_S16_AVX2(c_int16_0p1, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1x32:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[0,16-31]
		CLIP_S16_AVX2(c_int16_0p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_1x32:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m128i _zero_point_0;
		__m256i zero_point_0 = _mm256_setzero_si256();
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Load zero points (2 byte values).
		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale first 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2, zero_point_0)

		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (2 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (3 * 8));

		_zero_point_0 =
			_mm_loadu_si128(
			( __m128i const* )( ( int8_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
		if ( post_ops_attr.c_stor_type == S8 )
		{
			zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
		}
		else if ( post_ops_attr.c_stor_type == U8 )
		{
			zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
		}

		// Scale next 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p1, scale_1, scale_2, zero_point_0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1x32_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-31]
		CVT_STORE_S16_S8(c_int16_0p0, c_int16_0p1, 0, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0*16 )), c_int16_0p0 );

		// c[0, 16-31]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 1*16 )), c_int16_0p1 );
	}
}
#endif
