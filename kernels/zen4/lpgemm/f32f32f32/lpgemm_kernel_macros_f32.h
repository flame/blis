/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_F32_SGEMM_KERN_MACROS_H
#define LPGEMM_F32_SGEMM_KERN_MACROS_H

#include "../gelu_avx512.h"
#include "../silu_avx512.h"
#include "../sigmoid_avx512.h"
#include "../math_utils_avx512.h"

/* Zero-out all ZMM registers */
#define ZERO_ZMM_ALL \
    zmm0  = _mm512_setzero_ps(); \
    zmm1  = _mm512_setzero_ps(); \
    zmm2  = _mm512_setzero_ps(); \
    zmm3  = _mm512_setzero_ps(); \
    zmm4  = _mm512_setzero_ps(); \
    zmm5  = _mm512_setzero_ps(); \
    zmm6  = _mm512_setzero_ps(); \
    zmm7  = _mm512_setzero_ps(); \
    zmm8  = _mm512_setzero_ps(); \
    zmm9  = _mm512_setzero_ps(); \
    zmm10 = _mm512_setzero_ps(); \
    zmm11 = _mm512_setzero_ps(); \
    zmm12 = _mm512_setzero_ps(); \
    zmm13 = _mm512_setzero_ps(); \
    zmm14 = _mm512_setzero_ps(); \
    zmm15 = _mm512_setzero_ps(); \
    zmm16 = _mm512_setzero_ps(); \
    zmm17 = _mm512_setzero_ps(); \
    zmm18 = _mm512_setzero_ps(); \
    zmm19 = _mm512_setzero_ps(); \
    zmm20 = _mm512_setzero_ps(); \
    zmm21 = _mm512_setzero_ps(); \
    zmm22 = _mm512_setzero_ps(); \
    zmm23 = _mm512_setzero_ps(); \
    zmm24 = _mm512_setzero_ps(); \
    zmm25 = _mm512_setzero_ps(); \
    zmm26 = _mm512_setzero_ps(); \
    zmm27 = _mm512_setzero_ps(); \
    zmm28 = _mm512_setzero_ps(); \
    zmm29 = _mm512_setzero_ps(); \
    zmm30 = _mm512_setzero_ps(); \
    zmm31 = _mm512_setzero_ps();

/* Zero-out all YMM registers */
#define ZERO_YMM_ALL \
    ymm0  = _mm256_setzero_ps(); \
    ymm1  = _mm256_setzero_ps(); \
    ymm2  = _mm256_setzero_ps(); \
    ymm3  = _mm256_setzero_ps(); \
    ymm4  = _mm256_setzero_ps(); \
    ymm5  = _mm256_setzero_ps(); \
    ymm6  = _mm256_setzero_ps(); \
    ymm7  = _mm256_setzero_ps(); \
    ymm8  = _mm256_setzero_ps(); \
    ymm9  = _mm256_setzero_ps(); \
    ymm10 = _mm256_setzero_ps(); \
    ymm11 = _mm256_setzero_ps(); \
    ymm12 = _mm256_setzero_ps(); \
    ymm13 = _mm256_setzero_ps(); \
    ymm14 = _mm256_setzero_ps(); \
    ymm15 = _mm256_setzero_ps();

/* Zero-out all XMM registers */
#define ZERO_XMM_ALL \
    xmm0 = _mm_setzero_ps(); \
    xmm1 = _mm_setzero_ps(); \
    xmm2 = _mm_setzero_ps(); \
    xmm3 = _mm_setzero_ps(); \
    xmm4 = _mm_setzero_ps(); \
    xmm5 = _mm_setzero_ps(); \
    xmm6 = _mm_setzero_ps(); \
    xmm7 = _mm_setzero_ps();


/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32S_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_ps_mask( reg, zmm1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mul_ps( reg, relu_cmp_mask, reg, zmm2 ); \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32S_AVX512(reg, r, r2, x, z, dn, x_tanh, q) \
\
	GELU_TANH_F32_AVX512_DEF(reg, r, r2, x, z, dn, x_tanh, q); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32S_AVX512(reg, r, x, x_erf) \
\
	GELU_ERF_F32_AVX512_DEF(reg, r, x, x_erf); \

#define CLIP_F32S_AVX512(reg, min, max) \
\
	reg = _mm512_min_ps( _mm512_max_ps( reg, min ), max ); \

// Zero-out the given ZMM accumulator registers
#define ZERO_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3) \
    zmm0 = _mm512_setzero_ps(); \
    zmm1 = _mm512_setzero_ps(); \
    zmm2 = _mm512_setzero_ps(); \
    zmm3 = _mm512_setzero_ps();

#define ZERO_ACC_ZMM_3_REG(zmm0,zmm1,zmm2) \
    zmm0 = _mm512_setzero_ps(); \
    zmm1 = _mm512_setzero_ps(); \
    zmm2 = _mm512_setzero_ps();

#define ZERO_ACC_ZMM_2_REG(zmm0,zmm1) \
    zmm0 = _mm512_setzero_ps(); \
    zmm1 = _mm512_setzero_ps();

// Zero-out the given YMM accumulator registers
#define ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm2, ymm3) \
    ymm0 = _mm256_setzero_ps(); \
    ymm1 = _mm256_setzero_ps(); \
    ymm2 = _mm256_setzero_ps(); \
    ymm3 = _mm256_setzero_ps();

#define ZERO_ACC_YMM_3_REG(ymm0,ymm1,ymm2) \
    ymm0 = _mm256_setzero_ps(); \
    ymm1 = _mm256_setzero_ps(); \
    ymm2 = _mm256_setzero_ps();

#define ZERO_ACC_YMM_2_REG(ymm0,ymm1) \
    ymm0 = _mm256_setzero_ps(); \
    ymm1 = _mm256_setzero_ps();

// Zero-out the given YMM accumulator registers
#define ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3) \
    xmm0 = _mm_setzero_ps(); \
    xmm1 = _mm_setzero_ps(); \
    xmm2 = _mm_setzero_ps(); \
    xmm3 = _mm_setzero_ps();

#define ZERO_ACC_XMM_3_REG(xmm0,xmm1,xmm2) \
    xmm0 = _mm_setzero_ps(); \
    xmm1 = _mm_setzero_ps(); \
    xmm2 = _mm_setzero_ps();

#define ZERO_ACC_XMM_2_REG(xmm0,xmm1) \
    xmm0 = _mm_setzero_ps(); \
    xmm1 = _mm_setzero_ps();

/*Multiply alpha with accumulator registers and store back*/
#define ALPHA_MUL_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3,alpha) \
    zmm0 = _mm512_mul_ps(zmm0,alpha); \
    zmm1 = _mm512_mul_ps(zmm1,alpha); \
    zmm2 = _mm512_mul_ps(zmm2,alpha); \
    zmm3 = _mm512_mul_ps(zmm3,alpha);

#define ALPHA_MUL_ACC_ZMM_2_REG(zmm0,zmm1,alpha) \
    zmm0 = _mm512_mul_ps(zmm0,alpha); \
    zmm1 = _mm512_mul_ps(zmm1,alpha);

#define ALPHA_MUL_ACC_ZMM_1_REG(zmm0,alpha) \
    zmm0 = _mm512_mul_ps(zmm0,alpha);

/*Beta helpers*/
// F32 fma macro
#define F32_BETA_FMA(reg,scratch1,scratch2) \
	scratch1 = _mm512_mul_ps( scratch2, scratch1 ); \
	reg = _mm512_add_ps( scratch1, reg ); \

#define BF16_F32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	  (__m512)( _mm512_sllv_epi32( _mm512_cvtepi16_epi32( (__m256i)_mm256_loadu_epi16 \
	  ( \
	    ( ( bfloat16* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) )\
	  ) ), _mm512_set1_epi32 (16) ) );\
	F32_BETA_FMA(reg,scratch1,scratch2)\

// Downscale n < 16 mask load beta macro
#define BF16_F32_BETA_OP_NLT16F_MASK(lmask,reg,m_ind,n_ind,scratch1,scratch2) \
	scratch1 =  \
	  (__m512)( _mm512_sllv_epi32( _mm512_cvtepi16_epi32( (__m256i)_mm256_maskz_loadu_epi16 \
	  ( \
	    lmask, \
	    ( bfloat16* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
	  ) ), _mm512_set1_epi32 (16) ) );\
	F32_BETA_FMA(reg,scratch1,scratch2)\

//BF16->F32 store helper
#define CVT_STORE_F32_BF16_MASK(reg,m_ind,n_ind) \
	_mm256_mask_storeu_epi16 \
		( \
			( bfloat16* )post_ops_attr.buf_downscale + \
			( post_ops_attr.rs_c_downscale *  \
				( post_ops_attr.post_op_c_i + m_ind ) ) + \
			post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
			mask_all1, (__m256i) _mm512_cvtneps_pbh( reg ) \
		) \

// This conversion will be used when AVX512BF16 is not supported.
#define CVT_STORE_F32_BF16_MASK_AVX512( n_elem, reg, m_ind, n_ind ) \
{ \
	_mm512_storeu_ps((float*)temp, reg); \
	dest = ( bfloat16* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j + ( n_ind * 16 ); \
	for(i = 0; i < n_elem; i++) \
	{ \
		tlsb = ( temp[i] & ( uint32_t )0x00010000 ) > 16; \
		rounded = temp[i] + ( uint32_t )0x00007FFF + tlsb; \
		memcpy( (dest+i), ((char *)(&rounded))+2, sizeof(bfloat16)); \
	} \
}

// BF16 bias helper macros.
#define BF16_F32_BIAS_LOAD(scr,mask,n_ind) \
	scr = ( __m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_loadu_epi16 \
						( \
						  ( mask ), \
						  ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
						  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
		  ); \

// F32 bias helper macros.
#define S32_F32_BIAS_LOAD(scr,mask,n_ind) \
	scr = 	_mm512_cvtepi32_ps \
			( \
				_mm512_maskz_loadu_epi32 \
				( \
					( mask ), \
					( ( int32_t* ) post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
			); \

// S8 bias helper macros.
#define S8_F32_BIAS_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_loadu_epi8 \
			  ( \
				( mask ), \
				( ( int8_t* )post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			  ) \
			) \
			); \

// BF16 bias helper macros.
#define BF16_F32_BIAS_BCAST(scr,mask,m_ind) \
	scr = ( __m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_set1_epi16 \
						( \
						  ( mask ), \
						  *( ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
						  post_ops_attr.post_op_c_i + m_ind ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
		  ); \

// F32 bias helper macros.
#define S32_F32_BIAS_BCAST(scr,mask,m_ind) \
	scr = 	_mm512_cvtepi32_ps \
			( \
				_mm512_maskz_set1_epi32 \
				( \
					( mask ), \
					*( ( ( int32_t* ) post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_i + m_ind ) \
				) \
			); \

// S8 bias helper macros.
#define S8_F32_BIAS_BCAST(scr,mask,m_ind) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_set1_epi8 \
			  ( \
				( mask ), \
				*( ( ( int8_t* )post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_i + m_ind ) \
			  ) \
			) \
			); \

//zero point bf16->f32


#define BF16_F32_ZP_BCST(scr,n_ind, zp_mask) \
scr = ( __m512)( _mm512_sllv_epi32 \
		( \
			_mm512_cvtepi16_epi32 \
			( \
				_mm256_maskz_set1_epi16( zp_mask, \
					*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) \
			), _mm512_set1_epi32( 16 ) \
		) \
	);

#define BF16_F32_ZP_COL_BCST(scr,n_ind, zp_mask) \
scr = ( __m512)( _mm512_sllv_epi32 \
		( \
			_mm512_cvtepi16_epi32 \
			( \
				_mm256_maskz_set1_epi16( zp_mask,\
					*( (bfloat16* )post_ops_list_temp->op_args1 + \
                                post_ops_attr.post_op_c_i + n_ind ) ) \
			), _mm512_set1_epi32( 16 ) \
		) \
	);

#define BF16_F32_BIAS_GEMV(scr, n_ind, bias_mask) \
	BF16_F32_ZP_BCST(scr,n_ind, bias_mask); \

#define BF16_F32_BIAS_LOAD_GEMV(scr,mask,n_ind) \
	scr = ( __m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_loadu_epi16 \
						( \
						  ( mask ), \
						  ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
						  post_ops_attr.post_op_c_i + ( n_ind * 16 ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
		  ); \

#define F32_MATRIX_ADD_1COL(scr0,m_ind,r_ind0) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \

// Matrix Add post-ops helper macros
#define F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \

#define F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_add_ps( scr2, zmm ## r_ind2 ); \

#define F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_add_ps( scr2, zmm ## r_ind2 ); \
	zmm ## r_ind3 = _mm512_add_ps( scr3, zmm ## r_ind3 ); \

//BF16 matrix_add helper macros.
#define BF16_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = (__m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_loadu_epi16 \
						( \
						  mask, \
						  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
						  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
				  ); \
	scr = _mm512_mul_ps( scr, scl_fct ); \

#define BF16_F32_MATRIX_ADD_1COL_MASK(mask,scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_ADD_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define BF16_F32_MATRIX_ADD_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define BF16_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define BF16_F32_MATRIX_ADD_LOAD_GEMV(mask,scr,scl_fct,postop_ptr,n_ind) \
	scr = (__m512)( _mm512_sllv_epi32 \
		( \
		_mm512_cvtepi16_epi32 \
		( \
			_mm256_maskz_loadu_epi16 \
			( \
			mask, \
			(bfloat16 *)(matptr + postop_ptr + ( n_ind * 16 ) )\
			) \
		), _mm512_set1_epi32( 16 ) \
		) \
	); \
	scr = _mm512_mul_ps( scr, scl_fct );


// S8 matrix_add helper macros.
#define S8_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_cvtepi32_ps( \
		  _mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_loadu_epi8 \
			  ( \
				mask, \
				matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
				post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			  ) \
			) \
			); \
	scr = _mm512_mul_round_ps \
			( \
			  ( scr ), scl_fct, \
			  ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
			); \

#define S8_F32_MATRIX_ADD_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define S8_F32_MATRIX_ADD_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define S8_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define S8_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

// S32 matrix_add helper macros.
#define S32_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_cvtepi32_ps ( \
		  _mm512_maskz_loadu_epi32 \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			) \
			); \
	scr = _mm512_mul_round_ps \
			( \
			  ( scr ), scl_fct, \
			  ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	        ); \

#define S32_F32_MATRIX_ADD_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define S32_F32_MATRIX_ADD_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define S32_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define S32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

//F32 matrix_add helper macros.
#define F32_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \
	scr = _mm512_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_ADD_1COL_ZMM(scr0, \
				scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask, scr0, \
				scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define F32_F32_MATRIX_ADD_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define F32_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(k0,scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(k1,scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(k2,scr2,scl_fct2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(k3,scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

// Matrix Mul post-ops helper macros
#define F32_MATRIX_MUL_1COL(scr0,m_ind,r_ind0) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \

#define F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \

#define F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_mul_ps( scr2, zmm ## r_ind2 ); \

#define F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_mul_ps( scr2, zmm ## r_ind2 ); \
	zmm ## r_ind3 = _mm512_mul_ps( scr3, zmm ## r_ind3 ); \

#define BF16_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define BF16_F32_MATRIX_MUL_LOAD_GEMV(mask,scr,scl_fct,postop_ptr,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD_GEMV(mask,scr,scl_fct,postop_ptr,n_ind);

#define BF16_F32_MATRIX_MUL_1COL_MASK(mask,scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define BF16_F32_MATRIX_MUL_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define BF16_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,scl_fct0, \
				scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \


#define S8_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S8_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define S8_F32_MATRIX_MUL_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define S8_F32_MATRIX_MUL_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define S8_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,scl_fct0, \
				scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define S8_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define S32_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S32_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \


#define S32_F32_MATRIX_MUL_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define S32_F32_MATRIX_MUL_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define S32_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,scl_fct0, \
				scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define S32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \


#define F32_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	F32_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define F32_F32_MATRIX_MUL_1COL_ZMM(scr0, \
				scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask, scr0, \
				scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_2COL(scr0,scr1, \
				scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define F32_F32_MATRIX_MUL_3COL(scr0,scr1,scr2, \
				scl_fct0,scl_fct1,scl_fct2,m_ind,r_ind0,r_ind1,r_ind2) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define F32_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,scl_fct0, \
				scl_fct1,scl_fct2,scl_fct3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(k0,scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(k1,scr1,scl_fct1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(k2,scr2,scl_fct2,m_ind,2); \
	F32_F32_MATRIX_MUL_LOAD(k3,scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

//Downscale Post-ops helpers.
#define F32_SCL_MULRND(reg,selector,zero_point) \
	reg = _mm512_mul_ps( reg, selector ); \
	reg = _mm512_add_ps( reg, zero_point ); \

//u8 zero point helper macros
#define U8_F32_ZP_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtepi32_ps \
				( \
				_mm512_cvtepu8_epi32 \
				( \
				_mm_maskz_loadu_epi8 \
				( \
					( mask ), \
					( ( int8_t* )post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
				) \
				); \

//s8 zero point helper macros
#define S8_F32_ZP_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtepi32_ps \
				( \
				_mm512_cvtepi8_epi32 \
				( \
				_mm_maskz_loadu_epi8 \
				( \
					( mask ), \
					( ( int8_t* )post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
				) \
				); \

//bf16 zero point helper macros
#define BF16_F32_ZP_LOAD(scr,mask,n_ind) \
	scr = ( __m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_loadu_epi16 \
						( \
						  ( mask ), \
						  ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
						  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
		  ); \

//s32 zero point helper macros
#define S32_F32_ZP_LOAD(scr,mask,n_ind) \
	scr = 	_mm512_cvtepi32_ps \
			( \
				_mm512_maskz_loadu_epi32 \
				( \
					( mask ), \
					( ( int32_t* ) post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
			); \

//u8 zero point helper macros
#define U8_F32_ZP_BCAST(scr,mask,m_ind) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepu8_epi32 \
			( \
			  _mm_maskz_set1_epi8 \
			  ( \
				( mask ), \
				*( ( ( int8_t* )post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_i + m_ind ) \
			  ) \
			) \
			); \

//s8 zero point helper macros
#define S8_F32_ZP_BCAST(scr,mask,m_ind) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_set1_epi8 \
			  ( \
				( mask ), \
				*( ( ( int8_t* )post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_i + m_ind ) \
			  ) \
			) \
			); \

//bf16 zero point helper macros
#define BF16_F32_ZP_BCAST(scr,mask,m_ind) \
	scr = ( __m512)( _mm512_sllv_epi32 \
					( \
					  _mm512_cvtepi16_epi32 \
					  ( \
						_mm256_maskz_set1_epi16 \
						( \
						  ( mask ), \
						  *( ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
						  post_ops_attr.post_op_c_i +  m_ind ) \
						) \
					  ), _mm512_set1_epi32( 16 ) \
					) \
		  ); \

//s32 zero point helper macros
#define S32_F32_ZP_BCAST(scr,mask,m_ind) \
	scr = 	_mm512_cvtepi32_ps \
			( \
				_mm512_maskz_set1_epi32 \
				( \
					( mask ), \
					*( ( ( int32_t* ) post_ops_list_temp->op_args1 ) + \
					post_ops_attr.post_op_c_i + m_ind ) \
				) \
			); \

//u8 zero point helper macros
#define U8_F32_SCALAR_ZP_BCAST(scr,mask) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepu8_epi32 \
			( \
			  _mm_maskz_set1_epi8 \
			  ( \
				( mask ), \
				*( ( int8_t* )post_ops_list_temp->op_args1 ) \
			  ) \
			) \
			); \

//s8 zero point helper macros
#define S8_F32_SCALAR_ZP_BCAST(scr,mask) \
	scr = _mm512_cvtepi32_ps \
			( \
			_mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_set1_epi8 \
			  ( \
				( mask ), \
				*( ( int8_t* )post_ops_list_temp->op_args1 ) \
			  ) \
			) \
			); \

//bf16 zero point helper macros
#define BF16_F32_SCALAR_ZP_BCAST(scr,mask) \
	scr = ( __m512)( _mm512_sllv_epi32 \
				( \
					_mm512_cvtepi16_epi32 \
					( \
					_mm256_maskz_set1_epi16 \
					( \
						( mask ), \
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) \
					) \
					), _mm512_set1_epi32( 16 ) \
				) \
			); \

//s32 zero point helper macros
#define S32_F32_SCALAR_ZP_BCAST(scr,mask) \
	scr = 	_mm512_cvtepi32_ps \
			( \
			_mm512_maskz_set1_epi32 \
			( \
				( mask ), \
				*( ( int32_t* ) post_ops_list_temp->op_args1 ) \
			) \
			); \

#ifdef LPGEMM_BF16_JIT
#define CVT_STORE_F32_BF16_POST_OPS_MASK(ir,jr,reg,mask,m_ind,n_ind)
#else
// Downscale store bf16 macro
#define CVT_STORE_F32_BF16_POST_OPS_MASK(ir,jr,reg,mask,m_ind,n_ind) \
	_mm256_mask_storeu_epi16 \
	( \
	  ((bfloat16*)b) + ( rs_b * ( ir + m_ind ) ) + ( cs_b * ( jr + n_ind ) ), \
	  mask, (__m256i) _mm512_cvtneps_pbh( reg ) \
	)
#endif

// Downscale store s8 macro
#define CVT_STORE_F32_S8_POST_OPS_MASK(reg,mask,m_ind,n_ind) \
	_mm512_mask_cvtsepi32_storeu_epi8 \
	( \
	  b_q + ( rs_b * ( ir + m_ind ) ) + ( cs_b * ( jr + n_ind ) ), \
	  mask, _mm512_cvtps_epi32( reg ) \
	) \

// Downscale store u8 macro
#define CVT_STORE_F32_U8_POST_OPS_MASK(reg,mask,m_ind,n_ind) \
	_mm512_mask_cvtusepi32_storeu_epi8 \
	( \
	  b_q + ( rs_b * ( ir + m_ind ) ) + ( cs_b * ( jr + n_ind ) ), \
	  mask, _mm512_cvtps_epu32( _mm512_max_ps( reg, _mm512_set1_ps( 0 ) ) ) \
	) \

// Downscale store f32 macro
#define CVT_STORE_F32_S32_POST_OPS_MASK(reg,mask,m_ind,n_ind) \
	_mm512_mask_storeu_epi32 \
	( \
	  b_q + ( rs_b * ( ir + m_ind ) ) + ( cs_b * ( jr + n_ind ) ), \
	  mask, _mm512_cvtps_epi32 ( reg ) \
	) \

/*x_tanh = tanhf(x_tanh) */  \
#define TANH_F32S_AVX512(x_tanh, r, r2, x, z, dn, q) \
\
	TANHF_AVX512(x_tanh, r, r2, x, z, dn, q)

#endif //LPGEMM_F32_SGEMM_KERN_MACROS_H
