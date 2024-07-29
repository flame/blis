/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_F32_KERN_MACROS_H
#define LPGEMM_F32_KERN_MACROS_H

#include "../gelu_avx512.h"
#include "../silu_avx512.h"
#include "../math_utils_avx512.h"

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_ps_mask( reg, selector1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mul_ps( reg, relu_cmp_mask, reg, selector2 ); \

// F32 fma macro
#define F32_BETA_FMA(reg,scratch1,scratch2) \
	scratch1 = _mm512_mul_ps( scratch2, scratch1 ); \
	reg = _mm512_add_ps( scratch1, reg ); \

// Beta scale macro, scratch2=beta
#define F32_F32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm512_loadu_ps \
	( \
	  ( c + ( rs_c * ( m_ir + m_ind ) ) + ( n_ind * 16 ) ) \
	); \
	F32_BETA_FMA(reg,scratch1,scratch2) \

// Downscale beta scale macro, scratch2=beta
#define BF16_F32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	  (__m512)( _mm512_sllv_epi32( _mm512_cvtepi16_epi32( (__m256i)_mm256_loadu_epi16 \
	  ( \
	    ( ( bfloat16* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) )\
	  ) ), _mm512_set1_epi32 (16) ) );\
	F32_BETA_FMA(reg,scratch1,scratch2) \

// Default n < 16 mask load beta macro
#define F32_F32_BETA_OP_NLT16F_MASK(c,lmask,reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = _mm512_maskz_loadu_ps( lmask, c + ( rs_c * ( m_ir + m_ind ) ) + ( n_ind * 16 ) ); \
	F32_BETA_FMA(reg,scratch1,scratch2) \

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
	F32_BETA_FMA(reg,scratch1,scratch2) \

// zero_point(avx512 register) contains bf16 zp upscaled to f32.
#define SCL_MULRND_F32(reg,selector,zero_point) \
	reg = _mm512_mul_ps( reg, selector ); \
	reg = _mm512_add_ps( reg, zero_point ); \

#define CVT_STORE_F32_BF16_MASK(reg,m_ind,n_ind) \
	_mm256_mask_storeu_epi16 \
	( \
	  ( bfloat16* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask_all1, (__m256i) _mm512_cvtneps_pbh( reg ) \
	) \

#define CVT_STORE_F32_BF16_POST_OPS_MASK(reg,mask,m_ind,n_ind) \
	_mm256_mask_storeu_epi16 \
	( \
	  b_q + ( rs_b * ( ir + m_ind ) ) + ( cs_b * ( jr + n_ind ) ), \
	  mask, (__m256i) _mm512_cvtneps_pbh( reg ) \
	) \

// BF16 -> F32 convert helpers. reg: __m512
#define CVT_BF16_F32_INT_SHIFT(in) \
	( __m512 )_mm512_sllv_epi32( _mm512_cvtepi16_epi32( ( in ) ), \
								 _mm512_set1_epi32( 16 ) );

// BF16 bias helper macros.
#define BF16_F32_BIAS_LOAD(scr,mask,n_ind) \
	scr = (__m512)( _mm512_sllv_epi32 \
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

#define BF16_F32_BIAS_BCAST(scr,mask,m_ind) \
	scr = (__m512)( _mm512_sllv_epi32 \
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

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32_AVX512(reg, r, r2, x, z, dn, x_tanh, q) \
\
	GELU_TANH_F32_AVX512_DEF(reg, r, r2, x, z, dn, x_tanh, q); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32_AVX512(reg, r, x, x_erf) \
\
	GELU_ERF_F32_AVX512_DEF(reg, r, x, x_erf); \

#define CLIP_F32_AVX512(reg, min, max) \
\
	reg = _mm512_min_ps( _mm512_max_ps( reg, min ), max ); \

// Matrix Add post-ops helper macros
#define F32_MATRIX_ADD_1COL(scr0,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_add_ps( scr0, c_float_ ## m_ind ## p0 ); \

#define F32_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_add_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_add_ps( scr1, c_float_ ## m_ind ## p1 ); \

#define F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_add_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_add_ps( scr1, c_float_ ## m_ind ## p1 ); \
	c_float_ ## m_ind ## p2 = _mm512_add_ps( scr2, c_float_ ## m_ind ## p2 ); \

#define F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_add_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_add_ps( scr1, c_float_ ## m_ind ## p1 ); \
	c_float_ ## m_ind ## p2 = _mm512_add_ps( scr2, c_float_ ## m_ind ## p2 ); \
	c_float_ ## m_ind ## p3 = _mm512_add_ps( scr3, c_float_ ## m_ind ## p3 ); \

#define BF16_F32_MATRIX_ADD_LOAD(mask,scr,m_ind,n_ind) \
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

#define BF16_F32_MATRIX_ADD_1COL_PAR(mask,scr0,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(mask,scr0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_ADD_1COL(scr0,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define BF16_F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define BF16_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(k0,scr0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(k1,scr1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(k2,scr2,m_ind,2); \
	BF16_F32_MATRIX_ADD_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define F32_F32_MATRIX_ADD_LOAD(mask,scr,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \

#define F32_F32_MATRIX_ADD_1COL_PAR(mask,scr0,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(mask,scr0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_F32_MATRIX_ADD_1COL(scr0,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_F32_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define F32_F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define F32_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(k0,scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(k1,scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(k2,scr2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

// Matrix mul post-ops helper macros
#define F32_MATRIX_MUL_1COL(scr0,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_mul_ps( scr0, c_float_ ## m_ind ## p0 ); \

#define F32_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_mul_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_mul_ps( scr1, c_float_ ## m_ind ## p1 ); \

#define F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_mul_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_mul_ps( scr1, c_float_ ## m_ind ## p1 ); \
	c_float_ ## m_ind ## p2 = _mm512_mul_ps( scr2, c_float_ ## m_ind ## p2 ); \

#define F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_float_ ## m_ind ## p0 = _mm512_mul_ps( scr0, c_float_ ## m_ind ## p0 ); \
	c_float_ ## m_ind ## p1 = _mm512_mul_ps( scr1, c_float_ ## m_ind ## p1 ); \
	c_float_ ## m_ind ## p2 = _mm512_mul_ps( scr2, c_float_ ## m_ind ## p2 ); \
	c_float_ ## m_ind ## p3 = _mm512_mul_ps( scr3, c_float_ ## m_ind ## p3 ); \

#define BF16_F32_MATRIX_MUL_LOAD(mask,scr,m_ind,n_ind) \
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

#define BF16_F32_MATRIX_MUL_1COL_PAR(mask,scr0,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(mask,scr0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_MUL_1COL(scr0,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define BF16_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define BF16_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,scr0,scr1,scr2,scr3,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(k0,scr0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(k1,scr1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(k2,scr2,m_ind,2); \
	BF16_F32_MATRIX_MUL_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define F32_F32_MATRIX_MUL_LOAD(mask,scr,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \

#define F32_F32_MATRIX_MUL_1COL_PAR(mask,scr0,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(mask,scr0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_F32_MATRIX_MUL_1COL(scr0,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_F32_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define F32_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define F32_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(k0,scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(k1,scr1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(k2,scr2,m_ind,2); \
	F32_F32_MATRIX_MUL_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

//Zero-out the given ZMM accumulator registers
#define ZERO_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3) \
	zmm0 = _mm512_setzero_ps(); \
	zmm1 = _mm512_setzero_ps(); \
	zmm2 = _mm512_setzero_ps(); \
	zmm3 = _mm512_setzero_ps();

#endif // LPGEMM_F32_KERN_MACROS_H
