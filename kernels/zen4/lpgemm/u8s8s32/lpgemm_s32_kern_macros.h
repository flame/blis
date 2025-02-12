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

#ifndef LPGEMM_S32_KERN_MACROS_H
#define LPGEMM_S32_KERN_MACROS_H

#include "../gelu_avx512.h"
#include "../silu_avx512.h"
#include "../math_utils_avx512.h"
#include "../sigmoid_avx512.h"

#define S32_BETA_FMA(reg,scratch1,scratch2) \
	scratch1 = _mm512_mullo_epi32( scratch2, scratch1 ); \
	reg = _mm512_add_epi32( scratch1, reg ); \

#define S32_S32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = _mm512_loadu_si512( c + ( rs_c * ( m_ir + m_ind ) ) + ( n_ind * 16 ) ); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

#define S32_S32_BETA_OP2(m_ir,m_ind,scratch1,scratch2) \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \

#define S32_S32_BETA_OP3(m_ir,m_ind,scratch1,scratch2) \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \

#define S32_S32_BETA_OP4(m_ir,m_ind,scratch1,scratch2) \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \
	S32_S32_BETA_OP(c_int32_ ## m_ind ## p3,m_ir,m_ind,3,scratch1,scratch2); \

// Downscale S8 beta op.
#define S8_S32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm512_cvtepi8_epi32 \
	( \
	  _mm_maskz_loadu_epi8 \
	  ( \
		0xFFFF, \
	    ( int8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
	  ) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

#define S8_S32_BETA_OP2(m_ir,m_ind,scratch1,scratch2) \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \

#define S8_S32_BETA_OP3(m_ir,m_ind,scratch1,scratch2) \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \

#define S8_S32_BETA_OP4(m_ir,m_ind,scratch1,scratch2) \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \
	S8_S32_BETA_OP(c_int32_ ## m_ind ## p3,m_ir,m_ind,3,scratch1,scratch2); \

// Downscale U8 beta op.
#define U8_S32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm512_cvtepu8_epi32 \
	( \
	  _mm_maskz_loadu_epi8 \
	  ( \
		0xFFFF, \
	    ( uint8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
	  ) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

#define U8_S32_BETA_OP2(m_ir,m_ind,scratch1,scratch2) \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \

#define U8_S32_BETA_OP3(m_ir,m_ind,scratch1,scratch2) \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \

#define U8_S32_BETA_OP4(m_ir,m_ind,scratch1,scratch2) \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \
	U8_S32_BETA_OP(c_int32_ ## m_ind ## p3,m_ir,m_ind,3,scratch1,scratch2); \

// Downscale F32 beta op
#define F32_S32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm512_cvtps_epi32 \
    ( \
		_mm512_maskz_loadu_ps \
		( \
			0xFFFF, \
			( float* )post_ops_attr.buf_downscale + \
			( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
			post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
		) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

#define F32_S32_BETA_OP2(m_ir,m_ind,scratch1,scratch2) \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \

#define F32_S32_BETA_OP3(m_ir,m_ind,scratch1,scratch2) \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \

#define F32_S32_BETA_OP4(m_ir,m_ind,scratch1,scratch2) \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \
	F32_S32_BETA_OP(c_int32_ ## m_ind ## p3,m_ir,m_ind,3,scratch1,scratch2); \

// Downscale BF16 beta op
#define BF16_S32_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	    _mm512_cvtps_epi32 \
		( \
			(__m512)_mm512_sllv_epi32 \
			( \
				_mm512_cvtepi16_epi32 \
				( \
				_mm256_maskz_loadu_epi16 \
				( \
					0xFFFF, \
					( bfloat16* )post_ops_attr.buf_downscale + \
					( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
				), _mm512_set1_epi32( 16 ) \
			) \
		); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

#define BF16_S32_BETA_OP2(m_ir,m_ind,scratch1,scratch2) \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \

#define BF16_S32_BETA_OP3(m_ir,m_ind,scratch1,scratch2) \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \

#define BF16_S32_BETA_OP4(m_ir,m_ind,scratch1,scratch2) \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p0,m_ir,m_ind,0,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p1,m_ir,m_ind,1,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p2,m_ir,m_ind,2,scratch1,scratch2); \
	BF16_S32_BETA_OP(c_int32_ ## m_ind ## p3,m_ir,m_ind,3,scratch1,scratch2); \

// Default n < 16 beta macro
#define S32_S32_BETA_OP_NLT16F(reg,buf_,scratch1,scratch2) \
	scratch1 = _mm512_loadu_si512( buf_ ); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Default n < 16 mask load beta macro
#define S32_S32_BETA_OP_NLT16F_MASK(c,lmask,reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = _mm512_maskz_loadu_epi32( lmask, c + ( rs_c * ( m_ir + m_ind ) ) + ( n_ind * 16 ) ); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Downscale n < 16 mask load beta macro
#define S8_S32_BETA_OP_NLT16F_MASK(lmask,reg,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = _mm512_cvtepi8_epi32 \
	( \
	  _mm_maskz_loadu_epi8 \
	  ( \
	    lmask, \
	    ( int8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
	  ) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Downscale U8 n < 16 mask load beta macro
#define U8_S32_BETA_OP_NLT16F_MASK(lmask,reg,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = _mm512_cvtepu8_epi32 \
	( \
	  _mm_maskz_loadu_epi8 \
	  ( \
	    lmask, \
	    ( uint8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
	  ) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Downscale n < 16 mask load F32 beta macro
#define F32_S32_BETA_OP_NLT16F_MASK(lmask,reg,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm512_cvtps_epi32 \
    ( \
		_mm512_maskz_loadu_ps \
		( \
			lmask, \
			( float* )post_ops_attr.buf_downscale + \
			( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
			post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
		) \
	); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Downscale n < 16 mask load BF16 beta macro
#define BF16_S32_BETA_OP_NLT16F_MASK(lmask,reg,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	    _mm512_cvtps_epi32 \
		( \
			(__m512)_mm512_sllv_epi32 \
			( \
				_mm512_cvtepi16_epi32 \
				( \
				_mm256_maskz_loadu_epi16 \
				( \
					lmask, \
					( bfloat16* )post_ops_attr.buf_downscale + \
					( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
					post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
				), _mm512_set1_epi32( 16 ) \
			) \
		); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

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
#define BF16_S32_BIAS_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtps_epi32 \
				( \
					( __m512)( _mm512_sllv_epi32 \
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
				  ) \
				); \

// F32 bias helper macros.
#define F32_S32_BIAS_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtps_epi32 \
			( \
				_mm512_maskz_loadu_ps \
				( \
				( mask ), \
				( ( float* ) post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
				) \
			); \

// S8 bias helper macros.
#define S8_S32_BIAS_LOAD(scr,mask,n_ind) \
	scr = _mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_loadu_epi8 \
			  ( \
				( mask ), \
				( ( int8_t* )post_ops_list_temp->op_args1 ) + \
				post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			  ) \
			); \

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_ps_mask( reg, zero ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mul_ps( reg, relu_cmp_mask, reg, scale ); \

// ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0
#define RELU_SCALE_OP_S32_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_epi32_mask( reg, selector1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mullo_epi32( reg, relu_cmp_mask, reg, selector2 ); \

// Downscale macro
#define CVT_MULRND_F32(reg,scale,zero_point) \
	reg = _mm512_mul_round_ps \
	  ( \
		reg, \
		scale, \
		( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	); \
	if( post_ops_attr.c_stor_type == U8 ) \
	{ \
		reg = _mm512_add_ps( reg, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32( zero_point )) ); \
	}else{ \
		reg = _mm512_add_ps( reg, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32( zero_point )) ); \
	} \


// Downscale macro
#define CVT_MULRND_CVT32(reg,selector,zero_point) \
	reg = \
	_mm512_cvtps_epi32 \
	( \
	  _mm512_mul_round_ps \
	  ( \
		_mm512_cvtepi32_ps( reg ), \
		( __m512 )selector, \
		( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	  ) \
	); \
	reg = _mm512_add_epi32( reg, _mm512_cvtepi8_epi32( zero_point ) ); \

// Downscale store s8 macro
#define CVT_STORE_S32_S8(reg,m_ind,n_ind) \
	_mm512_mask_cvtsepi32_storeu_epi8 \
	( \
	  ( int8_t* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask_all1, reg \
	) \

// Downscale store s8 macro
#define CVT_STORE_F32_S8_MASK(mask,reg,m_ind,n_ind) \
	_mm512_mask_cvtsepi32_storeu_epi8 \
	( \
	  ( int8_t* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask, _mm512_cvtps_epi32(reg) \
	); \

#define CVT_STORE_F32_S8(reg,m_ind,n_ind) \
	CVT_STORE_F32_S8_MASK(mask_all1,reg,m_ind,n_ind) \

// Downscale store u8 macro
#define CVT_STORE_F32_U8_MASK(mask,reg,m_ind,n_ind) \
	_mm512_mask_cvtusepi32_storeu_epi8 \
	( \
	  ( int8_t* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask, _mm512_cvtps_epu32( _mm512_max_ps( reg, _mm512_set1_ps( 0 ) ) ) \
	); \

#define CVT_STORE_F32_U8(reg,m_ind,n_ind) \
	CVT_STORE_F32_U8_MASK(mask_all1,reg,m_ind,n_ind) \

#ifdef LPGEMM_BF16_JIT
#define CVT_STORE_F32_BF16_MASK(mask,reg,m_ind,n_ind)
#define CVT_STORE_F32_BF16_MASK_AVX2(reg,mask, ptr)
#else
// Downscale store bf16 macro
#define CVT_STORE_F32_BF16_MASK(mask,reg,m_ind,n_ind) \
	_mm256_mask_storeu_epi16 \
	( \
	  ( bfloat16* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask, (__m256i) _mm512_cvtneps_pbh( ( reg ) ) \
	); \

#define CVT_STORE_F32_BF16_MASK_AVX2(reg,mask, ptr) \
	_mm256_mask_storeu_epi16( ptr, mask, \
		(__m256i)_mm512_cvtneps_pbh( reg ) );
#endif


#define CVT_STORE_F32_BF16(reg,m_ind,n_ind) \
	CVT_STORE_F32_BF16_MASK(mask_all1,reg,m_ind,n_ind); \

// Downscale store f32 macro
#define CVT_STORE_S32_F32(reg,m_ind,n_ind) \
	_mm512_mask_storeu_ps  \
	( \
	  ( float* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask_all1, _mm512_cvtepi32_ps ( reg ) \
	) \

// Downscale store f32 macro
#define STORE_F32_MASK(mask,reg,m_ind,n_ind) \
	_mm512_mask_storeu_ps  \
	( \
	  ( float* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask,  ( reg ) \
	); \

#define STORE_F32(reg,m_ind,n_ind) \
	STORE_F32_MASK(mask_all1,reg,m_ind,n_ind); \

// Downscale n < 16 macro
#define CVT_MULRND_CVT32_LT16(reg,selector,zero_point) \
	reg = \
	_mm512_cvtps_epi32 \
	( \
	  _mm512_mul_round_ps \
	  ( \
	    _mm512_cvtepi32_ps( reg ), \
	    ( __m512 )selector, \
	    ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	  ) \
	); \
	reg = _mm512_add_epi32( reg, _mm512_cvtepi8_epi32( zero_point ) ); \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_S32_AVX512(reg, y, r, r2, x, z, dn, x_tanh, q) \
\
	y = _mm512_cvtepi32_ps( reg ); \
\
	GELU_TANH_F32_AVX512_DEF(y, r, r2, x, z, dn, x_tanh, q); \
\
	reg = _mm512_cvtps_epi32( y ); \


/* TANH */
#define TANH_S32_AVX512(reg, y, r, r2, x, z, dn, q) \
\
	y = _mm512_cvtepi32_ps( reg ); \
\
	TANHF_AVX512(y, r, r2, x, z, dn, q); \
\
	reg = _mm512_cvtps_epi32( y ); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_S32_AVX512(reg, y, r, x, x_erf) \
\
	y = _mm512_cvtepi32_ps( reg ); \
\
	GELU_ERF_F32_AVX512_DEF(y, r, x, x_erf); \
\
	reg = _mm512_cvtps_epi32( y ); \

#define CLIP_S32_AVX512(reg, min, max) \
\
	reg = _mm512_min_epi32( _mm512_max_epi32( reg, min ), max ); \

#define CLIP_F32_AVX512(reg, min, max) \
\
	reg = _mm512_min_ps( _mm512_max_ps( reg, min ), max ); \

// Gelu load helper macros.
#define S32_GELU_LOAD1R_1C(temp_buf,offset,stride,reg_base) \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ), reg_base ## p0); \

#define S32_GELU_LOAD1R_2C(temp_buf,offset,stride,reg_base) \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ), reg_base ## p0); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ), reg_base ## p1); \

#define S32_GELU_LOAD1R_3C(temp_buf,offset,stride,reg_base) \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ), reg_base ## p0); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ), reg_base ## p1); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 2 + offset ) * ( stride ) ), reg_base ## p2); \

#define S32_GELU_LOAD1R_4C(temp_buf,offset,stride,reg_base) \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ), reg_base ## p0); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ), reg_base ## p1); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 2 + offset ) * ( stride ) ), reg_base ## p2); \
	_mm512_storeu_si512( ( temp_buf ) + ( ( 3 + offset ) * ( stride ) ), reg_base ## p3); \

// Gelu store helper macros.
#define S32_GELU_STORE1R_1C(temp_buf,offset,stride,reg_base) \
	reg_base ## p0 = _mm512_loadu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ) ); \

#define S32_GELU_STORE1R_2C(temp_buf,offset,stride,reg_base) \
	reg_base ## p0 = _mm512_loadu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ) ); \
	reg_base ## p1 = _mm512_loadu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ) ); \

#define S32_GELU_STORE1R_3C(temp_buf,offset,stride,reg_base) \
	reg_base ## p0 = _mm512_loadu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ) ); \
	reg_base ## p1 = _mm512_loadu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ) ); \
	reg_base ## p2 = _mm512_loadu_si512( ( temp_buf ) + ( ( 2 + offset ) * ( stride ) ) ); \

#define S32_GELU_STORE1R_4C(temp_buf,offset,stride,reg_base) \
	reg_base ## p0 = _mm512_loadu_si512( ( temp_buf ) + ( ( 0 + offset ) * ( stride ) ) ); \
	reg_base ## p1 = _mm512_loadu_si512( ( temp_buf ) + ( ( 1 + offset ) * ( stride ) ) ); \
	reg_base ## p2 = _mm512_loadu_si512( ( temp_buf ) + ( ( 2 + offset ) * ( stride ) ) ); \
	reg_base ## p3 = _mm512_loadu_si512( ( temp_buf ) + ( ( 3 + offset ) * ( stride ) ) ); \

// Matrix Add post-ops helper macros
#define S32_MATRIX_ADD_1COL(scr0,m_ind) \
	c_int32_ ## m_ind ## p0 = _mm512_add_epi32( scr0, c_int32_ ## m_ind ## p0 ); \

#define S32_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	c_int32_ ## m_ind ## p0 = _mm512_add_epi32( scr0, c_int32_ ## m_ind ## p0 ); \
	c_int32_ ## m_ind ## p1 = _mm512_add_epi32( scr1, c_int32_ ## m_ind ## p1 ); \

#define S32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	c_int32_ ## m_ind ## p0 = _mm512_add_epi32( scr0, c_int32_ ## m_ind ## p0 ); \
	c_int32_ ## m_ind ## p1 = _mm512_add_epi32( scr1, c_int32_ ## m_ind ## p1 ); \
	c_int32_ ## m_ind ## p2 = _mm512_add_epi32( scr2, c_int32_ ## m_ind ## p2 ); \

#define S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_int32_ ## m_ind ## p0 = _mm512_add_epi32( scr0, c_int32_ ## m_ind ## p0 ); \
	c_int32_ ## m_ind ## p1 = _mm512_add_epi32( scr1, c_int32_ ## m_ind ## p1 ); \
	c_int32_ ## m_ind ## p2 = _mm512_add_epi32( scr2, c_int32_ ## m_ind ## p2 ); \
	c_int32_ ## m_ind ## p3 = _mm512_add_epi32( scr3, c_int32_ ## m_ind ## p3 ); \

#define S8_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_cvtepi8_epi32 \
			( \
			  _mm_maskz_loadu_epi8 \
			  ( \
				mask, \
				matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
				post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			  ) \
			); \
	scr = _mm512_cvtps_epi32( \
			_mm512_mul_round_ps \
			( \
			  _mm512_cvtepi32_ps( scr ), scl_fct, \
			  ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
			) \
		  ); \

#define S8_S32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S8_S32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_ADD_1COL(scr0,m_ind); \

#define S8_S32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_ADD_1COL(scr0,m_ind); \

#define S8_S32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define S8_S32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define S8_S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define S32_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_epi32 \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \
	scr = _mm512_cvtps_epi32( \
			_mm512_mul_round_ps \
			( \
			  _mm512_cvtepi32_ps( scr ), scl_fct, \
			  ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
			) \
		  ); \

#define S32_S32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S32_S32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_ADD_1COL(scr0,m_ind); \

#define S32_S32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_ADD_1COL(scr0,m_ind); \

#define S32_S32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define S32_S32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define S32_S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

// S32 with F32 matrix add post-ops helper macros
#define F32_MATRIX_ADD_1COL(scr0,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \

#define F32_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \

#define F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr2, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ) ) \
		); \

#define F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr2, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ) ) \
		); \
	c_int32_ ## m_ind ## p3 = \
		_mm512_cvtps_epi32( \
			_mm512_add_ps( ( __m512 )scr3, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p3 ) ) \
		); \

// BF16 buffer for matrix add/mul in u8s8s32.
#define BF16_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_sllv_epi32 \
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
			); \
	scr = ( __m512i )_mm512_mul_ps( ( __m512 )scr, scl_fct ); \

#define BF16_S32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	BF16_S32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_S32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_S32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define BF16_S32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define BF16_S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

// F32 buffer for matrix add/mul in u8s8s32.
#define F32_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = ( __m512i )_mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \
	scr = ( __m512i )_mm512_mul_ps( ( __m512 )scr, scl_fct ); \

#define F32_S32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	F32_S32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_S32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_S32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define F32_S32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define F32_S32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_S32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

// Matrix Mul post-ops helper macros
// mul_epi32 works on 64 bit lengths, with mul done for lower 32 bits.
// We only need 32 bit mul to get 32 bit output, so using mul_ps.
#define S32_MATRIX_MUL_1COL(scr0,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr0 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \

#define S32_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr0 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr1 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \

#define S32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr0 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr1 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr2 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ) ) \
		); \

#define S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr0 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr1 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr2 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ) ) \
		); \
	c_int32_ ## m_ind ## p3 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_ps( _mm512_cvtepi32_ps( scr3 ), \
						_mm512_cvtepi32_ps( c_int32_ ## m_ind ## p3 ) ) \
		); \

#define S8_S32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S8_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define S8_S32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S8_S32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_MUL_1COL(scr0,m_ind); \

#define S8_S32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_MUL_1COL(scr0,m_ind); \

#define S8_S32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define S8_S32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define S8_S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define S32_S32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S32_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define S32_S32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S32_S32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_MUL_1COL(scr0,m_ind); \

#define S32_S32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_MATRIX_MUL_1COL(scr0,m_ind); \

#define S32_S32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define S32_S32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define S32_S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

// S32 with F32 matrix add post-ops helper macros
#define F32_MATRIX_MUL_1COL(scr0,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \

#define F32_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \

#define F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr2, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \

#define F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	c_int32_ ## m_ind ## p0 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr0, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p0 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p1 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr1, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p1 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p2 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr2, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p2 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \
	c_int32_ ## m_ind ## p3 = \
		_mm512_cvtps_epi32( \
			_mm512_mul_round_ps( ( __m512 )scr3, _mm512_cvtepi32_ps( c_int32_ ## m_ind ## p3 ), \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ) \
		); \

// BF16 buffer for matrix add/mul in u8s8s32.
#define BF16_S32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	BF16_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define BF16_S32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	BF16_S32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_S32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_S32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define BF16_S32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define BF16_S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

// F32 buffer for matrix add/mul in u8s8s32.
#define F32_S32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	F32_S32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define F32_S32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	F32_S32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_S32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_S32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define F32_S32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define F32_S32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_S32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

// SiLU utility macros. al register expected to contains floats.
#define SWISH_S32_AVX512(in_reg, fl_reg, al, al_in, r, r2, z, dn, ex_out) \
	fl_reg = _mm512_cvtepi32_ps( in_reg ); \
	SWISH_F32_AVX512_DEF( fl_reg, al, al_in, r, r2, z, dn, ex_out); \
	in_reg = _mm512_cvtps_epi32( fl_reg ); \

// Sigmoid utility macros. al register expected to contains floats.
#define SIGMOID_S32_AVX512(in_reg, fl_reg, al_in, r, r2, z, dn, ex_out) \
	fl_reg = _mm512_cvtepi32_ps( in_reg ); \
	SIGMOID_F32_AVX512_DEF( fl_reg, al_in, r, r2, z, dn, ex_out); \
	in_reg = _mm512_cvtps_epi32( fl_reg ); \

//Zero-out the given ZMM accumulator registers
#define ZERO_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3) \
	zmm0 = _mm512_setzero_epi32(); \
	zmm1 = _mm512_setzero_epi32(); \
	zmm2 = _mm512_setzero_epi32(); \
	zmm3 = _mm512_setzero_epi32();

#define ZERO_ACC_XMM_4_REG(zmm0,zmm1,zmm2,zmm3) \
	zmm0 = _mm_setzero_si128 (); \
	zmm1 = _mm_setzero_si128 (); \
	zmm2 = _mm_setzero_si128 (); \
	zmm3 = _mm_setzero_si128 ();

#define CVT_STORE_S32_S8_MASK(reg,mask,m_ind,n_ind) \
  _mm512_mask_cvtsepi32_storeu_epi8 \
  ( \
    ( int8_t* )post_ops_attr.buf_downscale + \
    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
    post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
    mask, reg \
  ); \

  // Downscale store f32 macro
#define CVT_STORE_S32_F32_MASK(reg,mask,m_ind,n_ind) \
	_mm512_mask_storeu_ps  \
	( \
	  ( float* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask, _mm512_cvtepi32_ps ( reg ) \
	) \

// int32 to float conversion helpers.
#define CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, n_ind) \
	flt_reg_pfx ## m_ind ## n_ind = \
		_mm512_cvtepi32_ps( int_reg_pfx ## m_ind ## p ## n_ind); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_1COL(flt_reg_pfx, int_reg_pfx, m_ind) \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 0); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_2COL(flt_reg_pfx, int_reg_pfx, m_ind) \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 1); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_3COL(flt_reg_pfx, int_reg_pfx, m_ind) \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 2); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_4COL(flt_reg_pfx, int_reg_pfx, m_ind) \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_M_N(flt_reg_pfx, int_reg_pfx, m_ind, 3); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_12ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 3); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 4); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 5); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 6); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 7); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 8); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 9); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 10); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 11); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_9ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 3); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 4); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 5); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 6); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 7); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 8); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_6ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 3); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 4); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 5); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_5ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 3); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 4); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_4ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 3); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_3ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 2); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_2ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 1); \

#define CVT_ACCUM_REG_INT_TO_FLOAT_1ROWS_XCOL(flt_reg_pfx, int_reg_pfx, cols) \
	CVT_ACCUM_REG_INT_TO_FLOAT_ ## cols ## COL(flt_reg_pfx, int_reg_pfx, 0); \

// F32 matrix add post-ops helper macros
#define F32_ACC_MATRIX_ADD_1COL(scr0,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_add_ps( ( __m512 )scr0, ( acc_ ## m_ind ## 0 ) ) ; \

#define F32_ACC_MATRIX_ADD_2COL(scr0,scr1,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_add_ps( ( __m512 )scr0, ( acc_ ## m_ind ## 0 ) ) ; \
	acc_ ## m_ind ## 1 = _mm512_add_ps( ( __m512 )scr1, ( acc_ ## m_ind ## 1 ) ) ; \

#define F32_ACC_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_add_ps( ( __m512 )scr0, ( acc_ ## m_ind ## 0 ) ) ; \
	acc_ ## m_ind ## 1 = _mm512_add_ps( ( __m512 )scr1, ( acc_ ## m_ind ## 1 ) ) ; \
	acc_ ## m_ind ## 2 = _mm512_add_ps( ( __m512 )scr2, ( acc_ ## m_ind ## 2 ) ) ; \

#define F32_ACC_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_add_ps( ( __m512 )scr0, ( acc_ ## m_ind ## 0 ) ) ; \
	acc_ ## m_ind ## 1 = _mm512_add_ps( ( __m512 )scr1, ( acc_ ## m_ind ## 1 ) ) ; \
	acc_ ## m_ind ## 2 = _mm512_add_ps( ( __m512 )scr2, ( acc_ ## m_ind ## 2 ) ) ; \
	acc_ ## m_ind ## 3 = _mm512_add_ps( ( __m512 )scr3, ( acc_ ## m_ind ## 3 ) ) ; \

// BF16 buffer for matrix add/mul in u8s8s32.
#define BF16_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr =(__m512)(_mm512_sllv_epi32 \
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
			) );\
	scr = _mm512_mul_ps(scr, scl_fct ); \

#define BF16_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define BF16_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define BF16_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ACC_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \


#define BF16_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ACC_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \


// F32 buffer for matrix add/mul in u8s8s32.
#define F32_ACC_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \
	scr = _mm512_mul_ps( ( __m512 )scr, scl_fct ); \

#define F32_ONLY_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	F32_ACC_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_ONLY_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define F32_ONLY_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define F32_ONLY_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ACC_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define F32_ONLY_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ACC_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ACC_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

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

#define S8_F32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define S8_F32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define S8_F32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define S8_F32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ACC_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define S8_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ACC_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \

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

#define S32_F32_MATRIX_ADD_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define S32_F32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ACC_MATRIX_ADD_1COL(scr0,m_ind); \

#define S32_F32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ACC_MATRIX_ADD_2COL(scr0,scr1,m_ind); \

#define S32_F32_MATRIX_ADD_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ACC_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind); \

#define S32_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ACC_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind); \


// Matrix Mul post-ops helper macros
// mul_epi32 works on 64 bit lengths, with mul done for lower 32 bits.
// We only need 32 bit mul to get 32 bit output, so using mul_ps.
#define F32_ACC_MATRIX_MUL_1COL(scr0,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_mul_ps( scr0 , acc_ ## m_ind ## 0 ); \

#define F32_ACC_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_mul_ps( scr0 , acc_ ## m_ind ## 0 ); \
	acc_ ## m_ind ## 1 = _mm512_mul_ps( scr1 , acc_ ## m_ind ## 1 ); \

#define F32_ACC_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_mul_ps( scr0 , acc_ ## m_ind ## 0 ); \
	acc_ ## m_ind ## 1 = _mm512_mul_ps( scr1 , acc_ ## m_ind ## 1 ); \
	acc_ ## m_ind ## 2 = _mm512_mul_ps( scr2 , acc_ ## m_ind ## 2 ); \

#define F32_ACC_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	acc_ ## m_ind ## 0 = _mm512_mul_ps( scr0 , acc_ ## m_ind ## 0 ); \
	acc_ ## m_ind ## 1 = _mm512_mul_ps( scr1 , acc_ ## m_ind ## 1 ); \
	acc_ ## m_ind ## 2 = _mm512_mul_ps( scr2 , acc_ ## m_ind ## 2 ); \
	acc_ ## m_ind ## 3 = _mm512_mul_ps( scr3 , acc_ ## m_ind ## 3 ); \


#define F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind) \
	acc_ ## m_ind ## 0 =_mm512_mul_round_ps(scr0, acc_ ## m_ind ## 0 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \

#define F32_ONLY_MATRIX_MUL_2COL(scr0,scr1,m_ind) \
	acc_ ## m_ind ## 0 =_mm512_mul_round_ps(scr0, acc_ ## m_ind ## 0 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 1 =_mm512_mul_round_ps(scr1, acc_ ## m_ind ## 1, \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \

#define F32_ONLY_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind) \
	acc_ ## m_ind ## 0 =_mm512_mul_round_ps(scr0, acc_ ## m_ind ## 0 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 1 =_mm512_mul_round_ps(scr1, acc_ ## m_ind ## 1 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 2 =_mm512_mul_round_ps(scr2, acc_ ## m_ind ## 2 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \

#define F32_ONLY_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind) \
	acc_ ## m_ind ## 0 =_mm512_mul_round_ps( scr0, acc_ ## m_ind ## 0 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 1 =_mm512_mul_round_ps(scr1, acc_ ## m_ind ## 1 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 2 =_mm512_mul_round_ps(scr2, acc_ ## m_ind ## 2 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \
	acc_ ## m_ind ## 3 =_mm512_mul_round_ps(scr3, acc_ ## m_ind ## 3 , \
				   ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) ); \

#define S8_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S8_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define S8_F32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define S8_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define S8_F32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ONLY_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define S8_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ONLY_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define S8_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
							   scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S8_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ONLY_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#define S32_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	S32_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define S32_F32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define S32_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define S32_F32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ONLY_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define S32_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ONLY_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define S32_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
								scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	S32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ONLY_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

// BF16 buffer for matrix add/mul in u8s8s32.
#define BF16_F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define BF16_F32_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define BF16_F32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ONLY_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define BF16_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ONLY_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define BF16_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	BF16_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ONLY_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

// F32 buffer for matrix add/mul in u8s8s32.
#define F32_MATRIX_MUL_LOAD(mask,scr,scl_fct,m_ind,n_ind) \
	F32_ACC_MATRIX_ADD_LOAD(mask,scr,scl_fct,m_ind,n_ind); \

#define F32_U8S8_MATRIX_MUL_1COL_PAR(mask,scr0,scl_fct0,m_ind) \
	F32_MATRIX_MUL_LOAD(mask,scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_U8S8_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind) \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_ONLY_MATRIX_MUL_1COL(scr0,m_ind); \

#define F32_U8S8_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind) \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_ONLY_MATRIX_MUL_2COL(scr0,scr1,m_ind); \

#define F32_U8S8_MATRIX_MUL_3COL(scr0,scr1,scr2,scl_fct0,scl_fct1,scl_fct2,m_ind) \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_ONLY_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind); \

#define F32_U8S8_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3, \
				scl_fct0,scl_fct1,scl_fct2,scl_fct3,m_ind) \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,scl_fct2,m_ind,2); \
	F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,scl_fct3,m_ind,3); \
	F32_ONLY_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind); \

#endif // LPGEMM_S32_KERN_MACROS_H
