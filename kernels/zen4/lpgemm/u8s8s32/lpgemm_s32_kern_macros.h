/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.

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
#include "../math_utils_avx512.h"

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

// Downscale beta op.
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

// Default n < 16 beta macro
#define S32_S32_BETA_OP_NLT16F(reg,buf_,scratch1,scratch2) \
	scratch1 = _mm512_loadu_si512( buf_ ); \
	S32_BETA_FMA(reg,scratch1,scratch2) \

// Default n < 16 mask load beta macro
#define S32_S32_BETA_OP_NLT16F_MASK(lmask,reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
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

// ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0
#define RELU_SCALE_OP_S32_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_epi32_mask( reg, selector1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mullo_epi32( reg, relu_cmp_mask, reg, selector2 ); \

// Downscale macro
#define CVT_MULRND_CVT32(reg,selector) \
	reg = \
	_mm512_cvtps_epi32 \
	( \
	  _mm512_mul_round_ps \
	  ( \
		_mm512_cvtepi32_ps( reg ), \
		( __m512 )selector, \
		( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	  ) \
	) \

// Downscale store macro
#define CVT_STORE_S32_S8(reg,m_ind,n_ind) \
	_mm512_mask_cvtsepi32_storeu_epi8 \
	( \
	  ( int8_t* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
	  mask_all1, reg \
	) \

// Downscale n < 16 macro
#define CVT_MULRND_CVT32_LT16(reg,selector) \
	reg = \
	_mm512_cvtps_epi32 \
	( \
	  _mm512_mul_round_ps \
	  ( \
	    _mm512_cvtepi32_ps( reg ), \
	    ( __m512 )selector, \
	    ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	  ) \
	) \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */ 
#define GELU_TANH_S32_AVX512(reg, y, r, r2, x, z, dn, x_tanh, q) \
\
	y = _mm512_cvtepi32_ps( reg ); \
\
	GELU_TANH_F32_AVX512_DEF(y, r, r2, x, z, dn, x_tanh, q); \
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

// Load helper macros.
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

// Store helper macros.
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

#endif // LPGEMM_S32_KERN_MACROS_H
