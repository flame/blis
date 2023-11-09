/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_S16_KERN_MACROS_H
#define LPGEMM_S16_KERN_MACROS_H

#include "../gelu_avx2.h"
#include "../math_utils_avx2.h"

#define S8_MIN  (-128)
#define S8_MAX  (+127)

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_S16_AVX2(reg) \
	selector1 = _mm256_setzero_si256();\
	selector1 = _mm256_cmpgt_epi16 ( selector1, reg ); \
 \
	/* Only < 0 elements in b0. */ \
	b0 = _mm256_and_si256 ( selector1, reg ); \
\
	/* Only >= 0 elements in c_int16_0p0. */ \
	reg = _mm256_andnot_si256( selector1, reg ); \
 \
	/* Only scaling for < 0 elements. */ \
	b0 = _mm256_mullo_epi16( b0, selector2 ); \
 \
	/* Combine the scaled < 0 and >= 0 elements. */ \
	reg = _mm256_or_si256( b0, reg ); \

// s16 fma macro
#define S16_BETA_FMA(reg,scratch1,scratch2) \
	scratch1 = _mm256_mullo_epi16( scratch2, scratch1 ); \
	reg = _mm256_add_epi16( scratch1, reg ); \

// Beta scale macro, scratch2=beta
#define S16_S16_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm256_loadu_si256 \
	( \
	  ( __m256i const* )( c + ( rs_c * ( m_ir + m_ind ) ) + ( n_ind * 16 ) ) \
	); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

// Beta n < 16 scale macro, scratch2=beta
#define S16_S16_BETA_OP_NLT16(reg,buf_,scratch1,scratch2) \
	scratch1 = _mm256_loadu_si256( ( __m256i const* )buf_ ); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

// Downscale beta scale macro (s8 -> s16), scratch2=beta
#define S8_S16_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm256_cvtepi8_epi16 \
	( \
	  _mm_loadu_si128 \
	  ( \
	    ( __m128i const* )( ( int8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) )\
	  ) \
	); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

// Downscale beta scale macro (u8 -> s16), scratch2=beta
#define U8_S16_BETA_OP(reg,m_ir,m_ind,n_ind,scratch1,scratch2) \
	scratch1 = \
	_mm256_cvtepu8_epi16 \
	( \
	  _mm_loadu_si128 \
	  ( \
	    ( __m128i const* )( ( uint8_t* )post_ops_attr.buf_downscale + \
	    ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	    post_ops_attr.post_op_c_j + ( n_ind * 16 ) )\
	  ) \
	); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

// Downscale beta n < 16 scale macro (s8 -> s16), scratch2=beta
#define S8_S16_BETA_OP_NLT16(reg,buf_,scratch1,scratch2) \
	scratch1 = _mm256_cvtepi8_epi16( _mm_loadu_si128( ( __m128i const* )buf_ ) ); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

// Downscale beta n < 16 scale macro (u8 -> s16), scratch2=beta
#define U8_S16_BETA_OP_NLT16(reg,buf_,scratch1,scratch2) \
	scratch1 = _mm256_cvtepu8_epi16( _mm_loadu_si128( ( __m128i const* )buf_ ) ); \
	S16_BETA_FMA(reg,scratch1,scratch2) \

#define US8_S16_BETA_NLT16_MEMCP_HELPER(buf_,m_ind,bytes, C_type) \
	memcpy \
	( \
	  buf_, \
	  ( ( C_type* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j ), bytes \
	); \

#define S8_S16_BETA_NLT16_MEMCP_UTIL(buf_,m_ind,bytes) \
	US8_S16_BETA_NLT16_MEMCP_HELPER(buf_,m_ind,bytes,int8_t) \

#define U8_S16_BETA_NLT16_MEMCP_UTIL(buf_,m_ind,bytes) \
	US8_S16_BETA_NLT16_MEMCP_HELPER(buf_,m_ind,bytes,uint8_t) \
 
// Downscale macro
#define CVT_MULRND_CVT16(reg, scale0, scale1, zero_point_0) \
 \
	/* Extract the first 128 bits of the register*/ \
	temp[0] = _mm256_extractf128_si256( reg, 0 ); \
	/* Extract the second 128 bits of the register*/ \
	temp[1] = _mm256_extractf128_si256( reg, 1 ); \
 \
	temp_32[0] = _mm256_cvtepi16_epi32( temp[0] ); \
	temp_32[1] = _mm256_cvtepi16_epi32( temp[1] ); \
	temp_float[0] = _mm256_cvtepi32_ps( temp_32[0] ); \
	temp_float[1] = _mm256_cvtepi32_ps( temp_32[1] ); \
 \
	/* Multiply the C matrix by the scale value*/ \
	res_1 = _mm256_mul_ps( temp_float[0], scale0 ); \
	res_2 = _mm256_mul_ps( temp_float[1], scale1 ); \
 \
	/* Round the resultant value to the nearest float value. */ \
	res_1 = \
	    _mm256_round_ps \
	    ( \
	      res_1, ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
	    ); \
	res_2 = \
	    _mm256_round_ps \
	    ( \
	      res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) \
	    ); \
 \
	/* Convert the clipped float32 scaled rounded value to int32 */ \
	temp_32[0] = _mm256_cvtps_epi32( res_1 ); \
	temp_32[1] = _mm256_cvtps_epi32( res_2 ); \
 \
	/* Convert the s32 to s16 */ \
	reg = _mm256_packs_epi32( temp_32[0], temp_32[1] ); \
 \
	/*Permute to make sure the order is correct*/ \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	/* Zero point addition.*/ \
	reg = _mm256_add_epi16( reg, zero_point_0 ); \

// Downscale store macro helper
#define CVT_STORE_S16_SU8_HELPER(reg, m_ind, n_ind, C_type) \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	_mm256_storeu_si256 \
	( \
	  ( __m256i* )( ( C_type* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 32 ) ), \
	  reg \
	); \

// Downscale store macro (s16 -> s8)
#define CVT_STORE_S16_S8(reg0, reg1, m_ind, n_ind) \
   /* Convert the s16 to s8 */ \
	reg0 = _mm256_packs_epi16( reg0, reg1 ); \
	CVT_STORE_S16_SU8_HELPER(reg0, m_ind, n_ind, int8_t) \

// Downscale store macro (s16 -> u8)
#define CVT_STORE_S16_U8(reg0, reg1, m_ind, n_ind) \
   /* Convert the s16 to s8 */ \
	reg0 = _mm256_packus_epi16( reg0, reg1 ); \
	CVT_STORE_S16_SU8_HELPER(reg0, m_ind, n_ind, uint8_t) \

// Downscale store helper macro for fringe cases
#define CVT_STORE_S16_US8_2ROW_HELPER(reg, m_ind0, m_ind1, n_ind, C_type) \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	/* Extract the first 128 bits of the register*/ \
	temp[0] = _mm256_extractf128_si256( reg, 0 ); \
	/* Extract the second 128 bits of the register*/ \
	temp[1] = _mm256_extractf128_si256( reg, 1 ); \
 \
	_mm_storeu_si128 \
	( \
	  ( __m128i* )( ( C_type* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind0 ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ) ), \
	  temp[0] \
	); \
	_mm_storeu_si128 \
	( \
	  ( __m128i* )( ( C_type* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind1 ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ) ), \
	  temp[1] \
	); \

// Downscale store macro for fringe cases (s16 -> s8)
#define CVT_STORE_S16_S8_2ROW(reg0, reg1, m_ind0, m_ind1, n_ind) \
	/* Convert the s16 to s8 */ \
	reg0 = _mm256_packs_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_2ROW_HELPER(reg0, m_ind0, m_ind1, n_ind, int8_t) \

// Downscale store macro for fringe cases (s16 -> u8)
#define CVT_STORE_S16_U8_2ROW(reg0, reg1, m_ind0, m_ind1, n_ind) \
	/* Convert the s16 to u8 */ \
	reg0 = _mm256_packus_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_2ROW_HELPER(reg0, m_ind0, m_ind1, n_ind, uint8_t) \

// Downscale store helper macro for fringe cases
#define CVT_STORE_S16_US8_1ROW(reg, m_ind0, n_ind, C_type) \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	/* Extract the first 128 bits of the register*/ \
	temp[0] = _mm256_extractf128_si256( reg, 0 ); \
 \
	_mm_storeu_si128 \
	( \
	  ( __m128i* )( ( C_type* )post_ops_attr.buf_downscale + \
	  ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind0 ) ) + \
	  post_ops_attr.post_op_c_j + ( n_ind * 16 ) ), \
	  temp[0] \
	); \

// Downscale store (s16 -> s8) macro for fringe cases
#define CVT_STORE_S16_S8_1ROW(reg0, reg1, m_ind0, n_ind) \
	/* Convert the s16 to s8 */ \
	reg0 = _mm256_packs_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_1ROW(reg0, m_ind0, n_ind, int8_t) \

// Downscale store (s16 -> u8) macro for fringe cases
#define CVT_STORE_S16_U8_1ROW(reg0, reg1, m_ind0, n_ind) \
	/* Convert the s16 to u8 */ \
	reg0 = _mm256_packus_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_1ROW(reg0, m_ind0, n_ind, uint8_t) \

// Downscale store helper macro for n < 16 fringe cases
#define CVT_STORE_S16_US8_2ROW_NLT16(reg, buf0, buf1) \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	/* Extract the first 128 bits of the register*/ \
	temp[0] = _mm256_extractf128_si256( reg, 0 ); \
	/* Extract the second 128 bits of the register*/ \
	temp[1] = _mm256_extractf128_si256( reg, 1 ); \
 \
	_mm_storeu_si128( ( __m128i* )buf0, temp[0] ); \
	_mm_storeu_si128( ( __m128i* )buf1, temp[1] ); \

// Downscale store (int16 -> s8) macro for n < 16 fringe cases
#define CVT_STORE_S16_S8_2ROW_NLT16(reg0, reg1, buf0, buf1) \
	/* Convert the s16 to s8 */ \
	reg0 = _mm256_packs_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_2ROW_NLT16(reg0, buf0, buf1) \

// Downscale store (int16 -> u8) macro for n < 16 fringe cases
#define CVT_STORE_S16_U8_2ROW_NLT16(reg0, reg1, buf0, buf1) \
	/* Convert the s16 to s8 */ \
	reg0 = _mm256_packus_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_2ROW_NLT16(reg0, buf0, buf1) \

// Downscale store helper macro for n < 16 fringe cases
#define CVT_STORE_S16_US8_1ROW_NLT16(reg, buf0) \
	reg = _mm256_permute4x64_epi64( reg, 0XD8 ); \
 \
	/* Extract the first 128 bits of the register*/ \
	temp[0] = _mm256_extractf128_si256( reg, 0 ); \
 \
	_mm_storeu_si128( ( __m128i* )buf0, temp[0] ); \

// Downscale store (s16 -> s8) macro for n < 16 fringe cases
#define CVT_STORE_S16_S8_1ROW_NLT16(reg0, reg1, buf0) \
	/* Convert the s16 to s8 */ \
	reg0 = _mm256_packs_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_1ROW_NLT16(reg0, buf0) \

// Downscale store (s16 -> u8) macro for n < 16 fringe cases
#define CVT_STORE_S16_U8_1ROW_NLT16(reg0, reg1, buf0) \
	/* Convert the s16 to u8 */ \
	reg0 = _mm256_packus_epi16( reg0, reg1 ); \
	CVT_STORE_S16_US8_1ROW_NLT16(reg0, buf0) \

#define CVT_STORE_S16_US8_NLT16_MEMCP_HELPER(buf_,m_ind,bytes, C_type) \
	memcpy \
	( \
	  ( ( C_type* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j ), buf_, bytes \
	); \

#define CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf_,m_ind,bytes) \
	CVT_STORE_S16_US8_NLT16_MEMCP_HELPER(buf_,m_ind,bytes, int8_t) \

#define CVT_STORE_S16_U8_NLT16_MEMCP_UTIL(buf_,m_ind,bytes) \
	CVT_STORE_S16_US8_NLT16_MEMCP_HELPER(buf_,m_ind,bytes, uint8_t) \

//--------------------------------------------------------------------------
/* GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_S16_AVX2(reg, y1, y2, r, r2, x, z, dn, x_tanh, q) \
\
	y1 = _mm256_cvtepi32_ps( _mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg, 0)) ); \
	y2 = _mm256_cvtepi32_ps( _mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg, 1)) ); \
\
	GELU_TANH_F32_AVX2_DEF(y1, r, r2, x, z, dn, x_tanh, q); \
\
	GELU_TANH_F32_AVX2_DEF(y2, r, r2, x, z, dn, x_tanh, q); \
\
	reg = _mm256_packs_epi32(_mm256_cvtps_epi32(y1), _mm256_cvtps_epi32(y2));\
	reg = _mm256_permute4x64_epi64(reg, 0XD8);\


/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_S16_AVX2(reg, y1, y2, r, x, x_erf) \
\
	y1 = _mm256_cvtepi32_ps( _mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg, 0)) ); \
	y2 = _mm256_cvtepi32_ps( _mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg, 1)) ); \
\
	GELU_ERF_F32_AVX2_DEF(y1, r, x, x_erf); \
\
	GELU_ERF_F32_AVX2_DEF(y2, r, x, x_erf); \
\
	reg = _mm256_packs_epi32(_mm256_cvtps_epi32(y1), _mm256_cvtps_epi32(y2));\
	reg = _mm256_permute4x64_epi64(reg, 0XD8);\

#define CLIP_S16_AVX2(reg, min, max) \
\
	reg = _mm256_min_epi16( _mm256_max_epi16( reg, min ), max ); \

#endif //LPGEMM_S16_KERN_MACROS_H
