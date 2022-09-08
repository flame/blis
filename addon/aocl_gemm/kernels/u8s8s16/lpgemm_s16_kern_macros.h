/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
 \

//--------------------------------------------------------------------------

#define bli_mm256_s16_downscale(c_int16__p0, c_int16__p1, vec_loc)\
\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(c_int16__p0, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(c_int16__p0, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
  /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p0 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p0 = _mm256_permute4x64_epi64(c_int16__p0, 0XD8);\
\
   /* Extract the first 128 bits of the register*/\
	temp[0] = _mm256_extractf128_si256(c_int16__p1, 0);\
\
  /* Extract the second 128 bits of the register*/\
	temp[1] = _mm256_extractf128_si256(c_int16__p1, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
   /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p1 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p1 = _mm256_permute4x64_epi64(c_int16__p1, 0XD8);\
\
   /* Convert the s16 to s8 */\
	store_reg = _mm256_packs_epi16(c_int16__p0, c_int16__p1);\
\
  /* Store the result in s8 form */\
	_mm256_storeu_si256((__m256i *)(( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + vec_loc ) ) + post_op_c_j + ( 0 * 16 )), store_reg);\
\

//--------------------------------------------------------------------------

#define bli_mm256_s16_downscale2(c_int16__p0, c_int16__p1, vec_loc1, vec_loc2)\
\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(c_int16__p0, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(c_int16__p0, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
  /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p0 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p0 = _mm256_permute4x64_epi64(c_int16__p0, 0XD8);\
\
   /* Extract the first 128 bits of the register*/\
	temp[0] = _mm256_extractf128_si256(c_int16__p1, 0);\
\
  /* Extract the second 128 bits of the register*/\
	temp[1] = _mm256_extractf128_si256(c_int16__p1, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
   /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p1 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p1 = _mm256_permute4x64_epi64(c_int16__p1, 0XD8);\
\
   /* Convert the s16 to s8 */\
	store_reg = _mm256_packs_epi16(c_int16__p0, c_int16__p1);\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(store_reg, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(store_reg, 1);\
\
  /* Store the result in s8 form */\
	_mm_storeu_si128((__m128i *)(( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + vec_loc1 ) ) + post_op_c_j + ( 0 * 16 )), temp[0]);\
  _mm_storeu_si128((__m128i *)(( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + vec_loc2 ) ) + post_op_c_j + ( 0 * 16 )), temp[1]);\
\

//--------------------------------------------------------------------------

#define bli_mm256_s16_downscale2_lt16(c_int16__p0, c_int16__p1, vec_loc1, vec_loc2)\
\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(c_int16__p0, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(c_int16__p0, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
  /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p0 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p0 = _mm256_permute4x64_epi64(c_int16__p0, 0XD8);\
\
   /* Extract the first 128 bits of the register*/\
	temp[0] = _mm256_extractf128_si256(c_int16__p1, 0);\
\
  /* Extract the second 128 bits of the register*/\
	temp[1] = _mm256_extractf128_si256(c_int16__p1, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
   /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p1 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p1 = _mm256_permute4x64_epi64(c_int16__p1, 0XD8);\
\
   /* Convert the s16 to s8 */\
	store_reg = _mm256_packs_epi16(c_int16__p0, c_int16__p1);\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(store_reg, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(store_reg, 1);\
\
  /* Store the result in s8 form */\
  _mm_storeu_si128((__m128i *)store_buf, temp[0]);\
  memcpy( ( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + vec_loc1 ) ) + post_op_c_j + \
	  ( 0 * 16 ) , store_buf, ( n0_rem * sizeof( int8_t ) ) ); \
\
  _mm_storeu_si128((__m128i *)store_buf, temp[1]);\
  memcpy( ( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + vec_loc1 ) ) + post_op_c_j + \
	  ( 0 * 16 ) , store_buf, ( n0_rem * sizeof( int8_t ) ) ); \
\

//--------------------------------------------------------------------------

#define bli_mm256_s16_downscale2_edge(c_int16__p0, c_int16__p1)\
\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(c_int16__p0, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(c_int16__p0, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
  /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p0 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p0 = _mm256_permute4x64_epi64(c_int16__p0, 0XD8);\
\
  /* Convert the s32 to s16 */\
	c_int16__p1 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p1 = _mm256_permute4x64_epi64(c_int16__p1, 0XD8);\
\
   /* Convert the s16 to s8 */\
	store_reg = _mm256_packs_epi16(c_int16__p0, c_int16__p1);\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(store_reg, 0);\
\
  /* Store the result in s8 form */\
	_mm_storeu_si128((__m128i *)(( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + 0 ) ) + post_op_c_j + ( 0 * 16 )), temp[0]);\
\

//--------------------------------------------------------------------------

#define bli_mm256_s16_downscale2_edge_lt16(c_int16__p0, c_int16__p1)\
\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(c_int16__p0, 0);\
  /* Extract the second 128 bits of the register*/\
  temp[1] = _mm256_extractf128_si256(c_int16__p0, 1);\
\
  temp_32[0] = _mm256_cvtepi16_epi32(temp[0]);\
  temp_float[0] = _mm256_cvtepi32_ps(temp_32[0]);\
\
  /* Since s16 values cannot be converted to f32 directly,
	they are converted to s32, then to f32 and the scale is performed*/\
  temp_32[1] = _mm256_cvtepi16_epi32(temp[1]);\
  temp_float[1] = _mm256_cvtepi32_ps(temp_32[1]);\
\
  /* Multiply the C matrix by the scale value*/\
  res_1 = _mm256_mul_ps(temp_float[0], scale_1);\
  res_2 = _mm256_mul_ps(temp_float[0], scale_2);\
\
  /* Round the resultant value to the nearest integer*/\
  res_1 = _mm256_round_ps(res_1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
  res_2 = _mm256_round_ps(res_2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\
\
  /* Convert float32 scaled rounded value to int32 */\
  temp_32[0] = _mm256_cvtps_epi32(res_1);\
  temp_32[1] = _mm256_cvtps_epi32(res_2);\
\
  /* Convert the s32 to s16 */\
	c_int16__p0 = _mm256_packs_epi32(temp_32[0], temp_32[1]);\
\
  /*Permute to make sure the order is correct*/\
	c_int16__p0 = _mm256_permute4x64_epi64(c_int16__p0, 0XD8);\
\
   /* Convert the s16 to s8 */\
	store_reg = _mm256_packs_epi16(c_int16__p0, c_int16__p1);\
  /* Extract the first 128 bits of the register*/\
  temp[0] = _mm256_extractf128_si256(store_reg, 0);\
\
  /* Store the result in s8 form */\
  _mm_storeu_si128((__m128i *)store_buf, temp[0]);\
  memcpy( ( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + 0 ) ) + post_op_c_j + \
	  ( 0 * 16 ) , store_buf, ( n0_rem * sizeof( int8_t ) ) ); \
\

#endif //LPGEMM_S16_KERN_MACROS_H
