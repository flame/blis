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

#ifndef LPGEMM_S32_PACK_MACROS_H
#define LPGEMM_S32_PACK_MACROS_H

/* shift_idx:__m512i*/
#define MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx) \
	/* Multi shift uses indices that corresponds to the bit starting positions
	 * of each of the 8 int4 elements in a given 32 bits, which is 0, 4, 8, 12,
	 * 16, 20, 24, 28. */ \
	shift_idx = _mm512_set1_epi64( 0x1C1814100C080400lu );

/* shift_idx:__m256i*/
#define MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx) \
	/* Multi shift uses indices that corresponds to the bit starting positions
	 * of each of the 8 int4 elements in a given 32 bits, which is 0, 4, 8, 12,
	 * 16, 20, 24, 28. */ \
	shift_idx = _mm256_maskz_set1_epi64( _cvtu32_mask8( 0xFF ), \
					0x1C1814100C080400lu );

/* shift_idx:__m128i*/
#define MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx) \
	/* Multi shift uses indices that corresponds to the bit starting positions
	 * of each of the 8 int4 elements in a given 32 bits, which is 0, 4, 8, 12,
	 * 16, 20, 24, 28. */ \
	shift_idx = _mm_maskz_set1_epi64( _cvtu32_mask8( 0xFF ), \
					0x1C1814100C080400lu );

/* input:__m256i, output: __m512i*/
#define UPSCALE_INT4_TO_INT8_64ELEM_MULTISHIFT(input, output, shift_idx) \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). Unsigned conversion is
	 * used so as to ensure the signed bit in int4 at MSB position of 4
	 * byte group is not modified. */ \
	output = _mm512_multishift_epi64_epi8( shift_idx, \
					_mm512_cvtepu32_epi64( input ) ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm512_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm512_set1_epi8( 0x0F ) );

/* input:__m256i, output: __m512i*/
#define UPSCALE_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(input_0, input_1, \
				output, odd_shift_idx, conv_shift) \
	/* Unsigned conversion is used so as to ensure the signed bit.
     * in int4 at MSB position of 4 byte group is not modified. */ \
	__m512i upscale_input = _mm512_cvtepu32_epi64( input_0 ); \
	__m512i shift_input = _mm512_cvtepu32_epi64( input_1 ); \
 \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). */ \
	output = _mm512_multishift_epi64_epi8( odd_shift_idx, upscale_input ); \
 \
	/* Combine both the input registers, starting from elem[1] till elem[n-1]
	 * in output(without elem[0]), and first non zero element in shift_input.
	 * It is at this point that the first 4bit and last 4bit elements, the 2
	 * that were loaded extra due to byte level access are discarded. */ \
	output = _mm512_permutex2var_epi8( output, conv_shift, shift_input ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm512_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm512_set1_epi8( 0x0F ) );

/* input:__m128i, output: __m256i*/
#define UPSCALE_INT4_TO_INT8_32ELEM_MULTISHIFT(input, output, shift_idx) \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). Unsigned conversion is
	 * used so as to ensure the signed bit in int4 at MSB position of 4
	 * byte group is not modified. */ \
	output = _mm256_multishift_epi64_epi8( shift_idx, \
					_mm256_cvtepu32_epi64( input ) ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm256_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm256_set1_epi8( 0x0F ) );

/* input:__m128i, output: __m256i*/
#define UPSCALE_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(input_0, input_1, \
				output, odd_shift_idx, conv_shift) \
	/* Unsigned conversion is used so as to ensure the signed bit.
     * in int4 at MSB position of 4 byte group is not modified. */ \
	__m256i upscale_input = _mm256_cvtepu32_epi64( input_0 ); \
	__m256i shift_input = _mm256_cvtepu32_epi64( input_1 ); \
 \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). */ \
	output = _mm256_multishift_epi64_epi8( odd_shift_idx, upscale_input ); \
 \
	/* Combine both the input registers, starting from elem[1] till elem[n-1]
	 * in output(without elem[0]), and first non zero element in shift_input.
	 * It is at this point that the first 4bit and last 4bit elements, the 2
	 * that were loaded extra due to byte level access are discarded. */ \
	output = _mm256_permutex2var_epi8( output, conv_shift, shift_input ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm256_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm256_set1_epi8( 0x0F ) );

/* input:int64_t, output: __m128i*/
#define UPSCALE_INT4_TO_INT8_16ELEM_MULTISHIFT(input, output, shift_idx) \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). Unsigned conversion is
	 * used so as to ensure the signed bit in int4 at MSB position of 4
	 * byte group is not modified. */ \
	output = _mm_multishift_epi64_epi8( shift_idx, \
			        _mm_cvtepu32_epi64( input ) ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm_set1_epi8( 0x0F ) );

/* input:int64_t, output:__m128i*/
#define UPSCALE_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(input_0, input_1, \
				output, odd_shift_idx, conv_shift) \
	/* Unsigned conversion is used so as to ensure the signed bit.
     * in int4 at MSB position of 4 byte group is not modified. */ \
	input_0 = _mm_cvtepu32_epi64( input_0 ); \
	input_1 = _mm_cvtepu32_epi64( input_1 ); \
 \
	/* Upscale 32 bits/4 bytes (containing 8 int4 elements) into 64 bit
	 * /8 bytes (containing 8 int8 elements). */ \
	output = _mm_multishift_epi64_epi8( odd_shift_idx, input_0 ); \
 \
	/* Combine both the input registers, starting from elem[1] till elem[n-1]
	 * in output(without elem[0]), and first non zero element in shift_input.
	 * It is at this point that the first 4bit and last 4bit elements, the 2
	 * that were loaded extra due to byte level access are discarded. */ \
	output = _mm_permutex2var_epi8( output, conv_shift, input_1 ); \
 \
	/* The upper 4 bits of each converted int8 element is junk, zeroing it. */ \
	output = _mm_maskz_and_epi64( _cvtu32_mask8( 0xFF ), output, \
							_mm_set1_epi8( 0x0F ) );

#define SIGN_EXTEND_BITWISE_OPS_64ELEM(output, sign_comp) \
	/* Comparison of signed bit in int4 and appending sign bits. */ \
	/* Set 4th bit (bit[3]/MSB/sign bit) of negative int4 values (signed bit
	 * is 1) to 1 and rest every other bits to 0. */ \
	__m512i hi_bits_512 = _mm512_and_epi32( output, sign_comp ); \
 \
	/* Set 4th bit (bit[3]/MSB/sign bit) of positive int4 values (signed bit
	 * is 0) to 1 and rest every other bits to 0. */ \
	hi_bits_512 = _mm512_xor_epi32( hi_bits_512, sign_comp ); \
 \
	/* Set the sign extension bits on an int8_t size basis, this will then be
	 * OR with output to get the signed outputs. */ \
	hi_bits_512 = _mm512_add_epi8( hi_bits_512, _mm512_set1_epi8( 0xF8 ) ); \
 \
	output = _mm512_or_epi32( output, hi_bits_512 );

#define SIGN_EXTEND_BITWISE_OPS_32ELEM(output, sign_comp) \
	/* Comparison of signed bit in int4 and appending sign bits. */ \
	/* Set 4th bit (bit[3]/MSB/sign bit) of negative int4 values (signed bit
	 * is 1) to 1 and rest every other bits to 0. */ \
	__m256i hi_bits_256 = _mm256_maskz_and_epi32( _cvtu32_mask8( 0xFF ),\
					output, sign_comp ); \
 \
	/* Set 4th bit (bit[3]/MSB/sign bit) of positive int4 values (signed bit
	 * is 0) to 1 and rest every other bits to 0. */ \
	hi_bits_256 = _mm256_xor_epi32( hi_bits_256, sign_comp ); \
 \
	/* Set the sign extension bits on an int8_t size basis, this will then be
	 * OR with output to get the signed outputs. */ \
	hi_bits_256 = _mm256_add_epi8( hi_bits_256, _mm256_set1_epi8( 0xF8 ) ); \
 \
	output = _mm256_or_epi32( output, hi_bits_256 );

#define SIGN_EXTEND_BITWISE_OPS_16ELEM(output, sign_comp) \
	/* Comparison of signed bit in int4 and appending sign bits. */ \
	/* Set 4th bit (bit[3]/MSB/sign bit) of negative int4 values (signed bit
	 * is 1) to 1 and rest every other bits to 0. */ \
	__m128i hi_bits_128 = _mm_maskz_and_epi32( _cvtu32_mask8( 0xFF ),\
					output, sign_comp ); \
 \
	/* Set 4th bit (bit[3]/MSB/sign bit) of positive int4 values (signed bit
	 * is 0) to 1 and rest every other bits to 0. */ \
	hi_bits_128 = _mm_xor_epi32( hi_bits_128, sign_comp ); \
 \
	/* Set the sign extension bits on an int8_t size basis, this will then be
	 * OR with output to get the signed outputs. */ \
	hi_bits_128 = _mm_add_epi8( hi_bits_128, _mm_set1_epi8( 0xF8 ) ); \
 \
	output = _mm_or_epi32( output, hi_bits_128 );

/* input:__m256i, output: __m512i*/
#define CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(input, output, shift_idx, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_64ELEM_MULTISHIFT(input, output, shift_idx); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_64ELEM(output, sign_comp); \
	} \
} while (0);

/* input:__m256i, output: __m512i*/
#define CVT_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
		odd_shift_idx, conv_shift, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
				odd_shift_idx, conv_shift); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_64ELEM(output, sign_comp); \
	} \
} while (0);

/* input:__m128i, output: __m256i*/
#define CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(input, output, shift_idx, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_32ELEM_MULTISHIFT(input, output, shift_idx); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_32ELEM(output, sign_comp); \
	} \
} while (0);

/* input:__m128i, output: __m256i*/
#define CVT_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
		odd_shift_idx, conv_shift, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
				odd_shift_idx, conv_shift); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_32ELEM(output, sign_comp); \
	} \
} while (0);

/* input:int64_t, output: __m128i*/
#define CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(input, output, shift_idx, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_16ELEM_MULTISHIFT(input, output, shift_idx); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_16ELEM(output, sign_comp); \
	} \
} while (0);

/* input:int64_t, output: __m128i*/
#define CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
		odd_shift_idx, conv_shift, sign_comp, signed_scale) \
do { \
	UPSCALE_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(input_0, input_1, output, \
				odd_shift_idx, conv_shift); \
 \
	if ( signed_scale == TRUE ) \
	{ \
		SIGN_EXTEND_BITWISE_OPS_16ELEM(output, sign_comp); \
	} \
} while (0);

#define LOAD_16_COLS_AVX512                                     \
    a_reg[0] = _mm512_loadu_si512(b + (ldb * (jr + 0)) + kr);   \
    a_reg[1] = _mm512_loadu_si512(b + (ldb * (jr + 1)) + kr);   \
    a_reg[2] = _mm512_loadu_si512(b + (ldb * (jr + 2)) + kr);   \
    a_reg[3] = _mm512_loadu_si512(b + (ldb * (jr + 3)) + kr);   \
    a_reg[4] = _mm512_loadu_si512(b + (ldb * (jr + 4)) + kr);   \
    a_reg[5] = _mm512_loadu_si512(b + (ldb * (jr + 5)) + kr);   \
    a_reg[6] = _mm512_loadu_si512(b + (ldb * (jr + 6)) + kr);   \
    a_reg[7] = _mm512_loadu_si512(b + (ldb * (jr + 7)) + kr);   \
    a_reg[8] = _mm512_loadu_si512(b + (ldb * (jr + 8)) + kr);   \
    a_reg[9] = _mm512_loadu_si512(b + (ldb * (jr + 9)) + kr);   \
    a_reg[10] = _mm512_loadu_si512(b + (ldb * (jr + 10)) + kr); \
    a_reg[11] = _mm512_loadu_si512(b + (ldb * (jr + 11)) + kr); \
    a_reg[12] = _mm512_loadu_si512(b + (ldb * (jr + 12)) + kr); \
    a_reg[13] = _mm512_loadu_si512(b + (ldb * (jr + 13)) + kr); \
    a_reg[14] = _mm512_loadu_si512(b + (ldb * (jr + 14)) + kr); \
    a_reg[15] = _mm512_loadu_si512(b + (ldb * (jr + 15)) + kr);

#define UNPACKHILO32_AVX512                                  \
    b_reg[0] = _mm512_unpacklo_epi32(a_reg[0], a_reg[1]);    \
    b_reg[2] = _mm512_unpacklo_epi32(a_reg[2], a_reg[3]);    \
    b_reg[4] = _mm512_unpacklo_epi32(a_reg[4], a_reg[5]);    \
    b_reg[6] = _mm512_unpacklo_epi32(a_reg[6], a_reg[7]);    \
    b_reg[8] = _mm512_unpacklo_epi32(a_reg[8], a_reg[9]);    \
    b_reg[10] = _mm512_unpacklo_epi32(a_reg[10], a_reg[11]); \
    b_reg[12] = _mm512_unpacklo_epi32(a_reg[12], a_reg[13]); \
    b_reg[14] = _mm512_unpacklo_epi32(a_reg[14], a_reg[15]); \
                                                             \
    b_reg[1] = _mm512_unpackhi_epi32(a_reg[0], a_reg[1]);    \
    b_reg[3] = _mm512_unpackhi_epi32(a_reg[2], a_reg[3]);    \
    b_reg[5] = _mm512_unpackhi_epi32(a_reg[4], a_reg[5]);    \
    b_reg[7] = _mm512_unpackhi_epi32(a_reg[6], a_reg[7]);    \
    b_reg[9] = _mm512_unpackhi_epi32(a_reg[8], a_reg[9]);    \
    b_reg[11] = _mm512_unpackhi_epi32(a_reg[10], a_reg[11]); \
    b_reg[13] = _mm512_unpackhi_epi32(a_reg[12], a_reg[13]); \
    b_reg[15] = _mm512_unpackhi_epi32(a_reg[14], a_reg[15]);

#define UNPACKHILO64_AVX512                                  \
    a_reg[0] = _mm512_unpacklo_epi64(b_reg[0], b_reg[2]);    \
    a_reg[1] = _mm512_unpacklo_epi64(b_reg[4], b_reg[6]);    \
    a_reg[2] = _mm512_unpacklo_epi64(b_reg[8], b_reg[10]);   \
    a_reg[3] = _mm512_unpacklo_epi64(b_reg[12], b_reg[14]);  \
    a_reg[4] = _mm512_unpacklo_epi64(b_reg[1], b_reg[3]);    \
    a_reg[5] = _mm512_unpacklo_epi64(b_reg[5], b_reg[7]);    \
    a_reg[6] = _mm512_unpacklo_epi64(b_reg[9], b_reg[11]);   \
    a_reg[7] = _mm512_unpacklo_epi64(b_reg[13], b_reg[15]);  \
                                                             \
    a_reg[8] = _mm512_unpackhi_epi64(b_reg[0], b_reg[2]);    \
    a_reg[9] = _mm512_unpackhi_epi64(b_reg[4], b_reg[6]);    \
    a_reg[10] = _mm512_unpackhi_epi64(b_reg[8], b_reg[10]);  \
    a_reg[11] = _mm512_unpackhi_epi64(b_reg[12], b_reg[14]); \
    a_reg[12] = _mm512_unpackhi_epi64(b_reg[1], b_reg[3]);   \
    a_reg[13] = _mm512_unpackhi_epi64(b_reg[5], b_reg[7]);   \
    a_reg[14] = _mm512_unpackhi_epi64(b_reg[9], b_reg[11]);  \
    a_reg[15] = _mm512_unpackhi_epi64(b_reg[13], b_reg[15]);

#define PERMUTEX2_VAR64_AVX512                                              \
    b_reg[0] = _mm512_permutex2var_epi64(a_reg[0], selector1, a_reg[1]);    \
    b_reg[1] = _mm512_permutex2var_epi64(a_reg[2], selector1, a_reg[3]);    \
    b_reg[2] = _mm512_permutex2var_epi64(a_reg[8], selector1, a_reg[9]);    \
    b_reg[3] = _mm512_permutex2var_epi64(a_reg[10], selector1, a_reg[11]);  \
    b_reg[4] = _mm512_permutex2var_epi64(a_reg[4], selector1, a_reg[5]);    \
    b_reg[5] = _mm512_permutex2var_epi64(a_reg[6], selector1, a_reg[7]);    \
    b_reg[6] = _mm512_permutex2var_epi64(a_reg[12], selector1, a_reg[13]);  \
    b_reg[7] = _mm512_permutex2var_epi64(a_reg[14], selector1, a_reg[15]);  \
    b_reg[8] = _mm512_permutex2var_epi64(a_reg[0], selector2, a_reg[1]);    \
    b_reg[9] = _mm512_permutex2var_epi64(a_reg[2], selector2, a_reg[3]);    \
    b_reg[10] = _mm512_permutex2var_epi64(a_reg[8], selector2, a_reg[9]);   \
    b_reg[11] = _mm512_permutex2var_epi64(a_reg[10], selector2, a_reg[11]); \
    b_reg[12] = _mm512_permutex2var_epi64(a_reg[4], selector2, a_reg[5]);   \
    b_reg[13] = _mm512_permutex2var_epi64(a_reg[6], selector2, a_reg[7]);   \
    b_reg[14] = _mm512_permutex2var_epi64(a_reg[12], selector2, a_reg[13]); \
    b_reg[15] = _mm512_permutex2var_epi64(a_reg[14], selector2, a_reg[15]);

#define SHUFFLE64x2_AVX512                                        \
    a_reg[0] = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0x44);    \
    a_reg[1] = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0x44);    \
    a_reg[2] = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0x44);    \
    a_reg[3] = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0x44);    \
    a_reg[4] = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0x44);    \
    a_reg[5] = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0x44);  \
    a_reg[6] = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0x44);  \
    a_reg[7] = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0x44);  \
    a_reg[8] = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0xEE);    \
    a_reg[9] = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0xEE);    \
    a_reg[10] = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0xEE);   \
    a_reg[11] = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0xEE);   \
    a_reg[12] = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0xEE);   \
    a_reg[13] = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0xEE); \
    a_reg[14] = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0xEE); \
    a_reg[15] = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0xEE);

#define MASK_LOAD_16_COLS_AVX512(mask)                                      \
    a_reg[0] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 0)) + kr);   \
    a_reg[1] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 1)) + kr);   \
    a_reg[2] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 2)) + kr);   \
    a_reg[3] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 3)) + kr);   \
    a_reg[4] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 4)) + kr);   \
    a_reg[5] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 5)) + kr);   \
    a_reg[6] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 6)) + kr);   \
    a_reg[7] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 7)) + kr);   \
    a_reg[8] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 8)) + kr);   \
    a_reg[9] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 9)) + kr);   \
    a_reg[10] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 10)) + kr); \
    a_reg[11] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 11)) + kr); \
    a_reg[12] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 12)) + kr); \
    a_reg[13] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 13)) + kr); \
    a_reg[14] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 14)) + kr); \
    a_reg[15] = _mm512_maskz_loadu_epi8(mask, b + (ldb * (jr + 15)) + kr);

#endif //LPGEMM_S32_PACK_MACROS_H
