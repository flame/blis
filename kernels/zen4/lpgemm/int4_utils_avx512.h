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

#ifndef LPGEMM_INT4_CVT_UTILS_H
#define LPGEMM_INT4_CVT_UTILS_H

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

#define CREATE_CVT_INT4_INT8_PERM_IDX_64ELEM_ODD_LD(var_name) \
    const int64_t var_name[8] = { \
                    0x0807060504030201, 0x100F0E0D0C0B0A09, \
                    0X1817161514131211, 0X201F1E1D1C1B1A19, \
                    0X2827262524232221, 0X302F2E2D2C2B2A29, \
                    0X3837363534333231, 0X7B3F3E3D3C3B3A39 };

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

#define CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(var_name) \
    const int64_t var_name[4] = { \
                    0x0807060504030201, 0x100F0E0D0C0B0A09, \
                    0X1817161514131211, 0X3B1F1E1D1C1B1A19 };

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

#define CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(var_name) \
    const int64_t var_name[2] = { \
                    0x0807060504030201, 0x1B0F0E0D0C0B0A09 };

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

#define CREATE_CVT_INT8_INT4_PERM_IDX_64ELEM_2_ZMM_REG(var_name) \
	int8_t var_name[64] __attribute__((aligned(64))) = \
		{0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, \
		 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E, \
		 0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E, \
		 0x30, 0x32, 0x34, 0x36, 0x38, 0x3A, 0x3C, 0x3E, \
		 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A, 0x4C, 0x4E, \
		 0x50, 0x52, 0x54, 0x56, 0x58, 0x5A, 0x5C, 0x5E, \
		 0x60, 0x62, 0x64, 0x66, 0x68, 0x6A, 0x6C, 0x6E, \
		 0x70, 0x72, 0x74, 0x76, 0x78, 0x7A, 0x7C, 0x7E};

/* Conversion from int8 to int4. First split the elements in __m512i
 * register at even indices and odd indices into two separate __m256i
 * even and odd registers. Then shift the elements in odd by 4 to the
 * left and OR with even register. */
/* input_*:__m512i, output: __m512i */
#define CVT_INT8_INT4_64ELEM_2_ZMM_REG(input_0, input_1, output, \
		even_perm_idx, odd_perm_idx, clear_hi_bits) \
do { \
	output = _mm512_permutex2var_epi8( input_0, even_perm_idx, input_1 ); \
	__m512i odd_out = _mm512_permutex2var_epi8( input_0, \
									odd_perm_idx, input_1 ); \
 \
	/* Ensure the hi 4 bits are cleared. */ \
	output = _mm512_and_epi32( output, clear_hi_bits ); \
 \
	__m256i odd1_256 = _mm512_extracti64x4_epi64( odd_out, 0x0 ); \
	__m256i odd2_256 = _mm512_extracti64x4_epi64( odd_out, 0x1 ); \
 \
	/* Shift the elemts in odd register by 4 to the left. */ \
	odd1_256 = _mm512_cvtepi16_epi8( \
		_mm512_slli_epi16( _mm512_cvtepu8_epi16( odd1_256 ), 0x4 ) ); \
	odd2_256 = _mm512_cvtepi16_epi8( \
		_mm512_slli_epi16( _mm512_cvtepu8_epi16( odd2_256 ), 0x4 ) ); \
 \
	odd_out = _mm512_castsi256_si512( odd1_256 ); \
	odd_out = _mm512_inserti64x4( odd_out, odd2_256, 0x01 ); \
 \
	output = _mm512_or_epi32( output, odd_out ); \
} while (0);

#define CREATE_CVT_INT8_INT4_PERM_IDX_32ELEM_2_YMM_REG(var_name) \
	int8_t var_name[32] __attribute__((aligned(64))) = \
		{0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, \
		 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E, \
		 0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E, \
		 0x30, 0x32, 0x34, 0x36, 0x38, 0x3A, 0x3C, 0x3E};

/* input_*:__m256i, output: __m256i */
#define CVT_INT8_INT4_32ELEM_2_YMM_REG(input_0, input_1, output, \
		even_perm_idx, odd_perm_idx, clear_hi_bits) \
do { \
	output = _mm256_permutex2var_epi8( input_0, even_perm_idx, input_1 ); \
	__m256i odd_out = _mm256_permutex2var_epi8( input_0, \
									odd_perm_idx, input_1 ); \
 \
	/* Ensure the hi 4 bits are cleared. */ \
	output = _mm256_maskz_and_epi32( _cvtu32_mask8( 0xFF ), \
					output, clear_hi_bits ); \
 \
	/* Shift the elemts in odd register by 4 to the left. */ \
	odd_out = _mm512_cvtepi16_epi8( \
	_mm512_slli_epi16( _mm512_cvtepu8_epi16( odd_out ), 0x4 ) ); \
 \
	output = _mm256_or_epi32( output, odd_out ); \
} while (0);

#define CREATE_CVT_INT8_INT4_PERM_IDX_16ELEM_2_XMM_REG(var_name) \
	int8_t var_name[16] __attribute__((aligned(64))) = \
		{0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, \
		 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E};

/* input_*:__m128i, output: __m128i */
#define CVT_INT8_INT4_16ELEM_2_XMM_REG(input_0, input_1, output, \
		even_perm_idx, odd_perm_idx, clear_hi_bits) \
do { \
	output = _mm_permutex2var_epi8( input_0, even_perm_idx, input_1 ); \
	__m128i odd_out = _mm_permutex2var_epi8( input_0, \
									odd_perm_idx, input_1 ); \
 \
	/* Ensure the hi 4 bits are cleared. */ \
	output = _mm_maskz_and_epi32( _cvtu32_mask8( 0xFF ), \
					output, clear_hi_bits ); \
 \
	/* Shift the elemts in odd register by 4 to the left. */ \
	__mmask16 sel_all_mask = _cvtu32_mask16( 0xFFFF ); \
	odd_out = _mm256_maskz_cvtepi16_epi8( sel_all_mask, \
		_mm256_maskz_slli_epi16( sel_all_mask, \
		  _mm256_maskz_cvtepu8_epi16( sel_all_mask, odd_out ), 0x4 ) ); \
 \
	output = _mm_or_epi32( output, odd_out ); \
} while (0);


#define CVT_INT8_F32_SCAL_16( in, idx, scale_reg) \
    (_mm512_mul_ps( \
      _mm512_cvtepi32_ps( \
       _mm512_cvtepi8_epi32( \
        _mm512_extracti32x4_epi32( in, idx ) ) ), scale_reg ) )

#define CVT_INT8_F32_SCAL_8( in, idx, scale_reg) \
    (_mm512_mul_ps( \
      _mm512_cvtepi32_ps( \
       _mm512_cvtepi8_epi32( \
        _mm256_extracti32x4_epi32( in, idx ) ) ), scale_reg ) )


#endif //LPGEMM_INT4_CVT_UTILS_H
