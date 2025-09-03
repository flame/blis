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

#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "../int4_utils_avx512.h"

void packb_nr64_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   NC,
      const dim_t   KC,
      dim_t*        rs_p,
      dim_t*        cs_p,
      lpgemm_pre_op* pre_op,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nr64_bf16s4f32of32_col_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   NC,
      const dim_t   KC,
      dim_t*        rs_p,
      dim_t*        cs_p,
      lpgemm_pre_op* pre_op,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nr48_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nr32_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nr16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nrlt16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      const dim_t n0_partial_rem,
      AOCL_MATRIX_TYPE mtag
    );

void packb_nr64_bf16s4f32of32
     (
       int8_t*       pack_b_buffer,
       const int8_t* b,
       const dim_t   rs_b,
       const dim_t   cs_b,
       const dim_t   NC,
       const dim_t   KC,
       dim_t*        rs_p,
       dim_t*        cs_p,
       lpgemm_pre_op* pre_op,
       AOCL_MATRIX_TYPE mtag
     )
{
	if (cs_b == 1)
	{
		packb_nr64_bf16s4f32of32_row_major
		(
			pack_b_buffer, b, rs_b, NC,
			KC, rs_p, cs_p, pre_op, mtag
		);
	}
	else
	{
		packb_nr64_bf16s4f32of32_col_major
		(
			pack_b_buffer, b, cs_b, NC, KC,
			rs_p, cs_p, pre_op, mtag
		);
	}
}

void packb_nr64_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   NC,
      const dim_t   KC,
      dim_t*        rs_p,
      dim_t*        cs_p,
      lpgemm_pre_op* pre_op,
      AOCL_MATRIX_TYPE mtag
    )
{
	dim_t NR = 64;

	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	// KC when not multiple of 2 will have padding to make it multiple of 2
	// in packed buffer.
	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
		KC_updated += ( 2 - k_partial_pieces );
	}

	bool is_odd_stride = ( ( rs_b % 2 ) == 0 ) ? FALSE : TRUE;
	bool signed_upscale = TRUE;
	const dim_t incr_adj_factor = 2; // (Byte / 2) for int4 increments.

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	// Selectors for int4 -> int8 conversion.
	__m512i shift_idx_64;

	if(mtag == AWQ_B_MATRIX)
	{
		MULTISHIFT_AWQ_32BIT_8_INT4_IDX_64ELEM(shift_idx_64)
	}
	else
	{
		MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64)
	}

	__m512i sign_comp = _mm512_set1_epi8( 0x08 );
	__mmask32 hmask = _cvtu32_mask32(0xFFFFFFFF); // 32 bytes or 64 int4.
	__mmask32 hmask_odd = _cvtu32_mask32(0x80000000); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_64ELEM_ODD_LD(conv_shift_arr);
	__m512i conv_shift = _mm512_loadu_epi64(conv_shift_arr);

	// Selectors for int8 -> int4 conversion.
	CREATE_CVT_INT8_INT4_PERM_IDX_64ELEM_2_ZMM_REG(even_idx_arr)
	__m512i even_perm_idx = _mm512_loadu_si512( even_idx_arr );
	__m512i all_1s = _mm512_maskz_set1_epi8( _cvtu64_mask64( 0xFFFFFFFFFFFFFFFF ), 0x01 );
	__m512i odd_perm_idx = _mm512_add_epi8( even_perm_idx, all_1s );
	__m512i clear_hi_bits = _mm512_maskz_set1_epi8( _cvtu64_mask64( 0xFFFFFFFFFFFFFFFF ), 0x0F );

	__m256i h_a0;
	__m256i h_b0;
	__m256i h_b0_l4bit;

	__m512i a0;
	__m512i b0;
	__m512i r_lo;
	__m512i r_hi;
	__m512i s4_out;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
		{
			// Int4 array has to be accessed like byte array, but with
			// half the elements traversed in the byte array.
			h_a0 = _mm256_maskz_loadu_epi8( hmask,
				b + ( ( ( rs_b * ( kr + 0 ) ) + jc ) / incr_adj_factor ) );
			CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(h_a0, a0, shift_idx_64, \
				sign_comp, signed_upscale);

			// If the stride, i.e. rs_b is odd, then the stride increment
			// (rs_b * ...)/2 will point at the byte of which the high 4
			// bits is our desired starting element. However since data
			// access is at byte level, the low 4 bits of this byte will
			// be wrongly included, and additionally the last int4 element
			// won't be included either. Extra data movement done to
			// account for the same.
			// Since kr is a multiple of 2, only kr+1 will have the
			// aforementioned issue.
			if ( is_odd_stride == FALSE )
			{
				h_b0 = _mm256_maskz_loadu_epi8( hmask,
					b + ( ( ( rs_b * ( kr + 1 ) ) + jc ) / incr_adj_factor ) );
				CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(h_b0, b0, shift_idx_64, \
					sign_comp, signed_upscale);
			}
			else
			{
				h_b0 = _mm256_maskz_loadu_epi8( hmask,
					b + ( ( ( rs_b * ( kr + 1 ) ) + jc ) / incr_adj_factor ) );
				// Only load the last byte/ 32nd byte.
				h_b0_l4bit = _mm256_maskz_loadu_epi8( hmask_odd,
					b + ( ( ( rs_b * ( kr + 1 ) ) + jc ) / incr_adj_factor ) + 1 );
				CVT_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(h_b0, h_b0_l4bit, b0, \
					shift_idx_64, conv_shift, sign_comp, signed_upscale);
			}

			// Restructuring at int8 level.
			r_lo = _mm512_unpacklo_epi8( a0, b0 );
			r_hi = _mm512_unpackhi_epi8( a0, b0 );

			a0 = _mm512_permutex2var_epi64( r_lo, selector1, r_hi );
			b0 = _mm512_permutex2var_epi64( r_lo, selector1_1, r_hi );

			// To be converted to int4 for storing.
			CVT_INT8_INT4_64ELEM_2_ZMM_REG(a0, b0, s4_out, \
					even_perm_idx, odd_perm_idx, clear_hi_bits);

			// Int4 array has to be accessed like byte array, but with
			// half the elements traversed in the byte array.
			_mm512_storeu_si512( pack_b_buffer +
				( ( ( jc * KC_updated ) + ( kr * NR ) ) / incr_adj_factor ),
				s4_out );
		}
		// Handle k remainder.
		if( k_partial_pieces > 0)
		{
			h_a0 = _mm256_maskz_loadu_epi8( hmask,
				b + ( ( ( rs_b * ( k_full_pieces + 0 ) ) + jc ) /
					  incr_adj_factor ) );
			CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(h_a0, a0, shift_idx_64, \
				sign_comp, signed_upscale);

			b0 = _mm512_setzero_si512();

			// Restructuring at int8 level.
			r_lo = _mm512_unpacklo_epi8( a0, b0 );
			r_hi = _mm512_unpackhi_epi8( a0, b0 );

			a0 = _mm512_permutex2var_epi64( r_lo, selector1, r_hi );
			b0 = _mm512_permutex2var_epi64( r_lo, selector1_1, r_hi );

			// To be converted to int4 for storing.
			CVT_INT8_INT4_64ELEM_2_ZMM_REG(a0, b0, s4_out, \
					even_perm_idx, odd_perm_idx, clear_hi_bits);

			_mm512_storeu_si512( pack_b_buffer +
				( ( ( jc * KC_updated ) + ( k_full_pieces * NR ) ) /
				  incr_adj_factor ), s4_out );
		}
	}

	if(n_partial_pieces > 0)
	{
		dim_t n0_partial_rem = n_partial_pieces % 16;
		dim_t n0_partial_pack = 0;

		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(64) can be expressed
		// as n0 = 48 + n` / n0 = 32 + n` / n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n_partial_pieces / 48;
		dim_t n0_32 = n_partial_pieces / 32;
		dim_t n0_16 = n_partial_pieces / 16;

		if ( n0_48 == 1 )
		{
			packb_nr48_bf16s4f32of32_row_major
			(
			  ( pack_b_buffer +
				( ( n_full_pieces_loop_limit * KC_updated ) /
				  incr_adj_factor ) ),
			  ( b + ( n_full_pieces_loop_limit / incr_adj_factor ) ),
			  rs_b, KC, mtag
			);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr32_bf16s4f32of32_row_major
			(
			  ( pack_b_buffer +
				( ( n_full_pieces_loop_limit * KC_updated ) /
				  incr_adj_factor ) ),
			  ( b + ( n_full_pieces_loop_limit / incr_adj_factor ) ),
			  rs_b, KC, mtag
			);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr16_bf16s4f32of32_row_major
			(
			  ( pack_b_buffer +
				( ( n_full_pieces_loop_limit * KC_updated ) /
				  incr_adj_factor ) ),
			  ( b + ( n_full_pieces_loop_limit / incr_adj_factor ) ),
			  rs_b, KC, mtag
			);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16s4f32of32_row_major
			(
			  ( pack_b_buffer + ( ( ( n_full_pieces_loop_limit * KC_updated ) +
			    ( n0_partial_pack * KC_updated ) ) / incr_adj_factor ) ),
			  ( b + ( ( n_full_pieces_loop_limit + n0_partial_pack ) /
					  incr_adj_factor ) ),
			  rs_b, KC, n0_partial_rem, mtag
			);
		}
	}
	*rs_p = NR * 2;
	*cs_p = NR / 2;
}

void packb_nr48_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    )
{
	const dim_t NR = 48;
	const dim_t NR_32x2 = 64;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	bool is_odd_stride = ( ( rs_b % 2 ) == 0 ) ? FALSE : TRUE;
	bool signed_upscale = TRUE;
	const dim_t incr_adj_factor = 2; // (Byte / 2) for int4 increments.

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m256i selector1_32 = _mm256_setr_epi64x( 0x0, 0x1, 0x4, 0x5 );
	__m256i selector1_1_32 = _mm256_setr_epi64x( 0x2, 0x3, 0x6, 0x7 );

	// Selectors for int4 -> int8 conversion.
	// First 32 int4 elements selectors.
	__m256i shift_idx_32;

	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );
	__mmask16 hmask_32 = _cvtu32_mask16( 0x0000FFFF ); //16 bytes or 32 int4.
	__mmask16 hmask_odd_32 = _cvtu32_mask16( 0x00008000 ); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(conv_shift_arr_32);
	__m256i conv_shift_32 = _mm256_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_32 );

	// Next 16 int4 elements selectors.
	__m128i shift_idx_16;

	if(mtag == AWQ_B_MATRIX)
	{
		MULTISHIFT_AWQ_32BIT_8_INT4_IDX_32ELEM(shift_idx_32)
		MULTISHIFT_AWQ_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}
	else
	{
		MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32)
		MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}

	__m128i sign_comp_16 = _mm_set1_epi8( 0x08 );
	__mmask16 hmask_16 = _cvtu32_mask16( 0x000000FF ); //8 bytes or 16 int4.
	__mmask16 hmask_odd_16 = _cvtu32_mask16( 0x00000080 ); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
	__m128i conv_shift_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_16 );

	// Selectors for int8 -> int4 conversion.
	// First 32 int8 elements selectors.
	CREATE_CVT_INT8_INT4_PERM_IDX_32ELEM_2_YMM_REG(even_idx_arr_32);
	__m256i even_perm_idx_32 = _mm256_maskz_loadu_epi64( _cvtu32_mask8( 0xFF ),
												even_idx_arr_32 );
	__m256i all_1s_32 = _mm256_maskz_set1_epi8( _cvtu32_mask32( 0xFFFFFFFF ),
												0x01 );
	__m256i odd_perm_idx_32 = _mm256_add_epi8( even_perm_idx_32, all_1s_32 );
	__m256i clear_hi_bits_32 =
			_mm256_maskz_set1_epi8( _cvtu32_mask32( 0xFFFFFFFF ), 0x0F );

	// Next 16 int4 elements selectors.
	CREATE_CVT_INT8_INT4_PERM_IDX_16ELEM_2_XMM_REG(even_idx_arr_16);
	__m128i even_perm_idx_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0xFF ),
												even_idx_arr_16 );
	__m128i all_1s_16 = _mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ),
												0x01 );
	__m128i odd_perm_idx_16 = _mm_add_epi8( even_perm_idx_16, all_1s_16 );
	__m128i clear_hi_bits_16 =
			_mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ), 0x0F );

	__mmask16 sel_all_mask_16 = _cvtu32_mask16( 0xFFFF );

	__m128i h_a0_32;
	__m128i h_b0_32;
	__m128i h_b0_32_l4bit;
	__m128i a0_16;
	__m128i b0_16;
	__m128i r_lo_16;
	__m128i r_hi_16;
	__m128i s4_out_16;
	__m256i a0_32;
	__m256i b0_32;
	__m256i r_lo_32;
	__m256i r_hi_32;
	__m256i s4_out_32;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// First 32 columns.
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_32,
			b + ( ( rs_b * ( kr + 0 ) ) / incr_adj_factor ) );
		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_a0_32, a0_32, shift_idx_32, \
			sign_comp_32, signed_upscale);

		// Last 16 columns.
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( ( rs_b * ( kr + 0 ) ) + 32 ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_32, a0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);

		if ( is_odd_stride == FALSE )
		{
			// First 32 columns.
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) );
			CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_b0_32, b0_32, shift_idx_32, \
				sign_comp_32, signed_upscale);

			// Last 16 columns.
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( ( rs_b * ( kr + 1 ) ) + 32 ) / 2 ) );
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_b0_32, b0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);
		}
		else
		{
			// First 32 columns.
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) );
			// Only load the last byte/ 16th byte.
			h_b0_32_l4bit = _mm_maskz_loadu_epi8( hmask_odd_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) + 1 );
			CVT_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(h_b0_32, h_b0_32_l4bit, \
				b0_32, shift_idx_32, conv_shift_32, sign_comp_32, \
				signed_upscale);

			// Last 16 columns.
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_16,
				  b + ( ( ( rs_b * ( kr + 1 ) ) + 32 ) / 2 ) );
			// Only load the last byte/ 8th byte.
			h_b0_32_l4bit = _mm_maskz_loadu_epi8( hmask_odd_16,
				  b + ( ( ( rs_b * ( kr + 1 ) ) + 32 ) / 2 ) + 1 );
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(h_b0_32, h_b0_32_l4bit, \
				b0_16, shift_idx_16, conv_shift_16, sign_comp_16, \
				signed_upscale);
		}

		// Restructuring at int8 level.
		// First 32 columns.
		r_lo_32 = _mm256_unpacklo_epi8( a0_32, b0_32 );
		r_hi_32 = _mm256_unpackhi_epi8( a0_32, b0_32 );

		a0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_32, r_hi_32 );
		b0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_1_32, r_hi_32 );

		CVT_INT8_INT4_32ELEM_2_YMM_REG(a0_32, b0_32, s4_out_32, \
			even_perm_idx_32, odd_perm_idx_32, clear_hi_bits_32);

		_mm256_storeu_epi64( pack_b_buffer +
			( ( kr * NR ) / incr_adj_factor ), s4_out_32 );

		// Last 16 columns.
		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( ( kr * NR ) + NR_32x2 ) / incr_adj_factor ), s4_out_16 );
	}
	// Handle k remainder.
	if( k_partial_pieces > 0)
	{
		// First 32 columns.
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_32,
			b + ( ( rs_b * ( k_full_pieces + 0 ) ) / incr_adj_factor ) );
		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_a0_32, a0_32, shift_idx_32, \
			sign_comp_32, signed_upscale);
		b0_32 = _mm256_setzero_si256();

		r_lo_32 = _mm256_unpacklo_epi8( a0_32, b0_32 );
		r_hi_32 = _mm256_unpackhi_epi8( a0_32, b0_32 );

		a0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_32, r_hi_32 );
		b0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_1_32, r_hi_32 );

		CVT_INT8_INT4_32ELEM_2_YMM_REG(a0_32, b0_32, s4_out_32, \
			even_perm_idx_32, odd_perm_idx_32, clear_hi_bits_32);

		_mm256_storeu_epi64( pack_b_buffer +
			( ( k_full_pieces * NR ) / incr_adj_factor ), s4_out_32 );

		// Last 16 columns.
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_16,
			b + ( ( ( rs_b * ( k_full_pieces + 0 ) ) + 32 ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_32, a0_16, shift_idx_16, \
			sign_comp_16, signed_upscale);
		b0_16 = _mm_setzero_si128();

		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( ( k_full_pieces * NR ) + NR_32x2 ) / incr_adj_factor ), s4_out_16 );
	}
}

void packb_nr32_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    )
{
	const dim_t NR = 32;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	bool is_odd_stride = ( ( rs_b % 2 ) == 0 ) ? FALSE : TRUE;
	bool signed_upscale = TRUE;
	const dim_t incr_adj_factor = 2; // (Byte / 2) for int4 increments.

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m256i selector1_32 = _mm256_setr_epi64x( 0x0, 0x1, 0x4, 0x5 );
	__m256i selector1_1_32 = _mm256_setr_epi64x( 0x2, 0x3, 0x6, 0x7 );

	// Selectors for int4 -> int8 conversion.
	__m256i shift_idx_32;

	if(mtag == AWQ_B_MATRIX)
	{
		 MULTISHIFT_AWQ_32BIT_8_INT4_IDX_32ELEM(shift_idx_32)
	}
	else
	{
		MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32)
	}

	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );
	__mmask16 hmask_32 = _cvtu32_mask16( 0x0000FFFF ); //16 bytes or 32 int4.
	__mmask16 hmask_odd_32 = _cvtu32_mask16( 0x00008000 ); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(conv_shift_arr_32);
	__m256i conv_shift_32 = _mm256_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_32 );

	// Selectors for int8 -> int4 conversion.
	CREATE_CVT_INT8_INT4_PERM_IDX_32ELEM_2_YMM_REG(even_idx_arr_32);
	__m256i even_perm_idx_32 = _mm256_maskz_loadu_epi64( _cvtu32_mask8( 0xFF ),
												even_idx_arr_32 );
	__m256i all_1s_32 = _mm256_maskz_set1_epi8( _cvtu32_mask32( 0xFFFFFFFF ),
												0x01 );
	__m256i odd_perm_idx_32 = _mm256_add_epi8( even_perm_idx_32, all_1s_32 );
	__m256i clear_hi_bits_32 =
			_mm256_maskz_set1_epi8( _cvtu32_mask32( 0xFFFFFFFF ), 0x0F );

	__m128i h_a0_32;
	__m128i h_b0_32;
	__m128i h_b0_32_l4bit;
	__m256i a0_32;
	__m256i b0_32;
	__m256i r_lo_32;
	__m256i r_hi_32;
	__m256i s4_out_32;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_32,
			b + ( ( rs_b * ( kr + 0 ) ) / incr_adj_factor ) );
		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_a0_32, a0_32, shift_idx_32, \
			sign_comp_32, signed_upscale);

		if ( is_odd_stride == FALSE )
		{
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) );
			CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_b0_32, b0_32, shift_idx_32, \
				sign_comp_32, signed_upscale);
		}
		else
		{
			h_b0_32 = _mm_maskz_loadu_epi8( hmask_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) );
			// Only load the last byte/ 16th byte.
			h_b0_32_l4bit = _mm_maskz_loadu_epi8( hmask_odd_32,
				b + ( ( rs_b * ( kr + 1 ) ) / incr_adj_factor ) + 1 );
			CVT_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(h_b0_32, h_b0_32_l4bit, \
				b0_32, shift_idx_32, conv_shift_32, sign_comp_32, \
				signed_upscale);
		}

		// Restructuring at int8 level.
		// First 32 columns.
		r_lo_32 = _mm256_unpacklo_epi8( a0_32, b0_32 );
		r_hi_32 = _mm256_unpackhi_epi8( a0_32, b0_32 );

		a0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_32, r_hi_32 );
		b0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_1_32, r_hi_32 );

		CVT_INT8_INT4_32ELEM_2_YMM_REG(a0_32, b0_32, s4_out_32, \
			even_perm_idx_32, odd_perm_idx_32, clear_hi_bits_32);

		_mm256_storeu_epi64( pack_b_buffer +
			( ( kr * NR ) / incr_adj_factor ), s4_out_32 );
	}
	// Handle k remainder.
	if( k_partial_pieces > 0)
	{
		h_a0_32 = _mm_maskz_loadu_epi8( hmask_32,
			b + ( ( rs_b * ( k_full_pieces + 0 ) ) / incr_adj_factor ) );
		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(h_a0_32, a0_32, shift_idx_32, \
			sign_comp_32, signed_upscale);
		b0_32 = _mm256_setzero_si256();

		r_lo_32 = _mm256_unpacklo_epi8( a0_32, b0_32 );
		r_hi_32 = _mm256_unpackhi_epi8( a0_32, b0_32 );

		a0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_32, r_hi_32 );
		b0_32 = _mm256_permutex2var_epi64( r_lo_32, selector1_1_32, r_hi_32 );

		CVT_INT8_INT4_32ELEM_2_YMM_REG(a0_32, b0_32, s4_out_32, \
			even_perm_idx_32, odd_perm_idx_32, clear_hi_bits_32);

		_mm256_storeu_epi64( pack_b_buffer +
			( ( k_full_pieces * NR ) / incr_adj_factor ), s4_out_32 );
	}
}

void packb_nr16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      AOCL_MATRIX_TYPE mtag
    )
{
	const dim_t NR = 16;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	bool is_odd_stride = ( ( rs_b % 2 ) == 0 ) ? FALSE : TRUE;
	bool signed_upscale = TRUE;
	const dim_t incr_adj_factor = 2; // (Byte / 2) for int4 increments.

	// Selectors for int4 -> int8 conversion.
	__m128i shift_idx_16;

	if(mtag == AWQ_B_MATRIX)
	{
		MULTISHIFT_AWQ_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}
	else
	{
		 MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}

	__m128i sign_comp_16 = _mm_set1_epi8( 0x08 );
	__mmask16 hmask_16 = _cvtu32_mask16( 0x000000FF ); //8 bytes or 16 int4.
	__mmask16 hmask_odd_16 = _cvtu32_mask16( 0x00000080 ); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
	__m128i conv_shift_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_16 );

	// Selectors for int8 -> int4 conversion.
	CREATE_CVT_INT8_INT4_PERM_IDX_16ELEM_2_XMM_REG(even_idx_arr_16);
	__m128i even_perm_idx_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0xFF ),
												even_idx_arr_16 );
	__m128i all_1s_16 = _mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ),
												0x01 );
	__m128i odd_perm_idx_16 = _mm_add_epi8( even_perm_idx_16, all_1s_16 );
	__m128i clear_hi_bits_16 =
			_mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ), 0x0F );

	__mmask16 sel_all_mask_16 = _cvtu32_mask16( 0xFFFF );

	__m128i h_a0_16;
	__m128i h_b0_16;
	__m128i h_b0_16_l4bit;
	__m128i a0_16;
	__m128i b0_16;
	__m128i r_lo_16;
	__m128i r_hi_16;
	__m128i s4_out_16;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		h_a0_16 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( rs_b * ( kr + 0 ) ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_16, a0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);

		if ( is_odd_stride == FALSE )
		{
			h_b0_16 = _mm_maskz_loadu_epi8( hmask_16,
					b + ( ( rs_b * ( kr + 1 ) ) / 2 ) );
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_b0_16, b0_16, shift_idx_16, \
					sign_comp_16, signed_upscale);
		}
		else
		{
			h_b0_16 = _mm_maskz_loadu_epi8( hmask_16,
					b + ( ( rs_b * ( kr + 1 ) ) / 2 ) );
			// Only load the last byte/ 8th byte.
			h_b0_16_l4bit = _mm_maskz_loadu_epi8( hmask_odd_16,
					b + ( ( rs_b * ( kr + 1 ) ) / 2 ) + 1 );
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(h_b0_16, h_b0_16_l4bit, \
				b0_16, shift_idx_16, conv_shift_16, sign_comp_16, \
				signed_upscale);
		}

		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( kr * NR ) / incr_adj_factor ), s4_out_16 );
	}
	// Handle k remainder.
	if( k_partial_pieces > 0)
	{
		h_a0_16 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( rs_b * ( k_full_pieces + 0 ) ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_16, a0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);
		b0_16 = _mm_setzero_si128();

		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( k_full_pieces * NR ) / incr_adj_factor ), s4_out_16 );
	}
}

void packb_nrlt16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      const dim_t n0_partial_rem,
      AOCL_MATRIX_TYPE mtag
    )
{
	const dim_t NR = 16;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	bool is_odd_stride = ( ( rs_b % 2 ) == 0 ) ? FALSE : TRUE;
	bool signed_upscale = TRUE;
	const dim_t incr_adj_factor = 2; // (Byte / 2) for int4 increments.

	// Selectors for int4 -> int8 conversion.
	__m128i shift_idx_16;

	if(mtag == AWQ_B_MATRIX)
	{
		MULTISHIFT_AWQ_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}
	else
	{
		MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16)
	}

	__m128i sign_comp_16 = _mm_set1_epi8( 0x08 );
	// 16 int4 elems in 8 bytes, so adjusting the mask for nr < 16 by
	// a factor of 2. In case of odd remainder, the last int4 element
	// within the last byte (hi 4 bits) will be ingnored similar to
	// padding bits.
	__mmask16 hmask_16;
	if ( is_odd_stride == FALSE )
	{
		hmask_16 = _cvtu32_mask16( 0x000000FF >>
				( ( 16 - n0_partial_rem ) / 2 ) );
	}
	else
	{
		if ( ( n0_partial_rem % 2 ) == 0 )
		{
			// An interesting property here is that n0_partial_rem is
			// guaranteed to be < 16. In that case the largest even n0
			// rem would be 14, and the max number of bytes that will be
			// loaded including the extra 4 bit at the beginning will
			// only be 7 bytes out of 8. So in any case loading 1 more
			// byte will bring the last int4 in the register, while not
			// crossing the register boundaries.
			hmask_16 = _cvtu32_mask16( 0x000000FF >>
					( ( ( 16 - n0_partial_rem ) / 2 ) - 1 ) );
		}
		else
		{
			// If the n0 rem is odd, and if the starting position is an odd
			// index, then the last odd element will also be loaded as part
			// of loading the last byte (high 4 bits of last byte).
			hmask_16 = _cvtu32_mask16( 0x000000FF >>
					( ( 16 - n0_partial_rem ) / 2 ) );
		}
	}

	CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
	__m128i conv_shift_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_16 );

	// Selectors for int8 -> int4 conversion.
	CREATE_CVT_INT8_INT4_PERM_IDX_16ELEM_2_XMM_REG(even_idx_arr_16);
	__m128i even_perm_idx_16 = _mm_maskz_loadu_epi64( _cvtu32_mask8( 0xFF ),
												even_idx_arr_16 );
	__m128i all_1s_16 = _mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ),
												0x01 );
	__m128i odd_perm_idx_16 = _mm_add_epi8( even_perm_idx_16, all_1s_16 );
	__m128i clear_hi_bits_16 =
			_mm_maskz_set1_epi8( _cvtu32_mask16( 0xFFFF ), 0x0F );

	__mmask16 sel_all_mask_16 = _cvtu32_mask16( 0xFFFF );

	__m128i h_a0_16;
	__m128i h_b0_16;
	__m128i a0_16;
	__m128i b0_16;
	__m128i r_lo_16;
	__m128i r_hi_16;
	__m128i s4_out_16;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		h_a0_16 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( rs_b * ( kr + 0 ) ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_16, a0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);

		if ( is_odd_stride == FALSE )
		{
			h_b0_16 = _mm_maskz_loadu_epi8( hmask_16,
					b + ( ( rs_b * ( kr + 1 ) ) / 2 ) );
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_b0_16, b0_16, shift_idx_16, \
					sign_comp_16, signed_upscale);
		}
		else
		{
			h_b0_16 = _mm_maskz_loadu_epi8( hmask_16,
					b + ( ( rs_b * ( kr + 1 ) ) / 2 ) );
			// The last int4 elem is already loaded in the previous
			// register. Details given in comments about hmask_16.
			__m128i h_b0_16_l4bit = _mm_setzero_si128();
			CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(h_b0_16, h_b0_16_l4bit, \
				b0_16, shift_idx_16, conv_shift_16, sign_comp_16, \
				signed_upscale);
		}

		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( kr * NR ) / incr_adj_factor ), s4_out_16 );
	}
	// Handle k remainder.
	if( k_partial_pieces > 0)
	{
		h_a0_16 = _mm_maskz_loadu_epi8( hmask_16,
				b + ( ( rs_b * ( k_full_pieces + 0 ) ) / 2 ) );
		CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(h_a0_16, a0_16, shift_idx_16, \
				sign_comp_16, signed_upscale);
		b0_16 = _mm_setzero_si128();

		r_lo_16 = _mm_maskz_unpacklo_epi8( sel_all_mask_16, a0_16, b0_16 );
		r_hi_16 = _mm_maskz_unpackhi_epi8( sel_all_mask_16, a0_16, b0_16 );

		CVT_INT8_INT4_16ELEM_2_XMM_REG(r_lo_16, r_hi_16, s4_out_16, \
					even_perm_idx_16, odd_perm_idx_16, clear_hi_bits_16);

		_mm_storeu_epi64( pack_b_buffer +
			( ( k_full_pieces * NR ) / incr_adj_factor ), s4_out_16 );
	}
}


#define MASK_LOAD_16_COLS_AVX2( mask ) \
	a_reg[0] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) ); \
	a_reg[1] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 1 ) ) + kr ) / 2 ) ); \
	a_reg[2] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 2 ) ) + kr ) / 2 ) ); \
	a_reg[3] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 3 ) ) + kr ) / 2 ) ); \
	a_reg[4] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 4 ) ) + kr ) / 2 ) ); \
	a_reg[5] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 5 ) ) + kr ) / 2 ) ); \
	a_reg[6] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 6 ) ) + kr ) / 2 ) ); \
	a_reg[7] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 7 ) ) + kr ) / 2 ) ); \
	a_reg[8] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 8 ) ) + kr ) / 2 ) ); \
	a_reg[9] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 9 ) ) + kr ) / 2 ) ); \
	a_reg[10] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 10 ) ) + kr ) / 2 ) ); \
	a_reg[11] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 11 ) ) + kr ) / 2 ) ); \
	a_reg[12] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 12 ) ) + kr ) / 2 ) ); \
	a_reg[13] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 13 ) ) + kr ) / 2 ) ); \
	a_reg[14] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 14 ) ) + kr ) / 2 ) ); \
	a_reg[15] = _mm256_maskz_loadu_epi8( mask, ( b + \
				( ( ldb * ( jr + 15 ) ) + kr ) / 2 ) );

/* This order shift transforms a 32 bits sequence of int4 elements
 * representing an AWQ order [0, 2, 4, 6, 1, 3, 5, 7] into a
 * sequential order [0, 1, 2, 3, 4, 5, 6, 7].
 */  \
#define ORDER_SHIFT_AVX2( reg ) \
	MULTISHIFT_AWQ_32BIT_8_INT4_IDX_64ELEM( shift_idx_64 ); \
	CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(reg, a0, shift_idx_64, \
			sign_comp, signed_upscale); \
	even = _mm512_cvtepi16_epi8(a0); \
	odd = _mm512_cvtepi16_epi8( \
		 _mm512_srli_epi16( a0 , 0x4 )); \
	reg = _mm256_or_epi32( even, odd ); \

#define ORDER_SHIFT_AVX2_ALLREG \
	__m512i shift_idx_64; \
	__m512i a0; \
	__m512i sign_comp = _mm512_set1_epi8( 0x08 ); \
	__m256i even; \
	__m256i odd; \
	bool signed_upscale = TRUE; \
	ORDER_SHIFT_AVX2( a_reg[0]) \
	ORDER_SHIFT_AVX2( a_reg[1]) \
	ORDER_SHIFT_AVX2( a_reg[2]) \
	ORDER_SHIFT_AVX2( a_reg[3]) \
	ORDER_SHIFT_AVX2( a_reg[4]) \
	ORDER_SHIFT_AVX2( a_reg[5]) \
	ORDER_SHIFT_AVX2( a_reg[6]) \
	ORDER_SHIFT_AVX2( a_reg[7]) \
	ORDER_SHIFT_AVX2( a_reg[8]) \
	ORDER_SHIFT_AVX2( a_reg[9]) \
	ORDER_SHIFT_AVX2( a_reg[10]) \
	ORDER_SHIFT_AVX2( a_reg[11]) \
	ORDER_SHIFT_AVX2( a_reg[12]) \
	ORDER_SHIFT_AVX2( a_reg[13]) \
	ORDER_SHIFT_AVX2( a_reg[14]) \
	ORDER_SHIFT_AVX2( a_reg[15])

#define UNPACKHILO8_AVX2 \
	b_reg[0] = _mm256_unpacklo_epi8( a_reg[0], a_reg[1] ); \
	b_reg[2] = _mm256_unpacklo_epi8( a_reg[2], a_reg[3] ); \
	b_reg[4] = _mm256_unpacklo_epi8( a_reg[4], a_reg[5] ); \
	b_reg[6] = _mm256_unpacklo_epi8( a_reg[6], a_reg[7] ); \
	b_reg[8] = _mm256_unpacklo_epi8( a_reg[8], a_reg[9] ); \
	b_reg[10] = _mm256_unpacklo_epi8( a_reg[10], a_reg[11] ); \
	b_reg[12] = _mm256_unpacklo_epi8( a_reg[12], a_reg[13] ); \
	b_reg[14] = _mm256_unpacklo_epi8( a_reg[14], a_reg[15] ); \
\
	b_reg[1] = _mm256_unpackhi_epi8( a_reg[0], a_reg[1] ); \
	b_reg[3] = _mm256_unpackhi_epi8( a_reg[2], a_reg[3] ); \
	b_reg[5] = _mm256_unpackhi_epi8( a_reg[4], a_reg[5] ); \
	b_reg[7] = _mm256_unpackhi_epi8( a_reg[6], a_reg[7] ); \
	b_reg[9] = _mm256_unpackhi_epi8( a_reg[8], a_reg[9] ); \
	b_reg[11] = _mm256_unpackhi_epi8( a_reg[10], a_reg[11] ); \
	b_reg[13] = _mm256_unpackhi_epi8( a_reg[12], a_reg[13] ); \
	b_reg[15] = _mm256_unpackhi_epi8( a_reg[14], a_reg[15] );

#define UNPACKHILO16_AVX2 \
	a_reg[0] = _mm256_unpacklo_epi16( b_reg[0], b_reg[2] ); \
	a_reg[1] = _mm256_unpacklo_epi16( b_reg[4], b_reg[6] ); \
	a_reg[2] = _mm256_unpacklo_epi16( b_reg[8], b_reg[10] ); \
	a_reg[3] = _mm256_unpacklo_epi16( b_reg[12], b_reg[14] ); \
	a_reg[4] = _mm256_unpacklo_epi16( b_reg[1], b_reg[3] ); \
	a_reg[5] = _mm256_unpacklo_epi16( b_reg[5], b_reg[7] ); \
	a_reg[6] = _mm256_unpacklo_epi16( b_reg[9], b_reg[11]  ); \
	a_reg[7] = _mm256_unpacklo_epi16( b_reg[13], b_reg[15] ); \
\
	a_reg[8] = _mm256_unpackhi_epi16( b_reg[0], b_reg[2] ); \
	a_reg[9] = _mm256_unpackhi_epi16( b_reg[4], b_reg[6] ); \
	a_reg[10] = _mm256_unpackhi_epi16( b_reg[8], b_reg[10]  ); \
	a_reg[11] = _mm256_unpackhi_epi16( b_reg[12], b_reg[14] ); \
	a_reg[12] = _mm256_unpackhi_epi16( b_reg[1], b_reg[3] ); \
	a_reg[13] = _mm256_unpackhi_epi16( b_reg[5], b_reg[7] ); \
	a_reg[14] = _mm256_unpackhi_epi16( b_reg[9], b_reg[11]  ); \
	a_reg[15] = _mm256_unpackhi_epi16( b_reg[13], b_reg[15] );

#define UNPACKHILO32_AVX2 \
	b_reg[0] = _mm256_unpacklo_epi32( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm256_unpacklo_epi32( a_reg[2], a_reg[3] ); \
	b_reg[2] = _mm256_unpacklo_epi32( a_reg[4], a_reg[5] ); \
	b_reg[3] = _mm256_unpacklo_epi32( a_reg[6], a_reg[7] ); \
	b_reg[4] = _mm256_unpacklo_epi32( a_reg[8], a_reg[9] ); \
	b_reg[5] = _mm256_unpacklo_epi32( a_reg[10], a_reg[11] ); \
	b_reg[6] = _mm256_unpacklo_epi32( a_reg[12], a_reg[13] ); \
	b_reg[7] = _mm256_unpacklo_epi32( a_reg[14], a_reg[15] ); \
\
	b_reg[8] = _mm256_unpackhi_epi32( a_reg[0], a_reg[1]  ); \
	b_reg[9] = _mm256_unpackhi_epi32( a_reg[2], a_reg[3]  ); \
	b_reg[10] = _mm256_unpackhi_epi32( a_reg[4], a_reg[5] ); \
	b_reg[11] = _mm256_unpackhi_epi32( a_reg[6], a_reg[7] ); \
	b_reg[12] = _mm256_unpackhi_epi32( a_reg[8], a_reg[9] ); \
	b_reg[13] = _mm256_unpackhi_epi32( a_reg[10], a_reg[11] ); \
	b_reg[14] = _mm256_unpackhi_epi32( a_reg[12], a_reg[13] ); \
	b_reg[15] = _mm256_unpackhi_epi32( a_reg[14], a_reg[15] );

#define UNPACKHILO64_AVX2 \
	a_reg[0] = _mm256_unpacklo_epi64( b_reg[0], b_reg[1] ); \
	a_reg[1] = _mm256_unpacklo_epi64( b_reg[2], b_reg[3] ); \
	a_reg[2] = _mm256_unpacklo_epi64( b_reg[4], b_reg[5] ); \
	a_reg[3] = _mm256_unpacklo_epi64( b_reg[6], b_reg[7] ); \
	a_reg[4] = _mm256_unpacklo_epi64( b_reg[8], b_reg[9] ); \
	a_reg[5] = _mm256_unpacklo_epi64( b_reg[10], b_reg[11] ); \
	a_reg[6] = _mm256_unpacklo_epi64( b_reg[12], b_reg[13] ); \
	a_reg[7] = _mm256_unpacklo_epi64( b_reg[14], b_reg[15] ); \
\
	a_reg[8] = _mm256_unpackhi_epi64( b_reg[0], b_reg[1] ); \
	a_reg[9] = _mm256_unpackhi_epi64( b_reg[2], b_reg[3] ); \
	a_reg[10] = _mm256_unpackhi_epi64( b_reg[4], b_reg[5] ); \
	a_reg[11] = _mm256_unpackhi_epi64( b_reg[6], b_reg[7] ); \
	a_reg[12] = _mm256_unpackhi_epi64( b_reg[8], b_reg[9] ); \
	a_reg[13] = _mm256_unpackhi_epi64( b_reg[10], b_reg[11] ); \
	a_reg[14] = _mm256_unpackhi_epi64( b_reg[12], b_reg[13] ); \
	a_reg[15] = _mm256_unpackhi_epi64( b_reg[14], b_reg[15] );

//odd cases
#define REMOVE_EXTRA_BITS( reg ) \
	odd_256 = _mm512_cvtepi16_epi8( \
		_mm512_srli_epi16( _mm512_cvtepu8_epi16( reg ), 0x4 ) ); \
	even_256 = _mm512_cvtepi16_epi8( \
		_mm512_slli_epi16( _mm512_cvtepu8_epi16( reg ), 0x4 ) ); \
	even_256 =	_mm256_permutex2var_epi8( odd_256, shift_index, even_256 );\
	reg = _mm256_or_epi32( odd_256, even_256 ); \
\

#define MASK_LOAD_16_COLS_AVX2_ODD( mask1, mask2 ) \
	__m256i shift_index = _mm256_setr_epi8( \
					33, 34, 35, 36, 37, 38, 39, 40, \
					41, 42, 43, 44, 45, 46, 47, 48, \
					49, 50, 51, 52, 53, 54, 55, 56, \
					57, 58, 59, 60, 61, 62, 63, 32 ); \
	__m256i odd_256; \
	__m256i even_256; \
	a_reg[0] = _mm256_maskz_loadu_epi8( mask1,( b + ((  ldb * ( jr + 0 ) ) + kr ) / 2 ) ); \
	a_reg[1] = _mm256_maskz_loadu_epi8( mask2,( b + ((  ldb * ( jr + 1 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[1] )\
	a_reg[2] = _mm256_maskz_loadu_epi8( mask1,( b + ( (  ldb * ( jr + 2 ) ) + kr ) / 2 ) ); \
	a_reg[3] = _mm256_maskz_loadu_epi8( mask2,( b + ( (  ldb * ( jr + 3 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[3] )\
	a_reg[4] = _mm256_maskz_loadu_epi8( mask1,( b + ( (  ldb * ( jr + 4 ) ) + kr ) / 2 ) ); \
	a_reg[5] = _mm256_maskz_loadu_epi8( mask2,( b + ( (  ldb * ( jr + 5 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[5] )\
	a_reg[6] = _mm256_maskz_loadu_epi8( mask1,( b + ( (  ldb * ( jr + 6 ) ) + kr ) / 2 ) ); \
	a_reg[7] = _mm256_maskz_loadu_epi8( mask2,( b + ( (  ldb * ( jr + 7 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[7] )\
	a_reg[8] = _mm256_maskz_loadu_epi8( mask1,( b + ( (  ldb * ( jr + 8 ) ) + kr ) / 2 ) ); \
	a_reg[9] = _mm256_maskz_loadu_epi8( mask2,( b + ( (  ldb * ( jr + 9 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[9] )\
	a_reg[10] = _mm256_maskz_loadu_epi8( mask1, ( b + ( ( ldb * ( jr + 10 ) ) + kr ) / 2 ) ); \
	a_reg[11] = _mm256_maskz_loadu_epi8( mask2, ( b + ( ( ldb * ( jr + 11 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[11] )\
	a_reg[12] = _mm256_maskz_loadu_epi8( mask1, ( b + ( ( ldb * ( jr + 12 ) ) + kr ) / 2 ) ); \
	a_reg[13] = _mm256_maskz_loadu_epi8( mask2, ( b + ( ( ldb * ( jr + 13 ) ) + kr ) / 2 ) ); \
	REMOVE_EXTRA_BITS( a_reg[13] )\
	a_reg[14] = _mm256_maskz_loadu_epi8( mask1, ( b + ( ( ldb * ( jr + 14 ) ) + kr ) / 2 ) ); \
	a_reg[15] = _mm256_maskz_loadu_epi8( mask2, ( b + ( ( ldb * ( jr + 15 ) ) + kr ) / 2 ) );\
	REMOVE_EXTRA_BITS( a_reg[15] )

#define MASK_LOAD_16_COLS_AVX2_LAST_4BITS( mask ) \
	a_reg[0] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) ); \
	a_reg[1] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 1 ) ) + kr ) / 2 ) ); \
	a_reg[2] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 2 ) ) + kr ) / 2 ) ); \
	a_reg[3] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 3 ) ) + kr ) / 2 ) ); \
	a_reg[4] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 4 ) ) + kr ) / 2 ) ); \
	a_reg[5] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 5 ) ) + kr ) / 2 ) ); \
	a_reg[6] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 6 ) ) + kr ) / 2 ) ); \
	a_reg[7] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 7 ) ) + kr ) / 2 ) ); \
	a_reg[8] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 8 ) ) + kr ) / 2 ) ); \
	a_reg[9] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 9 ) ) + kr ) / 2 ) ); \
	a_reg[10] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 10 ) ) + kr ) / 2 ) ); \
	a_reg[11] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 11 ) ) + kr ) / 2 ) ); \
	a_reg[12] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 12 ) ) + kr ) / 2 ) ); \
	a_reg[13] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 13 ) ) + kr ) / 2 ) ); \
	a_reg[14] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 14 ) ) + kr ) / 2 ) ); \
	a_reg[15] = _mm256_maskz_loadu_epi8( mask, ( b + ( ( ldb * ( jr + 15 ) ) + kr ) / 2 ) );

#define RIGHT_SHIFT_LAST_4BITS_ODD \
	a_reg[1] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[1] ), 0x4 ) ); \
	a_reg[3] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[3] ), 0x4 ) ); \
	a_reg[5] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[5] ), 0x4 ) ); \
	a_reg[7] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[7] ), 0x4 ) ); \
	a_reg[9] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[9] ), 0x4 ) ); \
	a_reg[11] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[11] ), 0x4 ) ); \
	a_reg[13] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[13] ), 0x4 ) ); \
	a_reg[15] = _mm512_cvtepi16_epi8( _mm512_srli_epi16  \
			( _mm512_cvtepu8_epi16( a_reg[15] ), 0x4 ) );

#define CLEAN_HIGH_4BITS_EVEN \
	__m512i clear_hi_bits = _mm512_maskz_set1_epi8( \
			_cvtu64_mask64( 0xFFFFFFFFFFFFFFFF ), 0x0F ); \
	a_reg[0] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[0] ), clear_hi_bits ) ); \
	a_reg[2] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[2] ), clear_hi_bits ) ); \
	a_reg[4] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[4] ), clear_hi_bits ) ); \
	a_reg[6] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[6] ), clear_hi_bits ) ); \
	a_reg[8] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[8] ), clear_hi_bits ) ); \
	a_reg[10] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[10] ), clear_hi_bits ) ); \
	a_reg[12] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[12] ), clear_hi_bits ) ); \
	a_reg[14] = _mm512_maskz_cvtepi16_epi8( 0x0F, _mm512_and_epi32( \
			_mm512_cvtepu8_epi16( a_reg[14] ), clear_hi_bits ) );


void packb_nr_mult_16_bf16s4f32of32_col_major
    (
      int8_t*         pack_b_buffer,
      const int8_t*   b,
      const dim_t     NR,
      const dim_t     ldb,
      const dim_t     KC,
      AOCL_MATRIX_TYPE mtag
    )
{
	// Used for storing the mm256i elements for use in dpbf16_ps instruction.
	__mmask8 msk0 = _cvtu32_mask8( 0x0F );
	__mmask8 msk1 = _cvtu32_mask8( 0xF0 );

	__m256i a_reg[16];
	__m256i b_reg[16];

	bool is_odd_stride = ( ( ldb % 2 ) == 0 ) ? FALSE : TRUE;

	dim_t kr = 0;

	if( is_odd_stride == FALSE )
	{

		for ( kr= 0; ( kr + 63 ) < KC; kr += 64 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0xFFFFFFFF )
				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2_ALLREG
				}
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + 	( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );

				/*The 16 value decrement is to correct the masked
				store starting position with respect to the msk1.*/
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 32  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 34  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 36  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 38  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 40  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 42  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 44  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 46  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[14] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 48  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[1]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 50  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[9]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 52  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[5]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 54  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[13] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 56  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[3]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 58  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[11] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 60  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[7]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer + ( ( jr * 2 )
				+ ( ( kr + 62  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[15] );
			}
		}

		for ( ; ( kr + 31 ) < KC; kr += 32 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0x0000FFFF )
				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2_ALLREG
				}
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				//store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );
			}
		}

		for ( ; ( kr + 15 ) < KC; kr += 16 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0x000000FF )
				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2_ALLREG
				}
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
			}
		}

		for ( ; ( kr + 7 ) < KC; kr += 8 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0x0F )
				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2_ALLREG
				}
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			}
		}

		for ( ; ( kr + 3 ) < KC; kr += 4 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0x03 )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8] );
			}
		}

		for ( ; ( kr + 1 ) < KC; kr += 2 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2( 0x01 )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0] );
			}
		}


	}
	else
	{
		for ( ; ( kr + 31 ) < KC; kr += 32 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_ODD( 0x0000FFFF, 0x0001FFFF )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				//store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );
			}
		}

		for ( ; ( kr + 15 ) < KC; kr += 16 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_ODD( 0x00000000000000FF, 0x00000000000001FF )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR) ) /2 ) ), msk0, a_reg[12] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 8  ) * NR) ) /2 ) ), msk0, a_reg[2]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 10 ) * NR) ) /2 ) ), msk0, a_reg[10] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 12 ) * NR) ) /2 ) ), msk0, a_reg[6]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 14 ) * NR) ) /2 ) ), msk0, a_reg[14] );
			}
		}

		for ( ; ( kr + 7 ) < KC; kr += 8 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_ODD( 0x000000000000000F, 0x000000000000001F )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			}
		}

		for ( ; ( kr + 3 ) < KC; kr += 4 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_ODD( 0x0000000000000003, 0x000000000000007 )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0 ) * NR ) ) /2 ) ), msk0, a_reg[0] );
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 2 ) * NR ) ) /2 ) ), msk0, a_reg[8] );
			}
		}

		for ( ; ( kr + 1 ) < KC; kr += 2 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_ODD( 0x0000000000000001, 0x000000000000003 )
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR) ) /2 ) ), msk0, a_reg[0] );
			}
		}
		for ( ; kr < KC; kr += 1 )
		{
			for( dim_t jr = 0; jr < NR; jr += 16 )
			{
				/* Rearrange for dpbf16_ps, read 16 cols
				from B with 64 elements in each row.*/
				MASK_LOAD_16_COLS_AVX2_LAST_4BITS( 0x01 )
				RIGHT_SHIFT_LAST_4BITS_ODD
				CLEAN_HIGH_4BITS_EVEN
				UNPACKHILO8_AVX2
				UNPACKHILO16_AVX2
				UNPACKHILO32_AVX2
				UNPACKHILO64_AVX2

				// store to pack_b buffer
				_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( jr * 2 ) + ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0] );

			}
		}


	}


}

void packb_nrlt16_bf16s4f32of32_col_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t     ldb,
      const dim_t     KC,
      const dim_t     n0_partial_rem,
      AOCL_MATRIX_TYPE mtag
    )
{
	dim_t NR = 16;

	// Used for storing the mm256i elements for use in dpbf16_ps instruction.
	__mmask8 msk0 = _cvtu32_mask8( 0x0F );
	__mmask8 msk1 = _cvtu32_mask8( 0xF0 );

	__m256i a_reg[16];
	__m256i b_reg[16];
    bool is_odd_stride = ( ( ldb % 2 ) == 0 ) ? FALSE : TRUE;

	dim_t kr = 0, jr = 0;
	__m512i shift_idx_64;
	__m512i a0;
	__m512i sign_comp = _mm512_set1_epi8( 0x08 );
	__m256i even;
	__m256i odd;
	bool signed_upscale = TRUE;

	if( is_odd_stride == FALSE )
	{
		for ( kr = 0; ( kr + 63 ) < KC; kr += 64 )
		{
			for( jr = 0; jr < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_loadu_si256( ( __m256i const * ) ( b +
						( ( ldb * jr ) + kr ) / 2 ) );

				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );

			/*The 16 value decrement is to correct the masked
			store starting postion with respect to the msk1.*/
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 32  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 34  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 36  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 38  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 40  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 42  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 44  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 46  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[14] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 48  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[1]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 50  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[9]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 52  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[5]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 54  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[13] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 56  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[3]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 58  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[11] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 60  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[7]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
			( ( ( kr + 62  ) * NR ) ) /2 - 16 ) ), msk1, a_reg[15] );
		}

		for ( ; ( kr + 31 ) < KC; kr += 32 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0000FFFF,
					( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );

				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );
		}

		for ( ; ( kr + 15 ) < KC; kr += 16 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0xFF,
					( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );

				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
		}

		for ( ; ( kr + 7 ) < KC; kr += 8 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0F,
					(b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );

				if( mtag == AWQ_B_MATRIX )
				{
					ORDER_SHIFT_AVX2( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
		}

		for ( ; ( kr+3 ) < KC; kr += 4 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x03,
					( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8] );
		}

		for ( ; ( kr + 1 ) < KC; kr += 2 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x01,
					( b + ( ( ldb * ( jr + 0 ) ) + kr) / 2 ) );
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR) ) /2 ) ), msk0, a_reg[0] );
		}

		for ( ; kr < KC; kr += 1 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x01,
					( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) )/2 ) ), msk0, b_reg[0] );
		}
	}
    else
	{
		__m256i shift_index = _mm256_setr_epi8(
					33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
					44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
					55, 56, 57, 58, 59, 60, 61, 62, 63, 32 );
		__m256i odd_256;
		__m256i even_256;
		for ( ; ( kr + 31 ) < KC; kr += 32 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				if( jr % 2 == 0 )
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0000FFFF,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
				}
				else
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0001FFFF,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
					REMOVE_EXTRA_BITS( a_reg[jr] )
				}

			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 16 ) * NR ) ) /2 ) ), msk0, a_reg[1]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 18 ) * NR ) ) /2 ) ), msk0, a_reg[9]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 20 ) * NR ) ) /2 ) ), msk0, a_reg[5]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 22 ) * NR ) ) /2 ) ), msk0, a_reg[13] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 24 ) * NR ) ) /2 ) ), msk0, a_reg[3]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 26 ) * NR ) ) /2 ) ), msk0, a_reg[11] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 28 ) * NR ) ) /2 ) ), msk0, a_reg[7]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 30 ) * NR ) ) /2 ) ), msk0, a_reg[15] );
		}

		for ( ; ( kr + 15 ) < KC; kr += 16 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				if( jr % 2 == 0 )
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x00000000000000FF,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
				}
				else
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x00000000000001FF,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
					REMOVE_EXTRA_BITS( a_reg[jr] )
				}

			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 8  ) * NR ) ) /2 ) ), msk0, a_reg[2]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 10 ) * NR ) ) /2 ) ), msk0, a_reg[10] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 12 ) * NR ) ) /2 ) ), msk0, a_reg[6]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 14 ) * NR ) ) /2 ) ), msk0, a_reg[14] );
		}

		for ( ; ( kr + 7 ) < KC; kr += 8 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				if( jr % 2 == 0 )
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x000000000000000F,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
				}
				else
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x000000000000001F,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
					REMOVE_EXTRA_BITS( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 4  ) * NR ) ) /2 ) ), msk0, a_reg[4]  );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 6  ) * NR ) ) /2 ) ), msk0, a_reg[12] );
		}

		for ( ; ( kr+3 ) < KC; kr += 4 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				if( jr % 2 == 0 )
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0000000000000003,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
				}
				else
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x000000000000007,
						( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
					REMOVE_EXTRA_BITS( a_reg[jr] )
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0] );
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 2  ) * NR ) ) /2 ) ), msk0, a_reg[8] );
		}

		for ( ; ( kr + 1 ) < KC; kr += 2 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				if( jr % 2 == 0 )
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x0000000000000001,
						(b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
				}
				else
				{
					a_reg[jr] = _mm256_maskz_loadu_epi8( 0x000000000000003,
						(b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
					REMOVE_EXTRA_BITS(a_reg[jr])
				}
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}
			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, a_reg[0]  );
		}

		for ( ; kr < KC; kr += 1 )
		{
			for( jr = 0; jr  < n0_partial_rem; jr += 1 )
			{
				/*Rearrange for dpbf16_ps, read n0_partial_rem
				cols from B with 64 elements in each row*/
				a_reg[jr] = _mm256_maskz_loadu_epi8( 0x01,
					( b + ( ( ldb * ( jr + 0 ) ) + kr ) / 2 ) );
			}
			for( ; jr < NR; jr++ )
			{
				a_reg[jr] = _mm256_setzero_si256();
			}

			RIGHT_SHIFT_LAST_4BITS_ODD
			CLEAN_HIGH_4BITS_EVEN
			UNPACKHILO8_AVX2
			UNPACKHILO16_AVX2
			UNPACKHILO32_AVX2
			UNPACKHILO64_AVX2

			// store to pack_b buffer
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, b_reg[0] );
			//store remaining elements after 8 elements are stored
			_mm256_mask_storeu_epi32( ( ( pack_b_buffer +
				8 + ( ( ( kr + 0  ) * NR ) ) /2 ) ), msk0, b_reg[1] );
		}
	}
}



void packb_nr64_bf16s4f32of32_col_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   ldb,
      const dim_t   NC,
      const dim_t   KC,
      dim_t*        rs_b,
      dim_t*        cs_b,
      lpgemm_pre_op* pre_op,
      AOCL_MATRIX_TYPE mtag
    )
{
    dim_t NR = 64;
	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;


	dim_t n_partial_pieces = NC % NR;
	dim_t k_partial_pieces = KC % 2;
	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
	 	KC_updated += ( 2 - k_partial_pieces );
	}

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		packb_nr_mult_16_bf16s4f32of32_col_major
		(( pack_b_buffer + ((jc* KC_updated)/2)) ,
		(b + (jc*ldb)/2), 64, ldb, KC, mtag);
	}

	if(n_partial_pieces > 0)
	{

		dim_t n0_partial_rem = n_partial_pieces % 16;
		dim_t n0_partial_pack = 0;

		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(64) can be expressed
		// as n0 = 48 + n` / n0 = 32 + n` / n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n_partial_pieces / 48;
		dim_t n0_32 = n_partial_pieces / 32;
		dim_t n0_16 = n_partial_pieces / 16;

		if ( n0_48 == 1 )
		{
			packb_nr_mult_16_bf16s4f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated )/2 ),
				 ( b + (n_full_pieces_loop_limit * ldb )/2), 48, ldb, KC, mtag
				);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr_mult_16_bf16s4f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated )/2 ),
				 ( b + (n_full_pieces_loop_limit * ldb)/2 ), 32, ldb, KC, mtag
				);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr_mult_16_bf16s4f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated )/2 ),
				 ( b + (n_full_pieces_loop_limit * ldb)/2 ), 16, ldb, KC, mtag
				);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16s4f32of32_col_major
				(
				 ( pack_b_buffer + (( n_full_pieces_loop_limit * KC_updated )  +
				   ( n0_partial_pack * KC_updated ))/2 ),
				 ( b + (( n_full_pieces_loop_limit + n0_partial_pack ) * ldb)/2 ), ldb, KC,
				 n0_partial_rem, mtag
				);
		}
	}

	*rs_b = NR * 2;
	*cs_b = NR / 2;
}


#endif
