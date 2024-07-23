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
      dim_t*        cs_p
    );

void packb_nr48_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC
    );

void packb_nr32_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC
    );

void packb_nr16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC
    );

void packb_nrlt16_bf16s4f32of32_row_major
    (
      int8_t*       pack_b_buffer,
      const int8_t* b,
      const dim_t   rs_b,
      const dim_t   KC,
      const dim_t n0_partial_rem
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
       dim_t*        cs_p
     )
{
	if (cs_b == 1)
	{
		packb_nr64_bf16s4f32of32_row_major(pack_b_buffer,
						b, rs_b, NC, KC, rs_p, cs_p);
	}
	else
	{
		bli_print_msg("Only row major supported for int4 packing.",
				__FILE__, __LINE__);
		return;
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
      dim_t*        cs_p
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
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM( shift_idx_64 );

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
			  rs_b, KC
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
			  rs_b, KC
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
			  rs_b, KC
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
			  rs_b, KC, n0_partial_rem
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
      const dim_t   KC
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
	MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);

	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );
	__mmask16 hmask_32 = _cvtu32_mask16( 0x0000FFFF ); //16 bytes or 32 int4.
	__mmask16 hmask_odd_32 = _cvtu32_mask16( 0x00008000 ); // Last 1 int4.

	CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(conv_shift_arr_32);
	__m256i conv_shift_32 = _mm256_maskz_loadu_epi64( _cvtu32_mask8( 0X000000FF ),
					conv_shift_arr_32 );

	// Next 16 int4 elements selectors.
	__m128i shift_idx_16;
	MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);

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
      const dim_t   KC
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
	MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);

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
      const dim_t   KC
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
	MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);

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
      const dim_t n0_partial_rem
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
	MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);

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

#endif
