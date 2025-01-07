/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

void packb_nr16_f32f32f32of32_row_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nr16_f32f32f32of32_col_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nr16_f32f32f32of32
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t  rs_b,
       const dim_t  cs_b,
       const dim_t  NC,
       const dim_t  KC,
       dim_t*       rs_p,
       dim_t*       cs_p
     )
{
	if( cs_b == 1 )
	{
		packb_nr16_f32f32f32of32_row_major( pack_b_buffer, b,
		                                    rs_b, NC, KC, rs_p, cs_p );
	}
	else
	{
		packb_nr16_f32f32f32of32_col_major( pack_b_buffer, b,
		                                    cs_b, NC, KC, rs_p, cs_p );
	}
}

void packb_nr16_f32f32f32of32_row_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t  ldb,
       const dim_t  NC,
       const dim_t  KC,
       dim_t*       rs_b,
       dim_t*       cs_b
     )
{
    dim_t NR = 16;

	__m256 a0;
	__m256 b0;

	dim_t n_full_pieces_loop_limit = ( NC / NR ) * NR;
	dim_t n_partial_pieces = NC % NR;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < KC; kr += 1 )
		{
			a0 = _mm256_loadu_ps( b + ( jc + 0 ) + ( ldb * kr  ) );
			b0 = _mm256_loadu_ps( b + ( jc + 8 ) + ( ldb * kr  ) );

			//store to pack_b buffer
			_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 0 ), a0 );
			_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 8 ), b0 );
		}
	}
	if( n_partial_pieces > 0 )
	{
		for ( dim_t kr = 0; kr < KC; kr += 1 )
		{
			// No point in vectorizing fringe case since n_fringe is expected
			// to be laid out contiguously in pack buffer as nr_fringe*KC
			// instead of 8*KC + 4*KC + .., etc.
			memcpy( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					 ( ( kr * NR ) + 0 ),
					b + ( n_full_pieces_loop_limit + 0 ) + ( ldb * kr  ),
					n_partial_pieces * ( sizeof( float ) ) );

			// Zero out padding data.
			memset( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					 ( ( kr * NR ) + n_partial_pieces ),
					0, ( NR - n_partial_pieces ) * sizeof( float ) );
		}
	}

	*rs_b = NR;
	*cs_b = 1;
}

#define LOAD_PS_8x8() \
	a_reg[0] = _mm256_loadu_ps( b + ( ldb * ( jr + 0 ) ) + ( kr ) ); \
	a_reg[1] = _mm256_loadu_ps( b + ( ldb * ( jr + 1 ) ) + ( kr ) ); \
	a_reg[2] = _mm256_loadu_ps( b + ( ldb * ( jr + 2 ) ) + ( kr ) ); \
	a_reg[3] = _mm256_loadu_ps( b + ( ldb * ( jr + 3 ) ) + ( kr ) ); \
	a_reg[4] = _mm256_loadu_ps( b + ( ldb * ( jr + 4 ) ) + ( kr ) ); \
	a_reg[5] = _mm256_loadu_ps( b + ( ldb * ( jr + 5 ) ) + ( kr ) ); \
	a_reg[6] = _mm256_loadu_ps( b + ( ldb * ( jr + 6 ) ) + ( kr ) ); \
	a_reg[7] = _mm256_loadu_ps( b + ( ldb * ( jr + 7 ) ) + ( kr ) ); \

#define K_FRINGE_MEMCPY_LOAD_PS_8x8() \
	memcpy( buf0, b + ( ldb * ( jr + 0 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[0] = _mm256_loadu_ps( buf0 ); \
	memcpy( buf1, b + ( ldb * ( jr + 1 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[1] = _mm256_loadu_ps( buf1 ); \
	memcpy( buf2, b + ( ldb * ( jr + 2 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[2] = _mm256_loadu_ps( buf2 ); \
	memcpy( buf3, b + ( ldb * ( jr + 3 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[3] = _mm256_loadu_ps( buf3 ); \
	memcpy( buf4, b + ( ldb * ( jr + 4 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[4] = _mm256_loadu_ps( buf4 ); \
	memcpy( buf5, b + ( ldb * ( jr + 5 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[5] = _mm256_loadu_ps( buf5 ); \
	memcpy( buf6, b + ( ldb * ( jr + 6 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[6] = _mm256_loadu_ps( buf6 ); \
	memcpy( buf7, b + ( ldb * ( jr + 7 ) ) + ( kr ), \
				k_partial_pieces * sizeof( float ) ); \
	a_reg[7] = _mm256_loadu_ps( buf7 ); \

#define N_FRINGE_LOAD_PS_8x8() \
	for ( int i = 0; i < jr_elems; ++i ) \
	{ \
		a_reg[i] = _mm256_loadu_ps( b + ( ldb * ( jr + i ) ) + ( kr ) ); \
	} \
	for ( int i = jr_elems; i < n_sub_blk_wdth; ++i ) \
	{ \
		a_reg[i] = _mm256_setzero_ps(); \
	} \

#define KN_FRINGE_MEMCPY_LOAD_PS_8x8() \
	for ( int i = 0; i < jr_elems; ++i ) \
	{ \
		memcpy( buf0, b + ( ldb * ( jr + i ) ) + ( kr ), \
					k_partial_pieces * sizeof( float ) ); \
		a_reg[i] = _mm256_loadu_ps( buf0 ); \
	} \
	for ( int i = jr_elems; i < n_sub_blk_wdth; ++i ) \
	{ \
		a_reg[i] = _mm256_setzero_ps(); \
	} \

#define UNPACK_PS_8x8() \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	b_reg[0] = _mm256_unpacklo_ps( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm256_unpackhi_ps( a_reg[0], a_reg[1] ); \
	b_reg[2] = _mm256_unpacklo_ps( a_reg[2], a_reg[3] ); \
	b_reg[3] = _mm256_unpackhi_ps( a_reg[2], a_reg[3] ); \
	b_reg[4] = _mm256_unpacklo_ps( a_reg[4], a_reg[5] ); \
	b_reg[5] = _mm256_unpackhi_ps( a_reg[4], a_reg[5] ); \
	b_reg[6] = _mm256_unpacklo_ps( a_reg[6], a_reg[7] ); \
	b_reg[7] = _mm256_unpackhi_ps( a_reg[6], a_reg[7] ); \

#define UNPACK_PD_8x8() \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	a_reg[0] = ( __m256 )_mm256_unpacklo_pd( ( __m256d )b_reg[0], \
				( __m256d )b_reg[2] ); \
	a_reg[1] = ( __m256 )_mm256_unpackhi_pd( ( __m256d )b_reg[0], \
				( __m256d )b_reg[2] ); \
	a_reg[2] = ( __m256 )_mm256_unpacklo_pd( ( __m256d )b_reg[4], \
				( __m256d )b_reg[6] ); \
	a_reg[3] = ( __m256 )_mm256_unpackhi_pd( ( __m256d )b_reg[4], \
				( __m256d )b_reg[6] ); \
	a_reg[4] = ( __m256 )_mm256_unpacklo_pd( ( __m256d )b_reg[1], \
				( __m256d )b_reg[3] ); \
	a_reg[5] = ( __m256 )_mm256_unpackhi_pd( ( __m256d )b_reg[1], \
				( __m256d )b_reg[3] ); \
	a_reg[6] = ( __m256 )_mm256_unpacklo_pd( ( __m256d )b_reg[5], \
				( __m256d )b_reg[7] ); \
	a_reg[7] = ( __m256 )_mm256_unpackhi_pd( ( __m256d )b_reg[5], \
				( __m256d )b_reg[7] ); \

#define PERMUTE_R1_8x8() \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	b_reg[0] = _mm256_permute2f128_ps( a_reg[0], a_reg[2], 0x20 ); /* Row 0 */ \
	b_reg[1] = _mm256_permute2f128_ps( a_reg[0], a_reg[2], 0x31 ); /* Row 4 */ \
	b_reg[2] = _mm256_permute2f128_ps( a_reg[4], a_reg[6], 0x20 ); /* Row 2 */ \
	b_reg[3] = _mm256_permute2f128_ps( a_reg[4], a_reg[6], 0x31 ); /* Row 6 */ \
	b_reg[4] = _mm256_permute2f128_ps( a_reg[1], a_reg[3], 0x20 ); /* Row 1 */ \
	b_reg[5] = _mm256_permute2f128_ps( a_reg[1], a_reg[3], 0x31 ); /* Row 5 */ \
	b_reg[6] = _mm256_permute2f128_ps( a_reg[5], a_reg[7], 0x20 ); /* Row 3 */ \
	b_reg[7] = _mm256_permute2f128_ps( a_reg[5], a_reg[7], 0x31 ); /* Row 7 */ \

#define STORE_PS_8x8() \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 0 ) * NR ) + jr_offset, \
					b_reg[0] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 1 ) * NR ) + jr_offset, \
					b_reg[4] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 2 ) * NR ) + jr_offset, \
					b_reg[2] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 3 ) * NR ) + jr_offset, \
					b_reg[6] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 4 ) * NR ) + jr_offset, \
					b_reg[1] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 5 ) * NR ) + jr_offset, \
					b_reg[5] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 6 ) * NR ) + jr_offset, \
					b_reg[3] ); \
	_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 7 ) * NR ) + jr_offset, \
					b_reg[7] ); \

#define K_FRINGE_SAFE_STORE_PS_8x8() \
	a_reg[0] = b_reg[0]; \
	a_reg[1] = b_reg[4]; \
	a_reg[2] = b_reg[2]; \
	a_reg[3] = b_reg[6]; \
	a_reg[4] = b_reg[1]; \
	a_reg[5] = b_reg[5]; \
	a_reg[6] = b_reg[3]; \
	a_reg[7] = b_reg[7]; \
	for (int i = 0; i < k_partial_pieces; ++i ) \
	{ \
		_mm256_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + i ) * NR ) + jr_offset, \
						a_reg[i] ); \
	} \

void packb_nr16_f32f32f32of32_col_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     )
{
    const dim_t NR = 16;
	const dim_t n_sub_blk_wdth = 8;
	const dim_t k_reg_size = 8;

	float buf0[8] = { 0 };
	float buf1[8] = { 0 };
	float buf2[8] = { 0 };
	float buf3[8] = { 0 };
	float buf4[8] = { 0 };
	float buf5[8] = { 0 };
	float buf6[8] = { 0 };
	float buf7[8] = { 0 };

	__m256 a_reg[8];
	__m256 b_reg[8];

	dim_t n_full_pieces_loop_limit = ( NC / NR ) * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_full_pieces_loop_limit = ( KC / k_reg_size ) * k_reg_size;
	dim_t k_partial_pieces = KC % k_reg_size;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t jr = jc; jr < jc + NR; jr += n_sub_blk_wdth )
		{
			dim_t jr_offset = jr % NR;
			for ( dim_t kr = 0; kr < k_full_pieces_loop_limit; kr += k_reg_size )
			{
				LOAD_PS_8x8();
				UNPACK_PS_8x8();
				UNPACK_PD_8x8();
				PERMUTE_R1_8x8();
				STORE_PS_8x8();
			}
			if ( k_partial_pieces > 0 )
			{
				dim_t kr = k_full_pieces_loop_limit;

				K_FRINGE_MEMCPY_LOAD_PS_8x8();
				UNPACK_PS_8x8();
				UNPACK_PD_8x8();
				PERMUTE_R1_8x8();
				K_FRINGE_SAFE_STORE_PS_8x8();
			}
		}
	}

	if( n_partial_pieces > 0 )
	{
		dim_t jc = n_full_pieces_loop_limit;
		for ( dim_t jr = n_full_pieces_loop_limit; jr < NC; jr += n_sub_blk_wdth )
		{
			dim_t jr_offset = jr % NR;
			dim_t jr_elems = ( ( NC - jr ) >= n_sub_blk_wdth ) ? n_sub_blk_wdth : ( NC - jr );

			for ( dim_t kr = 0; kr < k_full_pieces_loop_limit; kr += k_reg_size )
			{
				N_FRINGE_LOAD_PS_8x8();
				UNPACK_PS_8x8();
				UNPACK_PD_8x8();
				PERMUTE_R1_8x8();
				STORE_PS_8x8();
			}
			if ( k_partial_pieces > 0 )
			{
				dim_t kr = k_full_pieces_loop_limit;

				KN_FRINGE_MEMCPY_LOAD_PS_8x8();
				UNPACK_PS_8x8();
				UNPACK_PD_8x8();
				PERMUTE_R1_8x8();
				K_FRINGE_SAFE_STORE_PS_8x8();
			}
		}
	}

	*rs_b = NR;
	*cs_b = 1;
}

#endif
