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
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

void packb_nr64_f32f32f32of32_row_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nr64_f32f32f32of32_col_major
     (
       float*       pack_b_buffer,
       const float* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nr64_f32f32f32of32
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
		packb_nr64_f32f32f32of32_row_major( pack_b_buffer, b,
		                                    rs_b, NC, KC, rs_p, cs_p );
	}
	else
	{
		packb_nr64_f32f32f32of32_col_major( pack_b_buffer, b,
		                                    cs_b, NC, KC, rs_p, cs_p );
	}
}

void packb_nr64_f32f32f32of32_row_major
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
    dim_t NR = 64;

	__m512 a0;
	__m512 b0;
	__m512 c0;
	__m512 d0;

	dim_t n_full_pieces_loop_limit = ( NC / NR ) * NR;
	dim_t n_partial_pieces = NC % NR;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < KC; kr += 1 )
		{
			a0 = _mm512_loadu_ps( b + ( ldb * kr  ) + ( jc + 0 ) );
			b0 = _mm512_loadu_ps( b + ( ldb * kr  ) + ( jc + 16 ) );
			c0 = _mm512_loadu_ps( b + ( ldb * kr  ) + ( jc + 32 ) );
			d0 = _mm512_loadu_ps( b + ( ldb * kr  ) + ( jc + 48 ) );

			//store to pack_b buffer
			_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 0 ), a0 );
			_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 16 ), b0 );
			_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 32 ), c0 );
			_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) +
					( ( kr * NR ) + 48 ), d0 );
		}
	}
	if( n_partial_pieces > 0 )
	{
        dim_t n0_partial_rem = n_partial_pieces % 16;

        dim_t n0_48 = n_partial_pieces / 48;
        dim_t n0_32 = n_partial_pieces / 32;
        dim_t n0_16 = n_partial_pieces / 16;

		__mmask16 lmask_0 = _cvtu32_mask16( 0x0 );
		__mmask16 lmask_1 = _cvtu32_mask16( 0x0 );
		__mmask16 lmask_2 = _cvtu32_mask16( 0x0 );
		__mmask16 lmask_3 = _cvtu32_mask16( 0x0 );

		if ( n0_48 > 0 )
		{
			lmask_0 = _cvtu32_mask16( 0xFFFF );
			lmask_1 = _cvtu32_mask16( 0xFFFF );
			lmask_2 = _cvtu32_mask16( 0xFFFF );
			lmask_3 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_partial_rem ) );
		}
		else if ( n0_32 > 0 )
		{
			lmask_0 = _cvtu32_mask16( 0xFFFF );
			lmask_1 = _cvtu32_mask16( 0xFFFF );
			lmask_2 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_partial_rem ) );
		}
		else if ( n0_16 > 0 )
		{
			lmask_0 = _cvtu32_mask16( 0xFFFF );
			lmask_1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_partial_rem ) );
		}
		else
		{
			lmask_0 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_partial_rem ) );
		}

		for ( dim_t kr = 0; kr < KC; kr += 1 )
		{
			a0 = _mm512_maskz_loadu_ps( lmask_0, b + ( ldb * kr  ) +
						( n_full_pieces_loop_limit + 0 ) );
			b0 = _mm512_maskz_loadu_ps( lmask_1, b + ( ldb * kr  ) +
						( n_full_pieces_loop_limit + 16 ) );
			c0 = _mm512_maskz_loadu_ps( lmask_2, b + ( ldb * kr  ) +
						( n_full_pieces_loop_limit + 32 ) );
			d0 = _mm512_maskz_loadu_ps( lmask_3, b + ( ldb * kr  ) +
						( n_full_pieces_loop_limit + 48 ) );

			//store to pack_b buffer
			_mm512_storeu_ps( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					( ( kr * NR ) + 0 ), a0 );
			_mm512_storeu_ps( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					( ( kr * NR ) + 16 ), b0 );
			_mm512_storeu_ps( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					( ( kr * NR ) + 32 ), c0 );
			_mm512_storeu_ps( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) +
					( ( kr * NR ) + 48 ), d0 );
		}
	}

	*rs_b = NR;
	*cs_b = 1;
}

#define MASK_LOAD_PS_16x16(msk) \
	a_reg[0] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 0 ) ) + ( kr ) ); \
	a_reg[1] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 1 ) ) + ( kr ) ); \
	a_reg[2] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 2 ) ) + ( kr ) ); \
	a_reg[3] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 3 ) ) + ( kr ) ); \
	a_reg[4] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 4 ) ) + ( kr ) ); \
	a_reg[5] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 5 ) ) + ( kr ) ); \
	a_reg[6] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 6 ) ) + ( kr ) ); \
	a_reg[7] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 7 ) ) + ( kr ) ); \
	a_reg[8] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 8 ) ) + ( kr ) ); \
	a_reg[9] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 9 ) ) + ( kr ) ); \
	a_reg[10] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 10 ) ) + ( kr ) ); \
	a_reg[11] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 11 ) ) + ( kr ) ); \
	a_reg[12] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 12 ) ) + ( kr ) ); \
	a_reg[13] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 13 ) ) + ( kr ) ); \
	a_reg[14] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 14 ) ) + ( kr ) ); \
	a_reg[15] = _mm512_maskz_loadu_ps( ( msk ), b + ( ldb * ( jr + 15 ) ) + ( kr ) ); \

#define MASK_ARR_LOAD_PS_16x16(msk_n_arr) \
	a_reg[0] = _mm512_maskz_loadu_ps( msk_n_arr[0], b + ( ldb * ( jr + 0 ) ) + ( kr ) ); \
	a_reg[1] = _mm512_maskz_loadu_ps( msk_n_arr[1], b + ( ldb * ( jr + 1 ) ) + ( kr ) ); \
	a_reg[2] = _mm512_maskz_loadu_ps( msk_n_arr[2], b + ( ldb * ( jr + 2 ) ) + ( kr ) ); \
	a_reg[3] = _mm512_maskz_loadu_ps( msk_n_arr[3], b + ( ldb * ( jr + 3 ) ) + ( kr ) ); \
	a_reg[4] = _mm512_maskz_loadu_ps( msk_n_arr[4], b + ( ldb * ( jr + 4 ) ) + ( kr ) ); \
	a_reg[5] = _mm512_maskz_loadu_ps( msk_n_arr[5], b + ( ldb * ( jr + 5 ) ) + ( kr ) ); \
	a_reg[6] = _mm512_maskz_loadu_ps( msk_n_arr[6], b + ( ldb * ( jr + 6 ) ) + ( kr ) ); \
	a_reg[7] = _mm512_maskz_loadu_ps( msk_n_arr[7], b + ( ldb * ( jr + 7 ) ) + ( kr ) ); \
	a_reg[8] = _mm512_maskz_loadu_ps( msk_n_arr[8], b + ( ldb * ( jr + 8 ) ) + ( kr ) ); \
	a_reg[9] = _mm512_maskz_loadu_ps( msk_n_arr[9], b + ( ldb * ( jr + 9 ) ) + ( kr ) ); \
	a_reg[10] = _mm512_maskz_loadu_ps( msk_n_arr[10], b + ( ldb * ( jr + 10 ) ) + ( kr ) ); \
	a_reg[11] = _mm512_maskz_loadu_ps( msk_n_arr[11], b + ( ldb * ( jr + 11 ) ) + ( kr ) ); \
	a_reg[12] = _mm512_maskz_loadu_ps( msk_n_arr[12], b + ( ldb * ( jr + 12 ) ) + ( kr ) ); \
	a_reg[13] = _mm512_maskz_loadu_ps( msk_n_arr[13], b + ( ldb * ( jr + 13 ) ) + ( kr ) ); \
	a_reg[14] = _mm512_maskz_loadu_ps( msk_n_arr[14], b + ( ldb * ( jr + 14 ) ) + ( kr ) ); \
	a_reg[15] = _mm512_maskz_loadu_ps( msk_n_arr[15], b + ( ldb * ( jr + 15 ) ) + ( kr ) ); \

#define UNPACK_PS_16x16() \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	b_reg[0] = _mm512_unpacklo_ps( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm512_unpackhi_ps( a_reg[0], a_reg[1] ); \
	b_reg[2] = _mm512_unpacklo_ps( a_reg[2], a_reg[3] ); \
	b_reg[3] = _mm512_unpackhi_ps( a_reg[2], a_reg[3] ); \
	b_reg[4] = _mm512_unpacklo_ps( a_reg[4], a_reg[5] ); \
	b_reg[5] = _mm512_unpackhi_ps( a_reg[4], a_reg[5] ); \
	b_reg[6] = _mm512_unpacklo_ps( a_reg[6], a_reg[7] ); \
	b_reg[7] = _mm512_unpackhi_ps( a_reg[6], a_reg[7] ); \
	b_reg[8] = _mm512_unpacklo_ps( a_reg[8], a_reg[9] ); \
	b_reg[9] = _mm512_unpackhi_ps( a_reg[8], a_reg[9] ); \
	b_reg[10] = _mm512_unpacklo_ps( a_reg[10], a_reg[11] ); \
	b_reg[11] = _mm512_unpackhi_ps( a_reg[10], a_reg[11] ); \
	b_reg[12] = _mm512_unpacklo_ps( a_reg[12], a_reg[13] ); \
	b_reg[13] = _mm512_unpackhi_ps( a_reg[12], a_reg[13] ); \
	b_reg[14] = _mm512_unpacklo_ps( a_reg[14], a_reg[15] ); \
	b_reg[15] = _mm512_unpackhi_ps( a_reg[14], a_reg[15] ); \

#define UNPACK_PD_16x16() \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	a_reg[0] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[0], ( __m512d )b_reg[2] ); \
	a_reg[1] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[0], ( __m512d )b_reg[2] ); \
	a_reg[2] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[4], ( __m512d )b_reg[6] ); \
	a_reg[3] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[4], ( __m512d )b_reg[6] ); \
	a_reg[4] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[8], ( __m512d )b_reg[10] ); \
	a_reg[5] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[8], ( __m512d )b_reg[10] ); \
	a_reg[6] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[12], ( __m512d )b_reg[14] ); \
	a_reg[7] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[12], ( __m512d )b_reg[14] ); \
	a_reg[8] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[1], ( __m512d )b_reg[3] ); \
	a_reg[9] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[1], ( __m512d )b_reg[3] ); \
	a_reg[10] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[5], ( __m512d )b_reg[7] ); \
	a_reg[11] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[5], ( __m512d )b_reg[7] ); \
	a_reg[12] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[9], ( __m512d )b_reg[11] ); \
	a_reg[13] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[9], ( __m512d )b_reg[11] ); \
	a_reg[14] = ( __m512 )_mm512_unpacklo_pd( ( __m512d )b_reg[13], ( __m512d )b_reg[15] ); \
	a_reg[15] = ( __m512 )_mm512_unpackhi_pd( ( __m512d )b_reg[13], ( __m512d )b_reg[15] ); \

#define PERMUTE_R1_16x16(selector1, selector2) \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	b_reg[0] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[0], selector1, ( __m512d )a_reg[2] ); \
	b_reg[1] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[0], selector2, ( __m512d )a_reg[2] ); \
	b_reg[2] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[4], selector1, ( __m512d )a_reg[6] ); \
	b_reg[3] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[4], selector2, ( __m512d )a_reg[6] ); \
	b_reg[4] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[8], selector1, ( __m512d )a_reg[10] ); \
	b_reg[5] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[8], selector2, ( __m512d )a_reg[10] ); \
	b_reg[6] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[12], selector1, ( __m512d )a_reg[14] ); \
	b_reg[7] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[12], selector2, ( __m512d )a_reg[14] ); \
	b_reg[8] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[1], selector1, ( __m512d )a_reg[3] ); \
	b_reg[9] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[1], selector2, ( __m512d )a_reg[3] ); \
	b_reg[10] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[5], selector1, ( __m512d )a_reg[7] ); \
	b_reg[11] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[5], selector2, ( __m512d )a_reg[7] ); \
	b_reg[12] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[9], selector1, ( __m512d )a_reg[11] ); \
	b_reg[13] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[9], selector2, ( __m512d )a_reg[11] ); \
	b_reg[14] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[13], selector1, ( __m512d )a_reg[15] ); \
	b_reg[15] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )a_reg[13], selector2, ( __m512d )a_reg[15] ); \

#define PERMUTE_R2_16x16(selector1_1, selector2_1) \
	/* Even indices contains lo parts, odd indices contains hi parts. */ \
	a_reg[0] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[0], selector1_1, \
				( __m512d )b_reg[2] ); /* Row 0 */ \
	a_reg[1] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[0], selector2_1, \
				( __m512d )b_reg[2] ); /* Row 4 */ \
	a_reg[2] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[4], selector1_1, \
				( __m512d )b_reg[6] ); /* Row 2 */ \
	a_reg[3] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[4], selector2_1, \
				( __m512d )b_reg[6] ); /* Row 6 */ \
	a_reg[4] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[8], selector1_1, \
				( __m512d )b_reg[10] ); /* Row 1 */ \
	a_reg[5] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[8], selector2_1, \
				( __m512d )b_reg[10] ); /* Row 5 */ \
	a_reg[6] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[12], selector1_1, \
				( __m512d )b_reg[14] ); /* Row 3 */ \
	a_reg[7] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[12], selector2_1, \
				( __m512d )b_reg[14] ); /* Row 7 */ \
	a_reg[8] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[1], selector1_1, \
				( __m512d )b_reg[3] ); /* Row 8 */ \
	a_reg[9] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[1], selector2_1, \
				( __m512d )b_reg[3] ); /* Row 12 */ \
	a_reg[10] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[5], selector1_1, \
				( __m512d )b_reg[7] ); /* Row 10 */ \
	a_reg[11] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[5], selector2_1, \
				( __m512d )b_reg[7] ); /* Row 14 */ \
	a_reg[12] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[9], selector1_1, \
				( __m512d )b_reg[11] ); /* Row 9 */ \
	a_reg[13] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[9], selector2_1, \
				( __m512d )b_reg[11] ); /* Row 13 */ \
	a_reg[14] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[13], selector1_1, \
				( __m512d )b_reg[15] ); /* Row 11 */ \
	a_reg[15] = ( __m512 )_mm512_permutex2var_pd( ( __m512d )b_reg[13], selector2_1, \
				( __m512d )b_reg[15] ); /* Row 15 */ \

#define STORE_PS_16x16() \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 0 ) * NR ) + jr_offset, a_reg[0] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 1 ) * NR ) + jr_offset, a_reg[4] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 2 ) * NR ) + jr_offset, a_reg[2] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 3 ) * NR ) + jr_offset, a_reg[6] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 4 ) * NR ) + jr_offset, a_reg[1] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 5 ) * NR ) + jr_offset, a_reg[5] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 6 ) * NR ) + jr_offset, a_reg[3] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 7 ) * NR ) + jr_offset, a_reg[7] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 8 ) * NR ) + jr_offset, a_reg[8] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 9 ) * NR ) + jr_offset, a_reg[12] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 10 ) * NR ) + jr_offset, a_reg[10] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 11 ) * NR ) + jr_offset, a_reg[14] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 12 ) * NR ) + jr_offset, a_reg[9] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 13 ) * NR ) + jr_offset, a_reg[13] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 14 ) * NR ) + jr_offset, a_reg[11] ); \
	_mm512_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 15 ) * NR ) + jr_offset, a_reg[15] ); \

#define MASK_STORE_PS_16x16(msk_arr) \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 0 ) * NR ) + jr_offset, \
					msk_arr[0], a_reg[0] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 1 ) * NR ) + jr_offset, \
					msk_arr[1], a_reg[4] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 2 ) * NR ) + jr_offset, \
					msk_arr[2], a_reg[2] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 3 ) * NR ) + jr_offset, \
					msk_arr[3], a_reg[6] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 4 ) * NR ) + jr_offset, \
					msk_arr[4], a_reg[1] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 5 ) * NR ) + jr_offset, \
					msk_arr[5], a_reg[5] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 6 ) * NR ) + jr_offset, \
					msk_arr[6], a_reg[3] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 7 ) * NR ) + jr_offset, \
					msk_arr[7], a_reg[7] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 8 ) * NR ) + jr_offset, \
					msk_arr[8], a_reg[8] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 9 ) * NR ) + jr_offset, \
					msk_arr[9], a_reg[12] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 10 ) * NR ) + jr_offset, \
					msk_arr[10], a_reg[10] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 11 ) * NR ) + jr_offset, \
					msk_arr[11], a_reg[14] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 12 ) * NR ) + jr_offset, \
					msk_arr[12], a_reg[9] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 13 ) * NR ) + jr_offset, \
					msk_arr[13], a_reg[13] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 14 ) * NR ) + jr_offset, \
					msk_arr[14], a_reg[11] ); \
	_mm512_mask_storeu_ps( pack_b_buffer + ( jc * KC ) + ( ( kr + 15 ) * NR ) + jr_offset, \
					msk_arr[15], a_reg[15] ); \

void packb_nr64_f32f32f32of32_col_major
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
    const dim_t NR = 64;
	const dim_t n_sub_blk_wdth = 16;
	const dim_t k_reg_size = 16;

	__m512 a_reg[16];
	__m512 b_reg[16];

	dim_t n_full_pieces_loop_limit = ( NC / NR ) * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_full_pieces_loop_limit = ( KC / k_reg_size ) * k_reg_size;
	dim_t k_partial_pieces = KC % k_reg_size;

	// First permute sequences.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector2 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	// Second permute sequences.
	__m512i selector1_1 = _mm512_setr_epi64( 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB );
	__m512i selector2_1 = _mm512_setr_epi64( 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF );

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t jr = jc; jr < jc + NR; jr += n_sub_blk_wdth )
		{
			dim_t jr_offset = jr % NR;
			__mmask16 msk = _cvtu32_mask16( 0xFFFF );
			for ( dim_t kr = 0; kr < k_full_pieces_loop_limit; kr += k_reg_size )
			{
				MASK_LOAD_PS_16x16(msk);
				UNPACK_PS_16x16();
				UNPACK_PD_16x16();
				PERMUTE_R1_16x16(selector1, selector2);
				PERMUTE_R2_16x16(selector1_1, selector2_1);
				STORE_PS_16x16();
			}
			if ( k_partial_pieces > 0 )
			{
				msk = _cvtu32_mask16( 0xFFFF >> ( k_reg_size - k_partial_pieces ) );
				dim_t kr = k_full_pieces_loop_limit;

				MASK_LOAD_PS_16x16(msk);
				UNPACK_PS_16x16();
				UNPACK_PD_16x16();
				PERMUTE_R1_16x16(selector1, selector2);
				PERMUTE_R2_16x16(selector1_1, selector2_1);

				__mmask16 msk_arr[16];
				for ( int i = 0; i < k_partial_pieces; ++i )
				{
					msk_arr[i] = _cvtu32_mask16(0xFFFF);
				}
				for ( int i = k_partial_pieces; i < 16; ++i )
				{
					msk_arr[i] = _cvtu32_mask16(0x0);
				}
				MASK_STORE_PS_16x16(msk_arr);
			}
		}
	}

	if( n_partial_pieces > 0 )
	{
		dim_t jc = n_full_pieces_loop_limit;
		for ( dim_t jr = n_full_pieces_loop_limit; jr < NC; jr += n_sub_blk_wdth )
		{
			dim_t jr_offset = jr % NR;
			__mmask16 msk_n_arr[16];

			dim_t jr_elems = ( ( NC - jr ) >= n_sub_blk_wdth ) ? n_sub_blk_wdth : ( NC - jr );
			for ( int i = 0; i < jr_elems; ++i )
			{
				msk_n_arr[i] = _cvtu32_mask16( 0xFFFF );
			}
			for ( int i = jr_elems; i < n_sub_blk_wdth; ++i )
			{
				msk_n_arr[i] = _cvtu32_mask16( 0x0 );
			}

			for ( dim_t kr = 0; kr < k_full_pieces_loop_limit; kr += k_reg_size )
			{
				MASK_ARR_LOAD_PS_16x16(msk_n_arr);
				UNPACK_PS_16x16();
				UNPACK_PD_16x16();
				PERMUTE_R1_16x16(selector1, selector2);
				PERMUTE_R2_16x16(selector1_1, selector2_1);
				STORE_PS_16x16();
			}
			if ( k_partial_pieces > 0 )
			{
				for ( int i = 0; i < jr_elems; ++i )
				{
					msk_n_arr[i] = _cvtu32_mask16( 0xFFFF &
						( 0xFFFF >> ( k_reg_size - k_partial_pieces ) ) );
				}
				for ( int i = jr_elems; i < n_sub_blk_wdth; ++i )
				{
					msk_n_arr[i] = _cvtu32_mask16( 0x0 );
				}
				dim_t kr = k_full_pieces_loop_limit;

				MASK_ARR_LOAD_PS_16x16(msk_n_arr);
				UNPACK_PS_16x16();
				UNPACK_PD_16x16();
				PERMUTE_R1_16x16(selector1, selector2);
				PERMUTE_R2_16x16(selector1_1, selector2_1);

				__mmask16 msk_arr[16];
				for (int i = 0; i < k_partial_pieces; ++i)
				{
					msk_arr[i] = _cvtu32_mask16(0xFFFF);
				}
				for (int i = k_partial_pieces; i < 16; ++i)
				{
					msk_arr[i] = _cvtu32_mask16(0x0);
				}
				MASK_STORE_PS_16x16(msk_arr);
			}
		}
	}

	*rs_b = NR;
	*cs_b = 1;
}

#endif
