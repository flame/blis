/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define F32_ROW_MAJOR_K_PACK_LOOP_AVX2() \
	a0 = _mm256_unpacklo_ps( a01, b0 ); \
	b0 = _mm256_unpackhi_ps( a01, b0 ); \
 \
	c0 = _mm256_unpacklo_ps( c01, d0 ); \
	d0 = _mm256_unpackhi_ps( c01, d0 ); \
 \
	e0 = _mm256_unpacklo_ps( e01, f0 ); \
	f0 = _mm256_unpackhi_ps( e01, f0 ); \
 \
	a01 = _mm256_castpd_ps( _mm256_unpacklo_pd( _mm256_castps_pd( a0 ), \
			_mm256_castps_pd( c0 ) ) ); \
	a0 = _mm256_castpd_ps( _mm256_unpackhi_pd( _mm256_castps_pd( a0 ), \
			_mm256_castps_pd( c0 ) ) ); \
 \
	c01 = _mm256_castpd_ps( _mm256_unpacklo_pd( _mm256_castps_pd( b0 ), \
			_mm256_castps_pd( d0 ) ) ); \
	c0 = _mm256_castpd_ps( _mm256_unpackhi_pd( _mm256_castps_pd( b0 ), \
			_mm256_castps_pd( d0 ) ) ); \
 \
	a0_128 = _mm256_castps256_ps128( a01 ); \
	b0_128 = _mm256_castps256_ps128( a0 ); \
	c0_128 = _mm256_castps256_ps128( c01 ); \
	d0_128 = _mm256_castps256_ps128( c0 ); \
	e0_128 = _mm256_castps256_ps128( e0 ); \
	f0_128 = _mm256_castps256_ps128( f0 ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 0, a0_128 ); \
	_mm_storel_pd( ( double*)( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 4 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 6, b0_128 ); \
	_mm_storeh_pd( ( double* )( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 10 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 12, c0_128 ); \
	_mm_storel_pd( ( double* )( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 16 ), \
		_mm_castps_pd( f0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 18, d0_128 ); \
	_mm_storeh_pd( ( double* )( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 22 ), \
		_mm_castps_pd( f0_128 ) ); \
 \
	a0_128 = _mm256_extractf128_ps( a01, 0x1 ); \
	b0_128 = _mm256_extractf128_ps( a0, 0x1 ); \
	c0_128 = _mm256_extractf128_ps( c01, 0x1 ); \
	d0_128 = _mm256_extractf128_ps( c0, 0x1 ); \
	e0_128 = _mm256_extractf128_ps( e0, 0x1 ); \
	f0_128 = _mm256_extractf128_ps( f0, 0x1 ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 24, a0_128 ); \
	_mm_storel_pd( ( double* )( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 28 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 30, b0_128 ); \
	_mm_storeh_pd( ( double*)(pack_a_buf + ( ic * KC ) + ( kr * MR ) + 34 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 36, c0_128 ); \
	_mm_storel_pd( ( double*)(pack_a_buf + ( ic * KC ) + ( kr * MR ) + 40 ), \
		_mm_castps_pd( f0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 42, d0_128 ); \
	_mm_storeh_pd( ( double*)(pack_a_buf + ( ic * KC ) + ( kr * MR ) + 46 ), \
		_mm_castps_pd( f0_128 ) ); \


#define F32_ROW_MAJOR_K_PACK_LOOP_SSE() \
	a0_128 = _mm_unpacklo_ps( a01_128, b0_128 ); \
	b0_128 = _mm_unpackhi_ps( a01_128, b0_128 ); \
 \
	c0_128 = _mm_unpacklo_ps( c01_128, d0_128 ); \
	d0_128 = _mm_unpackhi_ps( c01_128, d0_128 ); \
 \
	e0_128 = _mm_unpacklo_ps( e01_128, f0_128 ); \
	f0_128 = _mm_unpackhi_ps( e01_128, f0_128 ); \
 \
	a01_128 = _mm_castpd_ps( _mm_unpacklo_pd( _mm_castps_pd( a0_128 ), \
			_mm_castps_pd( c0_128 ) ) ); \
	a0_128 = _mm_castpd_ps( _mm_unpackhi_pd( _mm_castps_pd( a0_128 ), \
			_mm_castps_pd( c0_128 ) ) ); \
 \
	c01_128 = _mm_castpd_ps( _mm_unpacklo_pd( _mm_castps_pd( b0_128 ), \
			_mm_castps_pd( d0_128 ) ) ); \
	c0_128 = _mm_castpd_ps( _mm_unpackhi_pd( _mm_castps_pd( b0_128 ), \
			_mm_castps_pd( d0_128 ) ) ); \
 \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 0, a01_128 ); \
	_mm_storel_pd( ( double*)( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 4 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 6, a0_128 ); \
	_mm_storeh_pd( ( double*)( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 10 ), \
		_mm_castps_pd( e0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 12, c01_128 ); \
	_mm_storel_pd( ( double*)( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 16 ), \
		_mm_castps_pd( f0_128 ) ); \
	_mm_storeu_ps( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 18, c0_128 ); \
	_mm_storeh_pd( ( double*)( pack_a_buf + ( ic * KC ) + ( kr * MR ) + 22 ), \
		_mm_castps_pd( f0_128 ) ); \

// Row Major Packing in blocks of MRxKC
void packa_f32f32f32of32_row_major_avx2
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  lda,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	const dim_t MR = 6;
	const dim_t KR_NDIM = 8;

	dim_t m_full_pieces = MC / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = MC % MR;

	dim_t kr_full_pieces = KC / KR_NDIM;
	dim_t kr_full_pieces_loop_limit = kr_full_pieces * KR_NDIM;
	dim_t kr_partial_pieces = KC % KR_NDIM;

	__m256 a0;
	__m256 b0;
	__m256 c0;
	__m256 d0;
	__m256 e0;
	__m256 f0;
	__m256 a01;
	__m256 c01;
	__m256 e01;
	__m128 a0_128;
	__m128 b0_128;
	__m128 c0_128;
	__m128 d0_128;
	__m128 e0_128;
	__m128 f0_128;

	for ( dim_t ic = 0; ic < m_full_pieces_loop_limit; ic += MR )
	{
		for ( dim_t kr = 0; kr < kr_full_pieces_loop_limit; kr += KR_NDIM )
		{
			a01 = _mm256_loadu_ps( a + ( lda * ( ic + 0 ) ) + kr );
			b0 = _mm256_loadu_ps( a + ( lda * ( ic + 1 ) ) + kr );
			c01 = _mm256_loadu_ps( a + ( lda * ( ic + 2 ) ) + kr );
			d0 = _mm256_loadu_ps( a + ( lda * ( ic + 3 ) ) + kr );
			e01 = _mm256_loadu_ps( a + ( lda * ( ic + 4 ) ) + kr );
			f0 = _mm256_loadu_ps( a + ( lda * ( ic + 5 ) ) + kr );

			F32_ROW_MAJOR_K_PACK_LOOP_AVX2();
		}
		if ( kr_partial_pieces > 0 )
		{
			dim_t kr_partial_pieces_4 = ( kr_partial_pieces / 4 ) * 4;
			dim_t kr_partial_pieces_rem = kr_partial_pieces - kr_partial_pieces_4;

			dim_t kr = kr_full_pieces_loop_limit;
			if ( kr_partial_pieces_4 > 0 )
			{
				__m128 a01_128 = _mm_loadu_ps( a + ( lda * ( ic + 0 ) ) + kr );
				b0_128 = _mm_loadu_ps( a + ( lda * ( ic + 1 ) ) + kr );
				__m128 c01_128 = _mm_loadu_ps( a + ( lda * ( ic + 2 ) ) + kr );
				d0_128 = _mm_loadu_ps( a + ( lda * ( ic + 3 ) ) + kr );
				__m128 e01_128 = _mm_loadu_ps( a + ( lda * ( ic + 4 ) ) + kr );
				f0_128 = _mm_loadu_ps( a + ( lda * ( ic + 5 ) ) + kr );

				F32_ROW_MAJOR_K_PACK_LOOP_SSE();
			}
			kr += kr_partial_pieces_4;
			if ( kr_partial_pieces_rem > 0 )
			{
				for ( int ii = 0; ii < kr_partial_pieces_rem; ++ii )
				{
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 0 ) =
						*( a + ( lda * ( ic + 0 ) ) + ( kr + ii ) );
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 1 ) =
						*( a + ( lda * ( ic + 1 ) ) + ( kr + ii ) );
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 2 ) =
						*( a + ( lda * ( ic + 2 ) ) + ( kr + ii ) );
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 3 ) =
						*( a + ( lda * ( ic + 3 ) ) + ( kr + ii ) );
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 4 ) =
						*( a + ( lda * ( ic + 4 ) ) + ( kr + ii ) );
					*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + 5 ) =
						*( a + ( lda * ( ic + 5 ) ) + ( kr + ii ) );
				}
			}
		}
	}
	if ( m_partial_pieces > 0 )
	{
		dim_t ic = m_full_pieces_loop_limit;
		__m256 temp_a_reg[6];
		for ( dim_t kr = 0; kr < kr_full_pieces_loop_limit; kr += KR_NDIM )
		{
			for ( int ii = 0; ii < m_partial_pieces; ++ii )
			{
				temp_a_reg[ii] = _mm256_loadu_ps( a + ( lda * ( ic + ii ) ) + kr );
			}
			for ( int ii = m_partial_pieces; ii < MR; ++ii )
			{
				temp_a_reg[ii] = _mm256_setzero_ps();
			}
			a01 = temp_a_reg[0];
			b0 = temp_a_reg[1];
			c01 = temp_a_reg[2];
			d0 = temp_a_reg[3];
			e01 = temp_a_reg[4];
			f0 = temp_a_reg[5];

			F32_ROW_MAJOR_K_PACK_LOOP_AVX2();
		}
		if ( kr_partial_pieces > 0 )
		{
			dim_t kr_partial_pieces_4 = ( kr_partial_pieces / 4 ) * 4;
			dim_t kr_partial_pieces_rem = kr_partial_pieces - kr_partial_pieces_4;

			dim_t kr = kr_full_pieces_loop_limit;
			if ( kr_partial_pieces_4 > 0 )
			{
				__m128 temp_a_reg_128[6] = {0};
				for ( int ii = 0; ii < m_partial_pieces; ++ii )
				{
					temp_a_reg_128[ii] = _mm_loadu_ps( a + ( lda * ( ic + ii ) ) + kr );
				}
				for ( int ii = m_partial_pieces; ii < MR; ++ii )
				{
					temp_a_reg_128[ii] = _mm_setzero_ps();
				}
				__m128 a01_128 = temp_a_reg_128[0];
				b0_128 = temp_a_reg_128[1];
				__m128 c01_128 = temp_a_reg_128[2];
				d0_128 = temp_a_reg_128[3];
				__m128 e01_128 = temp_a_reg_128[4];
				f0_128 = temp_a_reg_128[5];

				F32_ROW_MAJOR_K_PACK_LOOP_SSE();
			}
			kr += kr_partial_pieces_4;
			if ( kr_partial_pieces_rem > 0 )
			{
				for ( int ii = 0; ii < kr_partial_pieces_rem; ++ii )
				{
					for ( int jj = 0; jj < m_partial_pieces; ++jj )
					{
						*( pack_a_buf + ( ic * KC ) + ( ( kr + ii ) * MR ) + jj ) =
							*( a + ( lda * ( ic + jj ) ) + ( kr + ii ) );
					}
				}
			}
		}
	}

	*rs_a = 1;
	*cs_a = 6;
}

#define F32_COL_MAJOR_K_PACK_STORE_SSE() \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 0 ) ), \
	  a0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 0 ) + 4 ) ), \
	  _mm_castps_pd( a0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 1 ) ), \
	  b0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 1 ) + 4 ) ), \
	  _mm_castps_pd( b0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 2 ) ), \
	  c0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 2 ) + 4 ) ), \
	  _mm_castps_pd( c0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 3 ) ), \
	  d0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 3 ) + 4 ) ), \
	  _mm_castps_pd( d0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 4 ) ), \
	  e0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 4 ) + 4 ) ), \
	  _mm_castps_pd( e0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 5 ) ), \
	  f0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 5 ) + 4 ) ), \
	  _mm_castps_pd( f0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 6 ) ), \
	  g0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 6 ) + 4 ) ), \
	  _mm_castps_pd( g0_2e ) \
	); \
	_mm_storeu_ps \
	( \
	  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 7 ) ), \
	  h0 \
	); \
	_mm_store_sd \
	( \
	  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 7 ) + 4 ) ), \
	  _mm_castps_pd( h0_2e ) \
	); \

#define F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, dst, src) \
	cp_st = 0; \
	if ( m_partial_4 > 0 ) \
	{ \
		( dst )[cp_st + 0] = ( src )[cp_st + 0]; \
		( dst )[cp_st + 1] = ( src )[cp_st + 1]; \
		( dst )[cp_st + 2] = ( src )[cp_st + 2]; \
		( dst )[cp_st + 3] = ( src )[cp_st + 3]; \
		cp_st += 4; \
	} \
	if ( m_partial_2 > 0 ) \
	{ \
		( dst )[cp_st + 0] = ( src )[cp_st + 0]; \
		( dst )[cp_st + 1] = ( src )[cp_st + 1]; \
		cp_st += 2; \
	} \
	if ( m_partial_1 > 0 ) \
	{ \
		( dst )[cp_st + 0] = ( src )[cp_st + 0]; \
		cp_st += 1; \
	} \

void packa_f32f32f32of32_col_major_avx2
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  lda,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	const dim_t MR = 6;
	const dim_t KR_NDIM = 8;

	dim_t m_full_pieces = MC / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = MC % MR;

	dim_t kr_full_pieces = KC / KR_NDIM;
	dim_t kr_full_pieces_loop_limit = kr_full_pieces * KR_NDIM;
	dim_t kr_partial_pieces = KC % KR_NDIM;

	__m128 a0;
	__m128 b0;
	__m128 c0;
	__m128 d0;
	__m128 e0;
	__m128 f0;
	__m128 g0;
	__m128 h0;
	__m128 a0_2e;
	__m128 b0_2e;
	__m128 c0_2e;
	__m128 d0_2e;
	__m128 e0_2e;
	__m128 f0_2e;
	__m128 g0_2e;
	__m128 h0_2e;

	for ( dim_t ic = 0; ic < m_full_pieces_loop_limit; ic += MR )
	{
		for ( dim_t kr = 0; kr < kr_full_pieces_loop_limit; kr += KR_NDIM )
		{
			// First 4 elements.
			a0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 0 ) ) );
			b0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 1 ) ) );
			c0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 2 ) ) );
			d0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 3 ) ) );
			e0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 4 ) ) );
			f0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 5 ) ) );
			g0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 6 ) ) );
			h0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 7 ) ) );

			// Last 2 elements.
			a0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 0 ) ) ) ) );
			b0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 1 ) ) ) ) );
			c0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 2 ) ) ) ) );
			d0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 3 ) ) ) ) );
			e0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 4 ) ) ) ) );
			f0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 5 ) ) ) ) );
			g0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 6 ) ) ) ) );
			h0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
						( lda * ( kr + 7 ) ) ) ) );

			F32_COL_MAJOR_K_PACK_STORE_SSE();
		}
		if ( kr_partial_pieces )
		{
			for ( dim_t kr = kr_full_pieces_loop_limit; kr < KC; ++kr )
			{
				a0 = _mm_loadu_ps( a + ic + ( lda * ( kr + 0 ) ) );
				a0_2e = _mm_castpd_ps( _mm_load_sd( ( double* )( a + ( ic + 4 ) + \
							( lda * ( kr + 0 ) ) ) ) );

				_mm_storeu_ps
				(
				  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 0 ) ),
				  a0
				);
				_mm_store_sd
				(
				  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + \
								( MR * 0 ) + 4 ) ),
				  _mm_castps_pd( a0_2e )
				);
			}
		}
	}
	if ( m_partial_pieces > 0 )
	{
		dim_t ic = m_full_pieces_loop_limit;
		dim_t m_partial_4 = ( m_partial_pieces / 4 ) * 4;
		dim_t m_partial_2 = ( ( m_partial_pieces - m_partial_4 ) / 2 ) * 2;
		dim_t m_partial_1 = m_partial_pieces - ( m_partial_4 + m_partial_2 );
		dim_t cp_st = 0;

		float temp_pack_a_buf[6] = { 0 };

		for ( dim_t kr = 0; kr < kr_full_pieces_loop_limit; kr += KR_NDIM )
		{
			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 0 ) ) );
			a0 = _mm_loadu_ps( temp_pack_a_buf );
			a0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 1 ) ) );
			b0 = _mm_loadu_ps( temp_pack_a_buf );
			b0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 2 ) ) );
			c0 = _mm_loadu_ps( temp_pack_a_buf );
			c0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 3 ) ) );
			d0 = _mm_loadu_ps( temp_pack_a_buf );
			d0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 4 ) ) );
			e0 = _mm_loadu_ps( temp_pack_a_buf );
			e0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 5 ) ) );
			f0 = _mm_loadu_ps( temp_pack_a_buf );
			f0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 6 ) ) );
			g0 = _mm_loadu_ps( temp_pack_a_buf );
			g0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
					a + ic + ( lda * ( kr + 7 ) ) );
			h0 = _mm_loadu_ps( temp_pack_a_buf );
			h0_2e = _mm_castpd_ps( _mm_load_sd(
						( double* )( temp_pack_a_buf + 4 ) ) );

			F32_COL_MAJOR_K_PACK_STORE_SSE();
		}
		if ( kr_partial_pieces )
		{
			for ( dim_t kr = kr_full_pieces_loop_limit; kr < KC; ++kr )
			{
				F32_COL_MAJOR_K_PACK_LOAD_MEMCPY(cp_st, temp_pack_a_buf, \
						a + ic + ( lda * ( kr + 0 ) ) );
				a0 = _mm_loadu_ps( temp_pack_a_buf );
				a0_2e = _mm_castpd_ps( _mm_load_sd(
							( double* )( temp_pack_a_buf + 4 ) ) );

				_mm_storeu_ps
				(
				  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 0 ) ),
				  a0
				);
				_mm_store_sd
				(
				  ( double* )( pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + \
								( MR * 0 ) + 4 ) ),
				  _mm_castps_pd( a0_2e )
				);
			}
		}
	}
}

void packa_mr6_f32f32f32of32_avx2
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  rs,
       const dim_t  cs,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	if( cs == 1 )
	{
		packa_f32f32f32of32_row_major_avx2
		( pack_a_buf, a, rs, MC, KC, rs_a, cs_a );
	}
	else
	{
		packa_f32f32f32of32_col_major_avx2
		( pack_a_buf, a, cs, MC, KC, rs_a, cs_a );
	}
}

#endif
