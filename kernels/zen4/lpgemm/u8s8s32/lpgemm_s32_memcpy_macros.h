/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_S32_MEMCPY_MACROS_H
#define LPGEMM_S32_MEMCPY_MACROS_H

// Copy macros to replace memcpy usage.
//
#define PASTE_S32_2TOKEN(tok,id) \
	tok ## id

#define PASTE_S32_3TOKEN(tok1,id,tok2) \
	tok1 ## id ## tok2

#define MEMCPY_S32_LT16_INIT(size) \
	dim_t part8 = ( size ) >> 3; \
	dim_t part4 = ( ( size ) - ( part8 << 3 ) ) >> 2; \
	dim_t part4_rem = ( size ) % 4; \
	dim_t frin_offset = 0; \

#define MEMCPY_S32_LT16_REINIT(size) \
	part8 = ( size ) >> 3; \
	part4 = ( ( size ) - ( part8 << 3 ) ) >> 2; \
	part4_rem = ( size ) % 4; \
	frin_offset = 0; \

// Copy for size < 4 for uint8 elements.
#define MEMCPY_S32GM_LT4_UINT8(dst_,src_,size) \
	{ \
		uint8_t* dst = ( uint8_t* )( dst_ ); \
		uint8_t* src = ( uint8_t* )( src_ ); \
		if ( ( size ) == 1 ) \
		{ \
			dst[0] = src[0]; \
		} \
		else if ( ( size ) == 2 ) \
		{ \
			dst[0] = src[0]; \
			dst[1] = src[1]; \
		} \
		else if ( ( size ) == 3 )\
		{ \
			dst[0] = src[0]; \
			dst[1] = src[1]; \
			dst[2] = src[2]; \
		} \
	} \

// NR modulo 4 case, remainder items are assigned as 1, 2, or 3 elements.
//
// Remainder 1 case for MR=6,5,4,3,2,1
#define MEMCPY_S32_LT16_REM4_PART1_4ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) )[frin_offset + fid]; \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id1) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id1) )[frin_offset + fid]; \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id2) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id2) )[frin_offset + fid]; \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id3) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id3) )[frin_offset + fid]; \

#define MEMCPY_S32_LT16_REM4_PART1_2ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) )[frin_offset + fid]; \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id1) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id1) )[frin_offset + fid]; \

#define MEMCPY_S32_LT16_REM4_PART1_1ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) )[frin_offset + fid] = ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) )[frin_offset + fid]; \

#define MEMCPY_S32_LT16_REM4_PART1_6ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_4ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_2ELE_INT32(dst,src,SINGLE_TYPE,fid,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART1_5ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_4ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_1ELE_INT32(dst,src,SINGLE_TYPE,fid,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART1_3ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_2ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_1ELE_INT32(dst,src,SINGLE_TYPE,fid,id2,id5,id2,id3,id4,id5) \

// Remainder 2 case for MR=6,5,4,3,2,1
#define MEMCPY_S32_LT16_REM4_PART2_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) ) + frin_offset ); \
	ds1[0] = sr1[0]; \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id1) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id1) ) + frin_offset ); \
	ds1[0] = sr1[0]; \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id2) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id2) ) + frin_offset ); \
	ds1[0] = sr1[0]; \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id3) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id3) ) + frin_offset ); \
	ds1[0] = sr1[0]; \

#define MEMCPY_S32_LT16_REM4_PART2_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) ) + frin_offset ); \
	ds1[0] = sr1[0]; \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id1) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id1) ) + frin_offset ); \
	ds1[0] = sr1[0]; \

#define MEMCPY_S32_LT16_REM4_PART2_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	ds1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(dst,id0) ) + frin_offset ); \
	sr1 = ( CAST_TYPE* )( ( ( SINGLE_TYPE* )PASTE_S32_2TOKEN(src,id0) ) + frin_offset ); \
	ds1[0] = sr1[0]; \

#define MEMCPY_S32_LT16_REM4_PART2_6ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART2_5ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART2_3ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id2,id5,id2,id3,id4,id5) \

// Remainder 3 case for MR=6,5,4,3,2,1
#define MEMCPY_S32_LT16_REM4_PART3_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_4ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART3_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_2ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART3_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART2_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART1_1ELE_INT32(dst,src,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART3_6ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART3_5ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_4ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_REM4_PART3_3ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_2ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_REM4_PART3_1ELE_INT32(dst,src,CAST_TYPE,SINGLE_TYPE,fid,id2,id5,id2,id3,id4,id5) \

// Copy macro for NR' < 4 case.
// Pre condition sizeof(CAST_TYPE) = 2 * sizeof(SINGLE_TYPE)
#define MEMCPY_S32_LT16_REM4_INT32(dst,src,NRID,CAST_TYPE,SINGLE_TYPE) \
	if ( part4_rem == 1 ) \
	{ \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_REM4_PART1_,NRID,ELE_INT32)(dst,src,SINGLE_TYPE,0,0,1,2,3,4,5) \
	} \
	else if ( part4_rem == 2 ) \
	{ \
		CAST_TYPE* ds1; \
		CAST_TYPE* sr1; \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_REM4_PART2_,NRID,ELE_INT32)(dst,src,CAST_TYPE,SINGLE_TYPE,0,1,2,3,4,5) \
	} \
	else if ( part4_rem == 3 )\
	{ \
		CAST_TYPE* ds1; \
		CAST_TYPE* sr1; \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_REM4_PART3_,NRID,ELE_INT32)(dst,src,CAST_TYPE,SINGLE_TYPE,2,0,1,2,3,4,5) \
	} \

// int32_t 256 Store Load
#define MEMCPY_S32_LT16_STR32_M256_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id0) ) ); \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id1), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id1) ) ); \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id2), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id2) ) ); \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id3), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id3) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_2ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id0) ) ); \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id1), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id1) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_1ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_epi32( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_epi32( PASTE_S32_2TOKEN(src,id0) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_6ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_2ELE_INT32(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M256_5ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_1ELE_INT32(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M256_3ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_2ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_1ELE_INT32(dst,src,id2,id5,id2,id3,id4,id5) \

// int32_t 128 Store Load
#define MEMCPY_S32_LT16_STR32_M128_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id1) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id1) + frin_offset ) ) ); \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id2) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id2) + frin_offset ) ) ); \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id3) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id3) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_2ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id1) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id1) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_1ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_epi32( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_epi32( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_6ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_2ELE_INT32(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M128_5ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_4ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_1ELE_INT32(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M128_3ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_2ELE_INT32(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_1ELE_INT32(dst,src,id2,id5,id2,id3,id4,id5) \

// float 256 Store Load
#define MEMCPY_S32_LT16_STR32_M256_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id0) ) ); \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id1), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id1) ) ); \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id2), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id2) ) ); \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id3), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id3) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_2ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id0) ) ); \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id1), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id1) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_1ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm256_storeu_ps( PASTE_S32_2TOKEN(dst,id0), _mm256_loadu_ps( PASTE_S32_2TOKEN(src,id0) ) ); \

#define MEMCPY_S32_LT16_STR32_M256_6ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_2ELE_FLOAT(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M256_5ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_1ELE_FLOAT(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M256_3ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_2ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M256_1ELE_FLOAT(dst,src,id2,id5,id2,id3,id4,id5) \

// float 128 Store Load
#define MEMCPY_S32_LT16_STR32_M128_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id1) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id1) + frin_offset ) ) ); \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id2) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id2) + frin_offset ) ) ); \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id3) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id3) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_2ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id1) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id1) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_1ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	_mm_storeu_ps( ( PASTE_S32_2TOKEN(dst,id0) + frin_offset ), _mm_loadu_ps( ( PASTE_S32_2TOKEN(src,id0) + frin_offset ) ) ); \

#define MEMCPY_S32_LT16_STR32_M128_6ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_2ELE_FLOAT(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M128_5ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_4ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_1ELE_FLOAT(dst,src,id4,id5,id2,id3,id4,id5) \

#define MEMCPY_S32_LT16_STR32_M128_3ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_2ELE_FLOAT(dst,src,id0,id1,id2,id3,id4,id5) \
	MEMCPY_S32_LT16_STR32_M128_1ELE_FLOAT(dst,src,id2,id5,id2,id3,id4,id5) \

// Main macro for int32_t copy for lt 16 elems.
#define MEMCPY_S32_LT16_INT32(NRID,CAST2_TYPE,SINGLE_TYPE,dst,src) \
{ \
	if ( part8 == 1 ) \
	{ \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_STR32_M256_,NRID,ELE_INT32)(dst,src,0,1,2,3,4,5) \
		frin_offset += 8; \
	} \
	if ( part4 == 1 ) \
	{ \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_STR32_M128_,NRID,ELE_INT32)(dst,src,0,1,2,3,4,5) \
		frin_offset += 4; \
	} \
	MEMCPY_S32_LT16_REM4_INT32(dst,src,NRID,CAST2_TYPE,SINGLE_TYPE) \
} \

// Reusing the int32_t based macros for int8 copy by modifying the types.
// Main macro for int8_t copy for lt 16 elems.
#define MEMCPY_S32_LT16_INT8(NRID,CAST8_TYPE,CAST4_TYPE,CAST2_TYPE,SINGLE_TYPE,dst,src) \
{ \
	if ( part8 == 1 ) \
	{ \
		CAST8_TYPE* ds1; \
		CAST8_TYPE* sr1; \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_REM4_PART2_,NRID,ELE_INT32)(dst,src,CAST8_TYPE,SINGLE_TYPE,0,1,2,3,4,5) \
		frin_offset += 8; \
	} \
	if ( part4 == 1 ) \
	{ \
		CAST4_TYPE* ds1; \
		CAST4_TYPE* sr1; \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_REM4_PART2_,NRID,ELE_INT32)(dst,src,CAST4_TYPE,SINGLE_TYPE,0,1,2,3,4,5) \
		frin_offset += 4; \
	} \
	MEMCPY_S32_LT16_REM4_INT32(dst,src,NRID,CAST2_TYPE,SINGLE_TYPE) \
} \

// Main macro for int32_t copy for lt 16 elems.
#define MEMCPY_S32_LT16_FLOAT(NRID,CAST2_TYPE,SINGLE_TYPE,dst,src) \
{ \
	if ( part8 == 1 ) \
	{ \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_STR32_M256_,NRID,ELE_FLOAT)(dst,src,0,1,2,3,4,5) \
		frin_offset += 8; \
	} \
	if ( part4 == 1 ) \
	{ \
		PASTE_S32_3TOKEN(MEMCPY_S32_LT16_STR32_M128_,NRID,ELE_FLOAT)(dst,src,0,1,2,3,4,5) \
		frin_offset += 4; \
	} \
	MEMCPY_S32_LT16_REM4_INT32(dst,src,NRID,CAST2_TYPE,SINGLE_TYPE) \
} \

#endif //LPGEMM_S32_MEMCPY_MACROS_H
