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

#include "lpgemm_f32_kern_macros.h"
#include "../int4_utils_avx512.h"

#ifdef LPGEMM_BF16_JIT

void packsclb_nr64_bf16s4f32of32
    (
      bfloat16* packb_bf16,
      const int8_t* b,
      const dim_t NC,
      const dim_t KC,
      dim_t *rs_p,
      dim_t *cs_p,
      lpgemm_pre_op_attr pre_ops_attr
    )
{
    //This bf16 packB_s4_bf16 is Not supported for gcc<11.2
}

#else //LPGEMM_BF16_JIT
/*
input:__m512i containing 64 int8 elements
output: two __m512 containing 16 f32 elements
*/
#define CVT_INT8_F32_SCAL_16( in, idx, scale_reg) \
    (_mm512_mul_ps( \
      _mm512_cvtepi32_ps( \
       _mm512_cvtepi8_epi32( \
        _mm512_extracti32x4_epi32( in, idx ) ) ), scale_reg ) )

void packsclb_nr48_bf16s4f32of32
(
    bfloat16* packb_bf16,
    const int8_t* b,
    const dim_t KC,
    bool signed_upscale,
    lpgemm_pre_op_attr pre_ops_attr
)
{
    dim_t NR = 48;

    dim_t pre_op_off = pre_ops_attr.pre_op_b_j;

    /* Regs to load int4 elements */
    __m256i ymm0, ymm1;
    /* Regs to store zero-point values */
    __m512i zero_point, zero_point0, zero_point1;
    /* Regs to store scale factor values */
    __m512 zmm4, zmm5, zmm6, zmm7, zmm8, zmm9;
    /* Regs to store intermediate int8 elements */
    __m512i zmm14, zmm15;
    /* Regs to store bf16 values */
    __m512bh zmm0, zmm1, zmm2;
    /* Regs to store masks */
    __m512i mask_zp1, mask_zp2, mask_scale1, mask_scale2;

    __m512i shift_idx_64;

    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);

    mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

    mask_zp2 = _mm512_set_epi64( 0x7F3F7E3E7D3D7C3C, 0x7B3B7A3A79397838,
                                 0x7737763675357434, 0x7333723271317030,
                                 0x6F2F6E2E6D2D6C2C, 0x6B2B6A2A69296828,
                                 0x6727662665256424, 0x6323622261216020 );

    mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
                                    0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
                                    0x10, 0x00 );

    mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
                                    0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
                                    0x18, 0x08);

   __m512i sign_comp = _mm512_set1_epi8(0x08);

    int8_t* b_group;
    bfloat16* packb_group;

    dim_t group_start = pre_ops_attr.pre_op_b_i / pre_ops_attr.group_size;
	dim_t group_end   = ( pre_ops_attr.pre_op_b_i + KC - 1 ) / pre_ops_attr.group_size;

    b_group = ( int8_t*) b;
    packb_group = packb_bf16;

    for( dim_t group = group_start; group <= group_end; group++ )
    {
        /* offset if pre_op_len == 'n' */
        pre_op_off = ( group * pre_ops_attr.pre_op_ld ) + pre_ops_attr.pre_op_b_j;

        /* load zero_point values */
        if( pre_ops_attr.zero_point_len == 0 )
        {
            zero_point = _mm512_setzero_si512();
        }
        else if ( pre_ops_attr.zero_point_len == 1 )
        {
            zero_point =
            _mm512_set1_epi8( *( ( int8_t* )pre_ops_attr.zero_point + group) );
        }
        else
        {
            zero_point = _mm512_maskz_loadu_epi8( 0xFFFFFFFFFFFF,
                                                  ( pre_ops_attr.zero_point +
                                                  pre_op_off ) );
        }


        /* interleave zero-point values */
        zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_point );
        zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

        if( pre_ops_attr.scale_factor_len > 1 )
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                                pre_op_off );
                zmm6 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor )
                                                + pre_op_off + 16 );
                zmm8 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                                pre_op_off + 32 );
            }
            else
            {
                // load and convert scale factor vectors to F32 type
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off) );
                zmm6 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off + 16 ) );
                zmm8 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off + 32 ) );
            }

            zmm5 = _mm512_permutex2var_ps( zmm4, mask_scale2, zmm4 );
            zmm4 = _mm512_permutex2var_ps( zmm4, mask_scale1, zmm4 );
            zmm7 = _mm512_permutex2var_ps( zmm6, mask_scale2, zmm6 );
            zmm6 = _mm512_permutex2var_ps( zmm6, mask_scale1, zmm6 );
            zmm9 = _mm512_permutex2var_ps( zmm8, mask_scale2, zmm8 );
            zmm8 = _mm512_permutex2var_ps( zmm8, mask_scale1, zmm8 );
        }
        else
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_set1_ps(*( ( float* )pre_ops_attr.scale_factor
                                                        + group ));
            }
            else
            {
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_set1_epi16(
                        *( ( bfloat16* )pre_ops_attr.scale_factor + group ) ));
            }

            zmm5 = zmm4;
            zmm6 = zmm4;
            zmm7 = zmm4;
            zmm8 = zmm4;
            zmm9 = zmm4;
        }

        dim_t k_start = bli_max( group * pre_ops_attr.group_size,
                                    pre_ops_attr.pre_op_b_i );
        dim_t k_end = bli_min((( group + 1 ) * pre_ops_attr.group_size - 1 ),
                                pre_ops_attr.pre_op_b_i + KC - 1);

        dim_t kg0 = k_end - k_start + 1;
        dim_t k_full_piece_blks_per_group = kg0 / 2;
        dim_t k_full_pieces_per_group = k_full_piece_blks_per_group * 2;
        dim_t k_partial_pieces = kg0 % 2;

        for( dim_t kr = 0; kr < k_full_pieces_per_group; kr += 2 )
        {
            ymm0 = _mm256_loadu_si256((__m256i const* )(b_group + ( kr * NR ) / 2 ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                        CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );

            ymm1 = _mm256_maskz_loadu_epi8(0xFFFF, ( b_group +
                                           ( kr * NR + 64 ) / 2 ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm1, zmm15, shift_idx_64, \
                                                sign_comp, signed_upscale);

            zmm15 = _mm512_sub_epi8( zmm15, zero_point1 );

            zmm2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 1, zmm9),
                                        CVT_INT8_F32_SCAL_16( zmm15, 0, zmm8) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ), (__m512i)zmm0 );
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ) + 32,
                                (__m512i)zmm1 );
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ) + 64,
                                (__m512i)zmm2 );
        }

        b_group += ( k_full_pieces_per_group * NR ) / 2;
        packb_group += k_full_pieces_per_group * NR;

        // Group_size is always even, so k_partial_pieces will always
        // appear in the last group. So, packb_group and b_group pointers
        // need not be updated after handling k_partial pieces.
        if( k_partial_pieces )
        {
            __m512i zero_reg = _mm512_setzero_si512();

            /* Interleave zero_point values with zeroes */
            zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_reg );
            zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

            ymm0 = _mm256_loadu_si256( (__m256i const *)(b_group ) );
            ymm1 = _mm256_maskz_loadu_epi8(0xFFFF, b_group + 32 );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );
            zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                        CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );


            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm1, zmm15, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm15 = _mm512_sub_epi8( zmm15, zero_point1 );

            zmm2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 1, zmm9),
                                        CVT_INT8_F32_SCAL_16( zmm15, 0, zmm8) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group, (__m512i)zmm0 );
            _mm512_storeu_si512( packb_group + 32, (__m512i)zmm1 );
            _mm512_storeu_si512( packb_group + 64 , (__m512i)zmm2 );
        }
    }
}

void packsclb_nr32_bf16s4f32of32
(
    bfloat16* packb_bf16,
    const int8_t* b,
    const dim_t KC,
    bool signed_upscale,
    lpgemm_pre_op_attr pre_ops_attr
)
{
    dim_t NR = 32;

    dim_t pre_op_off = pre_ops_attr.pre_op_b_j;

    /* Regs to load int4 elements */
    __m256i ymm0;
    /* Regs to store zero-point values */
    __m512i zero_point, zero_point0;
    /* Regs to store scale factor values */
    __m512 zmm4, zmm5, zmm6, zmm7;
    /* Regs to store intermediate int8 elements */
    __m512i zmm14;
    /* Regs to store bf16 values */
    __m512bh zmm0, zmm1;
    /* Regs to store masks */
    __m512i mask_zp1, mask_scale1, mask_scale2;

    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp = _mm512_set1_epi8(0x08);

    mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

    mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
                                    0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
                                    0x10, 0x00 );

    mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
                                    0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
                                    0x18, 0x08);

    int8_t* b_group;
    bfloat16* packb_group;

    dim_t group_start = pre_ops_attr.pre_op_b_i / pre_ops_attr.group_size;
	dim_t group_end   = ( pre_ops_attr.pre_op_b_i + KC - 1 ) / pre_ops_attr.group_size;

    b_group = ( int8_t*) b;
    packb_group = packb_bf16;

    for( dim_t group = group_start; group <= group_end; group++ )
    {
        /* offset if pre_op_len == 'n' */
        pre_op_off = ( group * pre_ops_attr.pre_op_ld ) + pre_ops_attr.pre_op_b_j;

        /* load zero_point values */
        if( pre_ops_attr.zero_point_len == 0 )
        {
            zero_point = _mm512_setzero_si512();
        }
        else if ( pre_ops_attr.zero_point_len == 1 )
        {
            zero_point =
            _mm512_set1_epi8( *( ( int8_t* )pre_ops_attr.zero_point + group) );
        }
        else
        {
            zero_point = _mm512_maskz_loadu_epi8( 0xFFFFFFFF,
                                                  ( pre_ops_attr.zero_point +
                                                  pre_op_off ) );
        }

        /* interleave zero-point values */
        zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

        if( pre_ops_attr.scale_factor_len > 1 )
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                                pre_op_off );
                zmm6 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor )
                                                + pre_op_off + 16 );
             }
            else
            {
                // load and convert scale factor vectors to F32 type
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off) );
                zmm6 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off + 16 ) );
             }

            zmm5 = _mm512_permutex2var_ps( zmm4, mask_scale2, zmm4 );
            zmm4 = _mm512_permutex2var_ps( zmm4, mask_scale1, zmm4 );
            zmm7 = _mm512_permutex2var_ps( zmm6, mask_scale2, zmm6 );
            zmm6 = _mm512_permutex2var_ps( zmm6, mask_scale1, zmm6 );
        }
        else
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_set1_ps(*( ( float* )pre_ops_attr.scale_factor
                                                        + group ));
            }
            else
            {
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_set1_epi16(
                        *( ( bfloat16* )pre_ops_attr.scale_factor + group ) ));
            }

            zmm5 = zmm4;
            zmm6 = zmm4;
            zmm7 = zmm4;
        }

        dim_t k_start = bli_max( group * pre_ops_attr.group_size,
                                    pre_ops_attr.pre_op_b_i );
        dim_t k_end = bli_min((( group + 1 ) * pre_ops_attr.group_size - 1 ),
                                pre_ops_attr.pre_op_b_i + KC - 1);

        dim_t kg0 = k_end - k_start + 1;
        dim_t k_full_piece_blks_per_group = kg0 / 2;
        dim_t k_full_pieces_per_group = k_full_piece_blks_per_group * 2;
        dim_t k_partial_pieces = kg0 % 2;

        for( dim_t kr = 0; kr < k_full_pieces_per_group; kr += 2 )
        {
            ymm0 = _mm256_loadu_si256((__m256i const* )(b_group + ( kr * NR ) / 2 ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                        CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ), (__m512i)zmm0 );
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ) + 32,
                                (__m512i)zmm1 );
        }
        b_group += ( k_full_pieces_per_group * NR ) / 2;
        packb_group += k_full_pieces_per_group * NR;

        // Group_size is always even, so k_partial_pieces will always
        // appear in the last group. So, packb_group and b_group pointers
        // need not be updated after handling k_partial pieces.
        if( k_partial_pieces )
        {
            __m512i zero_reg = _mm512_setzero_si512();

            /* Interleave zero_point values with zeroes */
            zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

            ymm0 = _mm256_loadu_si256( (__m256i const *)(b_group ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );
            zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                        CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group, (__m512i)zmm0 );
            _mm512_storeu_si512( packb_group + 32, (__m512i)zmm1 );
        }
    }
}


void packsclb_nr16_bf16s4f32of32
(
    bfloat16* packb_bf16,
    const int8_t* b,
    const dim_t KC,
    bool signed_upscale,
    lpgemm_pre_op_attr pre_ops_attr
)
{
    dim_t NR = 16;
    dim_t pre_op_off = pre_ops_attr.pre_op_b_j;

    /* Regs to load int4 elements */
    __m256i ymm0;
    /* Regs to store zero-point values */
    __m512i zero_point, zero_point0;
    /* Regs to store scale factor values */
    __m512 zmm4, zmm5;
    /* Regs to store intermediate int8 elements */
    __m512i zmm14;
    /* Regs to store bf16 values */
    __m512bh zmm0;
    /* Regs to store masks */
    __m512i mask_zp1, mask_scale1, mask_scale2;

    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp = _mm512_set1_epi8(0x08);

    mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                0x5717561655155414, 0x5313521251115010,
                                0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                0x4707460645054404, 0x4303420241014000 );

    mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
                                    0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
                                    0x10, 0x00 );

    mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
                                    0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
                                    0x18, 0x08);

    int8_t* b_group;
    bfloat16* packb_group;

    dim_t group_start = pre_ops_attr.pre_op_b_i / pre_ops_attr.group_size;
	dim_t group_end   = ( pre_ops_attr.pre_op_b_i + KC - 1 ) / pre_ops_attr.group_size;

    b_group = ( int8_t*) b;
    packb_group = packb_bf16;

    for( dim_t group = group_start; group <= group_end; group++ )
    {
        /* offset if pre_op_len == 'n' */
        pre_op_off = ( group * pre_ops_attr.pre_op_ld ) + pre_ops_attr.pre_op_b_j;

        /* load zero_point values */
        if( pre_ops_attr.zero_point_len == 0 )
        {
            zero_point = _mm512_setzero_si512();
        }
        else if ( pre_ops_attr.zero_point_len == 1 )
        {
            zero_point =
            _mm512_set1_epi8( *( ( int8_t* )pre_ops_attr.zero_point + group) );
        }
        else
        {
            zero_point = _mm512_maskz_loadu_epi8( 0xFFFF,
                                                  ( pre_ops_attr.zero_point +
                                                  pre_op_off ) );
        }

        /* interleave zero-point values */
        zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

        if( pre_ops_attr.scale_factor_len > 1 )
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                                pre_op_off );
             }
            else
            {
                // load and convert scale factor vectors to F32 type
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off) );
            }
            zmm5 = _mm512_permutex2var_ps( zmm4, mask_scale2, zmm4 );
            zmm4 = _mm512_permutex2var_ps( zmm4, mask_scale1, zmm4 );
        }
        else
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_set1_ps(*( ( float* )pre_ops_attr.scale_factor
                                                        + group ));
            }
            else
            {
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_set1_epi16(
                        *( ( bfloat16* )pre_ops_attr.scale_factor + group ) ));
            }

            zmm5 = zmm4;
        }

        dim_t k_start = bli_max( group * pre_ops_attr.group_size,
                                    pre_ops_attr.pre_op_b_i );
        dim_t k_end = bli_min((( group + 1 ) * pre_ops_attr.group_size - 1 ),
                                pre_ops_attr.pre_op_b_i + KC - 1);

        dim_t kg0 = k_end - k_start + 1;
        dim_t k_full_piece_blks_per_group = kg0 / 2;
        dim_t k_full_pieces_per_group = k_full_piece_blks_per_group * 2;
        dim_t k_partial_pieces = kg0 % 2;

        for( dim_t kr = 0; kr < k_full_pieces_per_group; kr += 2 )
        {
            ymm0 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )(b_group
                                                        + ( kr * NR ) / 2 ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group + ( ( kr + 0 ) * NR ), (__m512i)zmm0 );
        }
        b_group += ( k_full_pieces_per_group * NR ) / 2;
        packb_group += k_full_pieces_per_group * NR;

        // Group_size is always even, so k_partial_pieces will always
        // appear in the last group. So, packb_group and b_group pointers
        // need not be updated after handling k_partial pieces.
        if( k_partial_pieces )
        {
            __m512i zero_reg = _mm512_setzero_si512();

            /* Interleave zero_point values with zeroes */
            zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

            ymm0 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )(b_group) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            //store to pack_b buffer
            _mm512_storeu_si512( packb_group, (__m512i)zmm0 );
        }
    }
}


void packsclb_nrlt16_bf16s4f32of32
(
    bfloat16* packb_bf16,
    const int8_t* b,
    const dim_t KC,
    const dim_t n_rem,
    bool signed_upscale,
    lpgemm_pre_op_attr pre_ops_attr
)
{
    dim_t NR = 16;
    dim_t pre_op_off = pre_ops_attr.pre_op_b_j;

    /* Regs to load int4 elements */
    __m256i ymm0;
    /* Regs to store zero-point values */
    __m512i zero_point, zero_point0;
    /* Regs to store scale factor values */
    __m512 zmm4, zmm5;
    /* Regs to store intermediate int8 elements */
    __m512i zmm14;
    /* Regs to store bf16 values */
    __m512bh zmm0;
    /* Regs to store masks */
    __m512i mask_zp1, mask_scale1, mask_scale2;

    __m512i shift_idx_64;

    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);

    __m512i sign_comp = _mm512_set1_epi8(0x08);

    __mmask16 lmask = _cvtu32_mask16( 0xFFFF >> ( 16 - n_rem ) );

    mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

    mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
                                    0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
                                    0x10, 0x00 );

    mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
                                    0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
                                    0x18, 0x08);

    int8_t* b_group;
    bfloat16* packb_group;

    dim_t group_start = pre_ops_attr.pre_op_b_i / pre_ops_attr.group_size;
	dim_t group_end   = ( pre_ops_attr.pre_op_b_i + KC - 1 ) / pre_ops_attr.group_size;

    b_group = ( int8_t*) b;
    packb_group = packb_bf16;

    for( dim_t group = group_start; group <= group_end; group++ )
    {
        /* offset if pre_op_len == 'n' */
        pre_op_off = ( group * pre_ops_attr.pre_op_ld ) + pre_ops_attr.pre_op_b_j;

        /* load zero_point values */
        if( pre_ops_attr.zero_point_len == 0 )
        {
            zero_point = _mm512_setzero_si512();
        }
        else if ( pre_ops_attr.zero_point_len == 1 )
        {
            zero_point =
            _mm512_set1_epi8( *( ( int8_t* )pre_ops_attr.zero_point + group) );
        }
        else
        {
            zero_point = _mm512_maskz_loadu_epi8( lmask,
                                                  ( pre_ops_attr.zero_point +
                                                  pre_op_off ) );
        }

        /* interleave zero-point values */
        zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

        if( pre_ops_attr.scale_factor_len > 1 )
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_maskz_loadu_ps( lmask, (float*)( pre_ops_attr.scale_factor ) +
                                            pre_op_off );
             }
            else
            {
                // load and convert scale factor vectors to F32 type
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16(lmask,
                            (bfloat16*)( pre_ops_attr.scale_factor ) +
                                        pre_op_off) );
            }
            zmm5 = _mm512_permutex2var_ps( zmm4, mask_scale2, zmm4 );
            zmm4 = _mm512_permutex2var_ps( zmm4, mask_scale1, zmm4 );
        }
        else
        {
            if( pre_ops_attr.scale_factor_type == F32 )
            {
                zmm4 = _mm512_set1_ps(*( ( float* )pre_ops_attr.scale_factor
                                                        + group ));
            }
            else
            {
                zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_set1_epi16(
                        *( ( bfloat16* )pre_ops_attr.scale_factor + group ) ));
            }

            zmm5 = zmm4;
        }

        dim_t k_start = bli_max( group * pre_ops_attr.group_size,
                                    pre_ops_attr.pre_op_b_i );
        dim_t k_end = bli_min((( group + 1 ) * pre_ops_attr.group_size - 1 ),
                                pre_ops_attr.pre_op_b_i + KC - 1);

        dim_t kg0 = k_end - k_start + 1;
        dim_t k_full_piece_blks_per_group = kg0 / 2;
        dim_t k_full_pieces_per_group = k_full_piece_blks_per_group * 2;
        dim_t k_partial_pieces = kg0 % 2;

        for( dim_t kr = 0; kr < k_full_pieces_per_group; kr += 2 )
        {
            ymm0 = _mm256_maskz_loadu_epi8( lmask,
                                         (__m256i const* )(b_group + ( kr * NR ) / 2 ) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            //store to pack_b buffer
            _mm512_mask_storeu_epi32( packb_group + ( ( kr + 0 ) * NR ),
                                      lmask,  (__m512i)zmm0 );
        }

        b_group += ( k_full_pieces_per_group * NR ) / 2;
        packb_group += k_full_pieces_per_group * NR;

        // Group_size is always even, so k_partial_pieces will always
        // appear in the last group. So, packb_group and b_group pointers
        // need not be updated after handling k_partial pieces.
        if( k_partial_pieces )
        {
            __m512i zero_reg = _mm512_setzero_si512();

            /* Interleave zero_point values with zeroes */
            zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

            ymm0 = _mm256_maskz_loadu_epi8( lmask, (__m256i const* )(b_group) );

            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                sign_comp, signed_upscale);

            zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

            zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                        CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

            //store to pack_b buffer
            _mm512_mask_storeu_epi32( packb_group, lmask,  (__m512i)zmm0 );
        }
    }
}


void packsclb_nr64_bf16s4f32of32
    (
      bfloat16* packb_bf16,
      const int8_t* b,
      const dim_t NC,
      const dim_t KC,
      dim_t *rs_p,
      dim_t *cs_p,
      lpgemm_pre_op_attr pre_ops_attr
    )
{
    dim_t NR = 64;

    dim_t n_full_pieces = NC / NR;
    dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
    dim_t n_partial_pieces = NC % NR;
    bool signed_upscale = true;

    dim_t pre_op_off = pre_ops_attr.pre_op_b_j;

    dim_t group_start = pre_ops_attr.pre_op_b_i / pre_ops_attr.group_size;
	dim_t group_end   = ( pre_ops_attr.pre_op_b_i + KC - 1 ) / pre_ops_attr.group_size;

    /* Regs to store bf16 elems */
    __m512bh zmm0, zmm1, zmm2, zmm3;
    /* Regs to store F32 scale */
    __m512 zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11;
    /* Regs to store int8 elems zero-point values */
    __m512i zero_point, zero_point0, zero_point1;
    /* Reg to load int4 data */
    __m256i ymm0, ymm1;
    /* Reg to store intermediate int8 elements */
    __m512i zmm14, zmm15;
    /* Reg to store masks to interleave scale factor */
    __m512i mask_scale1, mask_scale2;
    /* Regs to store masks to interleave zero_point values */
    __m512i mask_zp1, mask_zp2;

    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);

    mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

    mask_zp2 = _mm512_set_epi64( 0x7F3F7E3E7D3D7C3C, 0x7B3B7A3A79397838,
                                 0x7737763675357434, 0x7333723271317030,
                                 0x6F2F6E2E6D2D6C2C, 0x6B2B6A2A69296828,
                                 0x6727662665256424, 0x6323622261216020 );

    mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
                                    0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
                                    0x10, 0x00 );

    mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
                                    0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
                                    0x18, 0x08);

    __m512i sign_comp = _mm512_set1_epi8(0x08);

    int8_t* b_group;
    bfloat16* packb_group;

    dim_t KC_updated = KC;
	KC_updated += (KC_updated & 0x1);

    for( dim_t jr = 0; jr < n_full_pieces_loop_limit; jr += NR )
    {
        b_group = ( int8_t*) b;
        packb_group = packb_bf16;

        for( dim_t group = group_start; group <= group_end; group++ )
        {
            /* offset if pre_op_len == 'n' */
            pre_op_off = ( group * pre_ops_attr.pre_op_ld ) + pre_ops_attr.pre_op_b_j;

            /* load zero_point values */
            if( pre_ops_attr.zero_point_len == 0 )
            {
                zero_point = _mm512_setzero_si512();
            }
            else if( pre_ops_attr.zero_point_len == 1 )
            {
                zero_point =
                _mm512_set1_epi8( *( ( int8_t* )pre_ops_attr.zero_point + group ) );
            }
            else
            {
                zero_point = _mm512_loadu_si512( ( pre_ops_attr.zero_point ) +
                                                 pre_op_off + jr );
            }

            /* interleave zero-point values */
            zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_point );
            zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

            /* load scale factor values */
            if( pre_ops_attr.scale_factor_len > 1 )
            {
                if( pre_ops_attr.scale_factor_type == F32 )
                {
                    // load and interleave scale factor vectors
                    zmm4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                            pre_op_off + jr);
                    zmm6 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                            pre_op_off + jr + 16 );
                    zmm8 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                            pre_op_off + jr + 32 );
                    zmm10 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
                                            pre_op_off + jr + 48 );
                }
                else
                {
                    // load and convert scale factor vectors to F32 type
					zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                             (bfloat16*)( pre_ops_attr.scale_factor ) +
                                          pre_op_off + jr) );
                    zmm6 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                             (bfloat16*)( pre_ops_attr.scale_factor ) +
                                          pre_op_off + jr + 16 ) );
                    zmm8 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                             (bfloat16*)( pre_ops_attr.scale_factor ) +
                                          pre_op_off + jr + 32 ) );
                    zmm10 = CVT_BF16_F32_INT_SHIFT(_mm256_loadu_epi16(
                             (bfloat16*)( pre_ops_attr.scale_factor ) +
                                          pre_op_off + jr + 48 ) );
                }


                zmm5 = _mm512_permutex2var_ps( zmm4, mask_scale2, zmm4 );
                zmm4 = _mm512_permutex2var_ps( zmm4, mask_scale1, zmm4 );
                zmm7 = _mm512_permutex2var_ps( zmm6, mask_scale2, zmm6 );
                zmm6 = _mm512_permutex2var_ps( zmm6, mask_scale1, zmm6 );
                zmm9 = _mm512_permutex2var_ps( zmm8, mask_scale2, zmm8 );
                zmm8 = _mm512_permutex2var_ps( zmm8, mask_scale1, zmm8 );
                zmm11 = _mm512_permutex2var_ps( zmm10, mask_scale2, zmm10 );
                zmm10 = _mm512_permutex2var_ps( zmm10, mask_scale1, zmm10 );

            }
            else
            {
                if( pre_ops_attr.scale_factor_type == F32 )
                {
                    zmm4 = _mm512_set1_ps(*( ( float* )pre_ops_attr.scale_factor
                                                           + group ));
                }
                else
                {
                    zmm4 = CVT_BF16_F32_INT_SHIFT(_mm256_set1_epi16(
                            *( (( bfloat16* )pre_ops_attr.scale_factor) + group ) ));
                }

                zmm5 = zmm4;
                zmm6 = zmm4;
                zmm7 = zmm4;
                zmm8 = zmm4;
                zmm9 = zmm4;
                zmm10 = zmm4;
                zmm11 = zmm4;
            }

            dim_t k_start = bli_max( group * pre_ops_attr.group_size,
                                     pre_ops_attr.pre_op_b_i );
            dim_t k_end = bli_min((( group + 1 ) * pre_ops_attr.group_size - 1 ),
                                  pre_ops_attr.pre_op_b_i + KC - 1);

            dim_t kg0 = k_end - k_start + 1;
            dim_t k_full_piece_blks_per_group = kg0 / 2;
            dim_t k_partial_pieces = kg0 % 2;
            dim_t k_full_pieces_per_group = k_full_piece_blks_per_group * 2;

            for( dim_t kr = 0; kr < k_full_pieces_per_group; kr += 2 )
            {
                // Int4 array has to be accessed like byte array, but with
                // half the elements traversed in the byte array.

                ymm0 = _mm256_loadu_si256(
                        (__m256i const *)(b_group + ( ( jr * KC_updated )
                        + ( ( kr + 0 ) * NR ) ) / 2 ) );
                ymm1 = _mm256_loadu_si256(
                        (__m256i const *)(b_group + ( ( jr * KC_updated )
                        + ( ( kr + 1 ) * NR ) ) / 2 ) );

                CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                    sign_comp, signed_upscale);

                zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

                zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                            CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );

                zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                            CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );


                CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm1, zmm15, shift_idx_64, \
                                    sign_comp, signed_upscale);

                zmm15 = _mm512_sub_epi8( zmm15, zero_point1 );

                zmm2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 1, zmm9),
                                            CVT_INT8_F32_SCAL_16( zmm15, 0, zmm8) );
                zmm3 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 3, zmm11),
                                            CVT_INT8_F32_SCAL_16( zmm15, 2, zmm10) );

                //store to pack_b buffer
                _mm512_storeu_si512( packb_group + ( jr * KC_updated ) +
                                    ( ( kr + 0 ) * NR ), (__m512i)zmm0 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated ) +
                                    ( ( kr + 0 ) * NR ) + 32, (__m512i)zmm1 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated ) +
                                    ( ( kr + 1 ) * NR ), (__m512i)zmm2 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated ) +
                                    ( ( kr + 1 ) * NR ) + 32, (__m512i)zmm3 );

            }

            b_group += ( k_full_pieces_per_group * NR ) / 2;
            packb_group += k_full_pieces_per_group * NR;

            // Group_size is always even, so k_partial_pieces will always
            // appear in the last group. So, packb_group and b_group pointers
            // need not be updated after handling k_partial pieces.
            if( k_partial_pieces )
            {
                __m512i zero_reg = _mm512_setzero_si512();

                /* Interleave zero_point values with zeroes */
                zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_reg );
                zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

                ymm0 = _mm256_loadu_si256( (__m256i const *)(b_group + ( jr * KC_updated ) / 2 ) );
                ymm1 = _mm256_loadu_si256( (__m256i const *)(b_group + ( ( jr * KC_updated ) + NR ) / 2 ) );

                CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm0, zmm14, shift_idx_64, \
                                    sign_comp, signed_upscale);

                zmm14 = _mm512_sub_epi8( zmm14, zero_point0 );

                zmm0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 1, zmm5),
                                            CVT_INT8_F32_SCAL_16( zmm14, 0, zmm4) );
                zmm1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm14, 3, zmm7),
                                            CVT_INT8_F32_SCAL_16( zmm14, 2, zmm6) );


                CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( ymm1, zmm15, shift_idx_64, \
                                    sign_comp, signed_upscale);

                zmm15 = _mm512_sub_epi8( zmm15, zero_point1 );

                zmm2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 1, zmm9),
                                            CVT_INT8_F32_SCAL_16( zmm15, 0, zmm8) );
                zmm3 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( zmm15, 3, zmm11),
                                            CVT_INT8_F32_SCAL_16( zmm15, 2, zmm10) );

                //store to pack_b buffer
                _mm512_storeu_si512( packb_group + ( jr * KC_updated ), (__m512i)zmm0 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated )+ 32, (__m512i)zmm1 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated )+ 64, (__m512i)zmm2 );
                _mm512_storeu_si512( packb_group + ( jr * KC_updated )+ 96, (__m512i)zmm3 );
            }
        }
    }
    if( n_partial_pieces > 0 )
    {
        pre_ops_attr.pre_op_b_j += n_full_pieces_loop_limit;

        // Handle NR edge cases
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
            packsclb_nr48_bf16s4f32of32
            (
              ( packb_bf16 + ( n_full_pieces_loop_limit * KC_updated ) ),
              ( b + ( n_full_pieces_loop_limit * KC_updated / 2 ) ), KC,
                signed_upscale, pre_ops_attr
            );

            n0_partial_pack = 48;
        }
        else if ( n0_32 == 1 )
        {
           packsclb_nr32_bf16s4f32of32
            (
              ( packb_bf16 + ( n_full_pieces_loop_limit * KC_updated ) ),
              ( b + ( n_full_pieces_loop_limit * KC_updated / 2 ) ), KC,
                signed_upscale, pre_ops_attr
            );

            n0_partial_pack = 32;
        }
        else if ( n0_16 == 1 )
        {
            packsclb_nr16_bf16s4f32of32
            (
              ( packb_bf16 + ( n_full_pieces_loop_limit * KC_updated ) ),
              ( b + ( n_full_pieces_loop_limit * KC_updated / 2 ) ), KC,
                signed_upscale, pre_ops_attr
            );

            n0_partial_pack = 16;
        }

        if ( n0_partial_rem > 0 )
        {
            pre_ops_attr.pre_op_b_j += n0_partial_pack;
            packsclb_nrlt16_bf16s4f32of32
            (
              ( packb_bf16 + ( n_full_pieces_loop_limit * KC_updated ) +
                ( n0_partial_pack * KC_updated ) ),
              ( b + ( ( n_full_pieces_loop_limit + n0_partial_pack ) * KC_updated / 2 )),
               KC, n0_partial_rem, signed_upscale, pre_ops_attr
            );
        }
    }

    *rs_p = NR * 2;
    *cs_p = NR / 2;
}

#endif  // LPGEMM_BF16_JIT
#endif // BLIS_ADDON_LPGEMM
