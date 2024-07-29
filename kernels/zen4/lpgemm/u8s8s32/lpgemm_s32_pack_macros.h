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

#include "../int4_utils_avx512.h"

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
