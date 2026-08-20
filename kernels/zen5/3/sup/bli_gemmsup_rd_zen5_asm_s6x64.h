/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/* Zero only accumulators for 4 B-columns (6x4 tile): zmm8-zmm31 */
#define INIT_ACCUM_4COL \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm22,zmm22,zmm22 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm25,zmm25,zmm25 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 ) \
    vxorps( zmm28,zmm28,zmm28 ) \
    vxorps( zmm29,zmm29,zmm29 ) \
    vxorps( zmm30,zmm30,zmm30 ) \
    vxorps( zmm31,zmm31,zmm31 )

/* Zero only accumulators for 3 B-columns (6x3 tile): zmm8-16, zmm20-28 */
#define INIT_ACCUM_3COL \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm22,zmm22,zmm22 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm25,zmm25,zmm25 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 ) \
    vxorps( zmm28,zmm28,zmm28 )

/* Zero only accumulators for 2 B-columns (6x2 tile): zmm8-13, zmm20-25 */
#define INIT_ACCUM_2COL \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm22,zmm22,zmm22 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm25,zmm25,zmm25 )

/* 5-row x 4-B-col accumulators: zmm 8-10,11-13,14-16,17-19,20-21,23-24,26-27,29-30 (20 regs) */
#define INIT_ACCUM_5x4 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 ) \
    vxorps( zmm29,zmm29,zmm29 ) \
    vxorps( zmm30,zmm30,zmm30 )

/* 4-row x 4-B-col accumulators: zmm 8-10,11-13,14-16,17-19,20,23,26,29 (16 regs) */
#define INIT_ACCUM_4x4 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm29,zmm29,zmm29 )

/* 3-row x 4-B-col accumulators: zmm 8-10,11-13,14-16,17-19 (12 regs) */
#define INIT_ACCUM_3x4 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 )

/* 2-row x 4-B-col accumulators: zmm 8-9,11-12,14-15,17-18 (8 regs) */
#define INIT_ACCUM_2x4 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 )

/* 1-row x 4-B-col accumulators: zmm 8,11,14,17 (4 regs) */
#define INIT_ACCUM_1x4 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm17,zmm17,zmm17 )

/* 5-row x 3-B-col accumulators: zmm 8-10,11-13,14-16,20-21,23-24,26-27 (15 regs) */
#define INIT_ACCUM_5x3 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 )

/* 4-row x 3-B-col accumulators: zmm 8-10,11-13,14-16,20,23,26 (12 regs) */
#define INIT_ACCUM_4x3 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm26,zmm26,zmm26 )

/* 3-row x 3-B-col accumulators: zmm 8-10,11-13,14-16 (9 regs) */
#define INIT_ACCUM_3x3 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 )

/* 2-row x 3-B-col accumulators: zmm 8-9,11-12,14-15 (6 regs) */
#define INIT_ACCUM_2x3 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 )

/* 1-row x 3-B-col accumulators: zmm 8,11,14 (3 regs) */
#define INIT_ACCUM_1x3 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm14,zmm14,zmm14 )

/* 5-row x 2-B-col accumulators: zmm 8-10,11-13,20-21,23-24 (10 regs) */
#define INIT_ACCUM_5x2 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 )

/* 4-row x 2-B-col accumulators: zmm 8-10,11-13,20,23 (8 regs) */
#define INIT_ACCUM_4x2 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm23,zmm23,zmm23 )

/* 3-row x 2-B-col accumulators: zmm 8-10,11-13 (6 regs) */
#define INIT_ACCUM_3x2 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 )

/* 2-row x 2-B-col accumulators: zmm 8-9,11-12 (4 regs) */
#define INIT_ACCUM_2x2 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 )

/* 1-row x 2-B-col accumulators: zmm 8,11 (2 regs) */
#define INIT_ACCUM_1x2 \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm11,zmm11,zmm11 )

#define VFMA6( Rload, R0, R1, R2, R3, R4, R5 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */ \
    vfmadd231ps( zmm1, zmm(Rload), zmm(R1) )     /* zmm(R1) += zmm1 * zmm(Rload) */ \
    vfmadd231ps( zmm2, zmm(Rload), zmm(R2) )     /* zmm(R2) += zmm2 * zmm(Rload) */ \
    vfmadd231ps( zmm3, zmm(Rload), zmm(R3) )     /* zmm(R3) += zmm3 * zmm(Rload) */ \
    vfmadd231ps( zmm4, zmm(Rload), zmm(R4) )     /* zmm(R4) += zmm4 * zmm(Rload) */ \
    vfmadd231ps( zmm5, zmm(Rload), zmm(R5) )     /* zmm(R5) += zmm5 * zmm(Rload) */

#define VFMA5( Rload, R0, R1, R2, R3, R4 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */ \
    vfmadd231ps( zmm1, zmm(Rload), zmm(R1) )     /* zmm(R1) += zmm1 * zmm(Rload) */ \
    vfmadd231ps( zmm2, zmm(Rload), zmm(R2) )     /* zmm(R2) += zmm2 * zmm(Rload) */ \
    vfmadd231ps( zmm3, zmm(Rload), zmm(R3) )     /* zmm(R3) += zmm3 * zmm(Rload) */ \
    vfmadd231ps( zmm4, zmm(Rload), zmm(R4) )     /* zmm(R4) += zmm4 * zmm(Rload) */

#define VFMA4( Rload, R0, R1, R2, R3 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */ \
    vfmadd231ps( zmm1, zmm(Rload), zmm(R1) )     /* zmm(R1) += zmm1 * zmm(Rload) */ \
    vfmadd231ps( zmm2, zmm(Rload), zmm(R2) )     /* zmm(R2) += zmm2 * zmm(Rload) */ \
    vfmadd231ps( zmm3, zmm(Rload), zmm(R3) )     /* zmm(R3) += zmm3 * zmm(Rload) */

#define VFMA3( Rload, R0, R1, R2 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */ \
    vfmadd231ps( zmm1, zmm(Rload), zmm(R1) )     /* zmm(R1) += zmm1 * zmm(Rload) */ \
    vfmadd231ps( zmm2, zmm(Rload), zmm(R2) )     /* zmm(R2) += zmm2 * zmm(Rload) */

#define VFMA2( Rload, R0, R1 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */ \
    vfmadd231ps( zmm1, zmm(Rload), zmm(R1) )     /* zmm(R1) += zmm1 * zmm(Rload) */

#define VFMA1( Rload, R0 ) \
    vfmadd231ps( zmm0, zmm(Rload), zmm(R0) )     /* zmm(R0) += zmm0 * zmm(Rload) */

/* zmm(R0)=[a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] */
/* zmm(R1)=[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15] */
/* zmm(R2)=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15] */
/* zmm(R3)=[d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15] */
/* Input: four 16-float accumulators in zmm(R0) through zmm(R3). */
/* Output: four final sums packed as [sum(a),sum(b),sum(c),sum(d)] in xmm(RD). */
#define ZMM_REDUCE_4(R0, R1, R2, R3, RD) \
    vshufps( imm(0x88), zmm(R1), zmm(R0), zmm0 )   /* zmm0=[a0,a2,b0,b2,a4,a6,b4,b6, */ \
                                                   /*       a8,a10,b8,b10,a12,a14,b12,b14] */ \
    vshufps( imm(0xDD), zmm(R1), zmm(R0), zmm1 )   /* zmm1=[a1,a3,b1,b3,a5,a7,b5,b7, */ \
                                                   /*       a9,a11,b9,b11,a13,a15,b13,b15] */ \
    vaddps( zmm1, zmm0, zmm0 )                     /* zmm0=[a0+a1,a2+a3,b0+b1,b2+b3, */ \
                                                   /*       a4+a5,a6+a7,b4+b5,b6+b7, */ \
                                                   /*       a8+a9,a10+a11,b8+b9,b10+b11, */ \
                                                   /*       a12+a13,a14+a15,b12+b13,b14+b15] */ \
    vshufps( imm(0x88), zmm(R3), zmm(R2), zmm2 )   /* zmm2=[c0,c2,d0,d2,c4,c6,d4,d6, */ \
                                                   /*       c8,c10,d8,d10,c12,c14,d12,d14] */ \
    vshufps( imm(0xDD), zmm(R3), zmm(R2), zmm3 )   /* zmm3=[c1,c3,d1,d3,c5,c7,d5,d7, */ \
                                                   /*       c9,c11,d9,d11,c13,c15,d13,d15] */ \
    vaddps( zmm3, zmm2, zmm2 )                     /* zmm2=[c0+c1,c2+c3,d0+d1,d2+d3, */ \
                                                   /*       c4+c5,c6+c7,d4+d5,d6+d7, */ \
                                                   /*       c8+c9,c10+c11,d8+d9,d10+d11, */ \
                                                   /*       c12+c13,c14+c15,d12+d13,d14+d15] */ \
    vshufps( imm(0x88), zmm2, zmm0, zmm1 )         /* zmm1=[a0+a1,b0+b1,c0+c1,d0+d1, */ \
                                                   /*       a4+a5,b4+b5,c4+c5,d4+d5, */ \
                                                   /*       a8+a9,b8+b9,c8+c9,d8+d9, */ \
                                                   /*       a12+a13,b12+b13,c12+c13,d12+d13] */ \
    vshufps( imm(0xDD), zmm2, zmm0, zmm3 )         /* zmm3=[a2+a3,b2+b3,c2+c3,d2+d3, */ \
                                                   /*       a6+a7,b6+b7,c6+c7,d6+d7, */ \
                                                   /*       a10+a11,b10+b11,c10+c11,d10+d11, */ \
                                                   /*       a14+a15,b14+b15,c14+c15,d14+d15] */ \
    vaddps( zmm3, zmm1, zmm1 )                     /* zmm1=[a0+...+a3,b0+...+b3,c0+...+c3,d0+...+d3, */ \
                                                   /*       a4+...+a7,b4+...+b7,c4+...+c7,d4+...+d7, */ \
                                                   /*       a8+...+a11,b8+...+b11,c8+...+c11,d8+...+d11, */ \
                                                   /*       a12+...+a15,b12+...+b15,c12+...+c15,d12+...+d15] */ \
                                                   /* zmm1 now holds four quarter-sums for each output row. */ \
    vextractf32x8( imm(0x01), zmm1, ymm0 )         /* ymm0=[a8+...+a11,b8+...+b11,c8+...+c11,d8+...+d11, */ \
                                                   /*       a12+...+a15,b12+...+b15,c12+...+c15,d12+...+d15] */ \
    vaddps( ymm1, ymm0, ymm0 )                     /* ymm0=[a0+...+a3+a8+...+a11,b0+...+b3+b8+...+b11, */ \
                                                   /*       c0+...+c3+c8+...+c11,d0+...+d3+d8+...+d11, */ \
                                                   /*       a4+...+a7+a12+...+a15,b4+...+b7+b12+...+b15, */ \
                                                   /*       c4+...+c7+c12+...+c15,d4+...+d7+d12+...+d15] */ \
    vextractf128( imm(0x01), ymm0, xmm1 )          /* xmm1=[a4+...+a7+a12+...+a15,b4+...+b7+b12+...+b15, */ \
                                                   /*       c4+...+c7+c12+...+c15,d4+...+d7+d12+...+d15] */ \
    vaddps( xmm0, xmm1, xmm(RD) )                  /* xmm(RD)=[sum(a),sum(b),sum(c),sum(d)] */

/* Input: three 16-float accumulators in zmm(R0), zmm(R1), and zmm(R2). */
/* Output: [sum(a),sum(b),sum(c),sum(c)] in xmm(RD). */
#define ZMM_REDUCE_3(R0, R1, R2, RD) \
    vshufps( imm(0x88), zmm(R1), zmm(R0), zmm0 )   /* zmm0=[a0,a2,b0,b2,a4,a6,b4,b6, */ \
                                                   /*       a8,a10,b8,b10,a12,a14,b12,b14] */ \
    vshufps( imm(0xDD), zmm(R1), zmm(R0), zmm1 )   /* zmm1=[a1,a3,b1,b3,a5,a7,b5,b7, */ \
                                                   /*       a9,a11,b9,b11,a13,a15,b13,b15] */ \
    vaddps( zmm1, zmm0, zmm0 )                     /* zmm0=[a0+a1,a2+a3,b0+b1,b2+b3, */ \
                                                   /*       a4+a5,a6+a7,b4+b5,b6+b7, */ \
                                                   /*       a8+a9,a10+a11,b8+b9,b10+b11, */ \
                                                   /*       a12+a13,a14+a15,b12+b13,b14+b15] */ \
    vshufps( imm(0x88), zmm(R2), zmm(R2), zmm2 )   /* zmm2=[c0,c2,c0,c2,c4,c6,c4,c6, */ \
                                                   /*       c8,c10,c8,c10,c12,c14,c12,c14] */ \
    vshufps( imm(0xDD), zmm(R2), zmm(R2), zmm3 )   /* zmm3=[c1,c3,c1,c3,c5,c7,c5,c7, */ \
                                                   /*       c9,c11,c9,c11,c13,c15,c13,c15] */ \
    vaddps( zmm3, zmm2, zmm2 )                     /* zmm2=[c0+c1,c2+c3,c0+c1,c2+c3, */ \
                                                   /*       c4+c5,c6+c7,c4+c5,c6+c7, */ \
                                                   /*       c8+c9,c10+c11,c8+c9,c10+c11, */ \
                                                   /*       c12+c13,c14+c15,c12+c13,c14+c15] */ \
    vshufps( imm(0x88), zmm2, zmm0, zmm1 )         /* zmm1=[a0+a1,b0+b1,c0+c1,c0+c1, */ \
                                                   /*       a4+a5,b4+b5,c4+c5,c4+c5, */ \
                                                   /*       a8+a9,b8+b9,c8+c9,c8+c9, */ \
                                                   /*       a12+a13,b12+b13,c12+c13,c12+c13] */ \
    vshufps( imm(0xDD), zmm2, zmm0, zmm3 )         /* zmm3=[a2+a3,b2+b3,c2+c3,c2+c3, */ \
                                                   /*       a6+a7,b6+b7,c6+c7,c6+c7, */ \
                                                   /*       a10+a11,b10+b11,c10+c11,c10+c11, */ \
                                                   /*       a14+a15,b14+b15,c14+c15,c14+c15] */ \
    vaddps( zmm3, zmm1, zmm1 )                     /* zmm1=[a0+...+a3,b0+...+b3,c0+...+c3,c0+...+c3, */ \
                                                   /*       a4+...+a7,b4+...+b7,c4+...+c7,c4+...+c7, */ \
                                                   /*       a8+...+a11,b8+...+b11,c8+...+c11,c8+...+c11, */ \
                                                   /*       a12+...+a15,b12+...+b15,c12+...+c15,c12+...+c15] */ \
    vextractf32x8( imm(0x01), zmm1, ymm0 )         /* ymm0=[a8+...+a11,b8+...+b11,c8+...+c11,c8+...+c11, */ \
                                                   /*       a12+...+a15,b12+...+b15,c12+...+c15,c12+...+c15] */ \
    vaddps( ymm1, ymm0, ymm0 )                     /* ymm0=[a0+...+a3+a8+...+a11,b0+...+b3+b8+...+b11, */ \
                                                   /*       c0+...+c3+c8+...+c11,c0+...+c3+c8+...+c11, */ \
                                                   /*       a4+...+a7+a12+...+a15,b4+...+b7+b12+...+b15, */ \
                                                   /*       c4+...+c7+c12+...+c15,c4+...+c7+c12+...+c15] */ \
    vextractf128( imm(0x01), ymm0, xmm1 )          /* xmm1=[a4+...+a7+a12+...+a15,b4+...+b7+b12+...+b15, */ \
                                                   /*       c4+...+c7+c12+...+c15,c4+...+c7+c12+...+c15] */ \
    vaddps( xmm0, xmm1, xmm(RD) )                  /* xmm(RD)=[sum(a),sum(b),sum(c),sum(c)] */

/* Input: two 16-float accumulators in zmm(R0) and zmm(R1). */
/* Output: [sum(a),sum(b),sum(a),sum(b)] in xmm(RD). */
#define ZMM_REDUCE_2(R0, R1, RD) \
    vshufps( imm(0x88), zmm(R1), zmm(R0), zmm0 )   /* zmm0=[a0,a2,b0,b2,a4,a6,b4,b6, */ \
                                                   /*       a8,a10,b8,b10,a12,a14,b12,b14] */ \
    vshufps( imm(0xDD), zmm(R1), zmm(R0), zmm1 )   /* zmm1=[a1,a3,b1,b3,a5,a7,b5,b7, */ \
                                                   /*       a9,a11,b9,b11,a13,a15,b13,b15] */ \
    vaddps( zmm1, zmm0, zmm0 )                     /* zmm0=[a0+a1,a2+a3,b0+b1,b2+b3, */ \
                                                   /*       a4+a5,a6+a7,b4+b5,b6+b7, */ \
                                                   /*       a8+a9,a10+a11,b8+b9,b10+b11, */ \
                                                   /*       a12+a13,a14+a15,b12+b13,b14+b15] */ \
    vextractf32x8( imm(0x01), zmm0, ymm1 )         /* ymm1=[a8+a9,a10+a11,b8+b9,b10+b11, */ \
                                                   /*       a12+a13,a14+a15,b12+b13,b14+b15] */ \
    vaddps( ymm0, ymm1, ymm0 )                     /* ymm0=[a0+a1+a8+a9,a2+a3+a10+a11, */ \
                                                   /*       b0+b1+b8+b9,b2+b3+b10+b11, */ \
                                                   /*       a4+a5+a12+a13,a6+a7+a14+a15, */ \
                                                   /*       b4+b5+b12+b13,b6+b7+b14+b15] */ \
    vextractf128( imm(0x01), ymm0, xmm1 )          /* xmm1=[a4+a5+a12+a13,a6+a7+a14+a15, */ \
                                                   /*       b4+b5+b12+b13,b6+b7+b14+b15] */ \
    vaddps( xmm0, xmm1, xmm0 )                     /* xmm0=[a0+a1+a4+a5+a8+a9+a12+a13, */ \
                                                   /*       a2+a3+a6+a7+a10+a11+a14+a15, */ \
                                                   /*       b0+b1+b4+b5+b8+b9+b12+b13, */ \
                                                   /*       b2+b3+b6+b7+b10+b11+b14+b15] */ \
    vshufps( imm(0x88), xmm0, xmm0, xmm(RD) )      /* xmm(RD)=[a0+a1+a4+a5+a8+a9+a12+a13, */ \
                                                   /*         b0+b1+b4+b5+b8+b9+b12+b13, */ \
                                                   /*         a0+a1+a4+a5+a8+a9+a12+a13, */ \
                                                   /*         b0+b1+b4+b5+b8+b9+b12+b13] */ \
    vshufps( imm(0xDD), xmm0, xmm0, xmm3 )         /* xmm3=[a2+a3+a6+a7+a10+a11+a14+a15, */ \
                                                   /*       b2+b3+b6+b7+b10+b11+b14+b15, */ \
                                                   /*       a2+a3+a6+a7+a10+a11+a14+a15, */ \
                                                   /*       b2+b3+b6+b7+b10+b11+b14+b15] */ \
    vaddps( xmm3, xmm(RD), xmm(RD) )               /* xmm(RD)=[sum(a),sum(b),sum(a),sum(b)] */ 

#define C_STOR_BZ_2_FLOATS(R_rs_c, R0, R1, R2) \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R2), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R2)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_2_FLOATS1(R_rs_c, R0) \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ 

#define C_STOR_BZ_2_FLOATS2(R_rs_c, R0, R1) \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_MASKED(R_rs_c, R0, R1, R2) \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R2), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_MASKED2(R_rs_c, R0, R1) \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ 

#define C_STOR_BZ_MASKED1(R_rs_c, R0) \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ 

#define C_STOR_BZ(R_rs_c, R0, R1, R2) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem( rcx ) )               /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R2), mem( rcx ) )               /* mem(rcx) <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ2(R_rs_c, R0, R1) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem( rcx ) )               /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ1(R_rs_c, R0) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_CONT(R_rs_c, R0, R1, R2) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem( rcx ) )               /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R2), mem( rcx ) )               /* mem(rcx) <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ2_CONT(R_rs_c, R0, R1) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem( rcx ) )               /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ1_CONT(R_rs_c, R0) \
    vmovups( xmm(R0), mem( rcx ) )               /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_MASKED_CONT(R_rs_c, R0, R1, R2) \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R2), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_MASKED2_CONT(R_rs_c, R0, R1) \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_2_FLOATS_CONT(R_rs_c, R0, R1, R2) \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R2), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R2)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_BZ_2_FLOATS2_CONT(R_rs_c, R0, R1) \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define ALPHA_SCALE(Ralpha, R0, R1, R2) \
    vmulps( xmm(Ralpha), xmm(R0), xmm(R0) )              /* xmm(R0) = alpha * xmm(R0) */ \
    vmulps( xmm(Ralpha), xmm(R1), xmm(R1) )              /* xmm(R1) = alpha * xmm(R1) */ \
    vmulps( xmm(Ralpha), xmm(R2), xmm(R2) )              /* xmm(R2) = alpha * xmm(R2) */

#define C_STOR(R_rs_c, Rbeta, R0, R1, R2) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R1) )    /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), (rcx) )                    /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R2) )    /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovups( xmm(R2), (rcx) )                    /* mem(rcx) <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_CONT(R_rs_c, Rbeta, R0, R1, R2) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R1) )    /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), (rcx) )                    /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R2) )    /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovups( xmm(R2), (rcx) )                    /* mem(rcx) <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_MASKED(R_rs_c, Rbeta, R0, R1, R2) \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R2) )     /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovups( xmm(R2), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_MASKED_CONT(R_rs_c, Rbeta, R0, R1, R2) \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R2) )     /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovups( xmm(R2), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R2) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_2_FLOATS(R_rs_c, Rbeta, R0, R1, R2) \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R2) )     /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovlps( xmm(R2), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R2)[0:1] */ \
    add( R_rs_c, rcx )                              /* rcx += rs_c * sizeof(float) */

#define C_STOR_2_FLOATS_CONT(R_rs_c, Rbeta, R0, R1, R2) \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R2) )     /* xmm(R2) = beta * C + xmm(R2) */ \
    vmovlps( xmm(R2), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R2)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define ALPHA_SCALE2(Ralpha, R0, R1) \
    vmulps( xmm(Ralpha), xmm(R0), xmm(R0) )              /* xmm(R0) = alpha * xmm(R0) */ \
    vmulps( xmm(Ralpha), xmm(R1), xmm(R1) )              /* xmm(R1) = alpha * xmm(R1) */

#define ALPHA_SCALE1(Ralpha, R0) \
    vmulps( xmm(Ralpha), xmm(R0), xmm(R0) )              /* xmm(R0) = alpha * xmm(R0) */

#define C_STOR2_CONT(R_rs_c, Rbeta, R0, R1) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R1) )    /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), (rcx) )                    /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR1_CONT(R_rs_c, Rbeta, R0) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_MASKED2(R_rs_c, Rbeta, R0, R1) \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_MASKED2_CONT(R_rs_c, Rbeta, R0, R1) \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_2_FLOATS2(R_rs_c, Rbeta, R0, R1) \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_2_FLOATS2_CONT(R_rs_c, Rbeta, R0, R1) \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vmovsd( mem(rcx), xmm1 )                     /* xmm1 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovlps( xmm(R1), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R1)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR2(R_rs_c, Rbeta, R0, R1) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */ \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R1) )    /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), (rcx) )                    /* mem(rcx) <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR1(R_rs_c, Rbeta, R0) \
    vfmadd231ps( (rcx), xmm(Rbeta), xmm(R0) )    /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovups( xmm(R0), (rcx) )                    /* mem(rcx) <- xmm(R0) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_MASKED1(R_rs_c, Rbeta, R1) \
    vmovups( mem(rcx), xmm(1 MASK_KZ(2)) )       /* xmm1 <- mem(rcx){k2,z} */ \
    vfmadd231ps( xmm(Rbeta), xmm1, xmm(R1) )     /* xmm(R1) = beta * C + xmm(R1) */ \
    vmovups( xmm(R1), mem(rcx) MASK_K(2) )       /* mem(rcx){k2} <- xmm(R1) */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */

#define C_STOR_2_FLOATS1(R_rs_c, Rbeta, R0) \
    vmovsd( mem(rcx), xmm0 )                     /* xmm0 <- mem(rcx)[0:1] */ \
    vfmadd231ps( xmm(Rbeta), xmm0, xmm(R0) )     /* xmm(R0) = beta * C + xmm(R0) */ \
    vmovlps( xmm(R0), mem(rcx) )                 /* mem(rcx)[0:1] <- xmm(R0)[0:1] */ \
    add( R_rs_c, rcx )                           /* rcx += rs_c * sizeof(float) */
