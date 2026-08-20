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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
#define PREFETCH_DIST_C 4
#define MR 24
#define NR 4

/* Macro to reset the registers for accumulation */
#define RESET_REGISTERS \
    VXORPS(ZMM(0), ZMM(0), ZMM(0)) \
    VXORPS(ZMM(1), ZMM(1), ZMM(1)) \
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) \
    VXORPS(ZMM(3), ZMM(3), ZMM(3)) \
    VXORPS(ZMM(4), ZMM(4), ZMM(4)) \
    VXORPS(ZMM(5), ZMM(5), ZMM(5)) \
    VXORPS(ZMM(6), ZMM(6), ZMM(6)) \
    VXORPS(ZMM(7), ZMM(7), ZMM(7)) \
    VXORPS(ZMM(8), ZMM(8), ZMM(8)) \
    VXORPS(ZMM(9), ZMM(9), ZMM(9)) \
    VXORPS(ZMM(10), ZMM(10), ZMM(10)) \
    VXORPS(ZMM(11), ZMM(11), ZMM(11)) \
    VXORPS(ZMM(12), ZMM(12), ZMM(12)) \
    VXORPS(ZMM(13), ZMM(13), ZMM(13)) \
    VXORPS(ZMM(14), ZMM(14), ZMM(14)) \
    VXORPS(ZMM(15), ZMM(15), ZMM(15)) \
    VXORPS(ZMM(16), ZMM(16), ZMM(16)) \
    VXORPS(ZMM(17), ZMM(17), ZMM(17)) \
    VXORPS(ZMM(18), ZMM(18), ZMM(18)) \
    VXORPS(ZMM(19), ZMM(19), ZMM(19)) \
    VXORPS(ZMM(20), ZMM(20), ZMM(20)) \
    VXORPS(ZMM(21), ZMM(21), ZMM(21)) \
    VXORPS(ZMM(22), ZMM(22), ZMM(22)) \
    VXORPS(ZMM(23), ZMM(23), ZMM(23)) \
    VXORPS(ZMM(24), ZMM(24), ZMM(24)) \
    VXORPS(ZMM(25), ZMM(25), ZMM(25)) \
    VXORPS(ZMM(26), ZMM(26), ZMM(26)) \
    VXORPS(ZMM(27), ZMM(27), ZMM(27)) \
    VXORPS(ZMM(28), ZMM(28), ZMM(28)) \
    VXORPS(ZMM(30), ZMM(30), ZMM(30)) \
    VXORPS(ZMM(31), ZMM(31), ZMM(31)) \

/* Macro to permute in case of 3 loads(24x? cases) */
#define PERMUTE_24C(R1, R2, R3) \
    VPERMILPS(IMM(0xB1), ZMM(R1), ZMM(R1)) \
    VPERMILPS(IMM(0xB1), ZMM(R2), ZMM(R2)) \
    VPERMILPS(IMM(0xB1), ZMM(R3), ZMM(R3)) \

/* Macro to permute in case of 2 loads(16x? cases) */
#define PERMUTE_16C(R1, R2) \
    VPERMILPS(IMM(0xB1), ZMM(R1), ZMM(R1)) \
    VPERMILPS(IMM(0xB1), ZMM(R2), ZMM(R2)) \

/* Macro to permute in case of 1 loads(16x? cases) */
#define PERMUTE_8C(R1) \
    VPERMILPS(IMM(0xB1), ZMM(R1), ZMM(R1)) \

/* Macro to get the PERMUTE_? signature from the list */
#define GET_PERMUTE(_1, _2, _3, NAME, ...)  NAME

/* Overloaded macro PERMUTE with variable arguments */
#define PERMUTE(...)\
    GET_PERMUTE(__VA_ARGS__, \
    PERMUTE_24C, PERMUTE_16C, PERMUTE_8C)(__VA_ARGS__) \

/* Macro for fma op in case of 3 loads(24x? cases) */
#define FMA_24C(B, R1, R2, R3) \
    VFMADD231PS(ZMM(0), ZMM(B), ZMM(R1)) \
    VFMADD231PS(ZMM(1), ZMM(B), ZMM(R2)) \
    VFMADD231PS(ZMM(2), ZMM(B), ZMM(R3)) \

/* Macro for fma op in case of 2 loads(16x? cases) */
#define FMA_16C(B, R1, R2) \
    VFMADD231PS(ZMM(0), ZMM(B), ZMM(R1)) \
    VFMADD231PS(ZMM(1), ZMM(B), ZMM(R2)) \

/* Macro for fma op in case of 1 load(8x? cases) */
#define FMA_8C(B, R1) \
    VFMADD231PS(ZMM(0), ZMM(B), ZMM(R1)) \

/* Macro to get the FMA_? signature from the list */
#define GET_FMA(_1, _2, _3, _4, NAME, ...)  NAME

/* Overloaded macro FMA with variable arguments */
#define FMA(...) \
    GET_FMA(__VA_ARGS__, \
    FMA_24C, FMA_16C, FMA_8C)(__VA_ARGS__) \

/* Macro for accumalation in case of 3 loads(24x? cases) */
#define ACC_COL_24C(R1, I1, R2, I2, R3, I3) \
    VFMADDSUB231PS(ZMM(R1), ZMM(29), ZMM(I1)) \
    VFMADDSUB231PS(ZMM(R2), ZMM(29), ZMM(I2)) \
    VFMADDSUB231PS(ZMM(R3), ZMM(29), ZMM(I3)) \

/* Macro for accumalation in case of 2 loads(16x? cases) */
#define ACC_COL_16C(R1, I1, R2, I2) \
    VFMADDSUB231PS(ZMM(R1), ZMM(29), ZMM(I1)) \
    VFMADDSUB231PS(ZMM(R2), ZMM(29), ZMM(I2)) \

/* Macro for accumalation in case of 1 load(8x? cases) */
#define ACC_COL_8C(R1, I1) \
    VFMADDSUB231PS(ZMM(R1), ZMM(29), ZMM(I1)) \

/* Macro to get the ACC_COL_? signature from the list */
#define GET_ACC_COL(_1, _2, _3, _4, _5, _6, NAME, ...)  NAME

/* Overloaded macro ACC_COL with variable arguments */
#define ACC_COL(...) \
    GET_ACC_COL(__VA_ARGS__, \
    ACC_COL_24C, _0, ACC_COL_16C, _1, ACC_COL_8C)(__VA_ARGS__) \

/* Macro for scaling with alpha if it is complex
   in case of 3 loads(24x? cases) */
#define ALPHA_GENERIC_24C(R1, R2, R3) \
    VMULPS(ZMM(0), ZMM(R1), ZMM(2)) \
    VMULPS(ZMM(1), ZMM(R1), ZMM(R1)) \
    VMULPS(ZMM(0), ZMM(R2), ZMM(30)) \
    VMULPS(ZMM(1), ZMM(R2), ZMM(R2)) \
    VMULPS(ZMM(0), ZMM(R3), ZMM(31)) \
    VMULPS(ZMM(1), ZMM(R3), ZMM(R3)) \
    PERMUTE(R1, R2, R3) \
    ACC_COL(2, R1, 30, R2, 31, R3) \

/* Macro for scaling with alpha if it is complex
   in case of 2 loads(16x? cases) */
#define ALPHA_GENERIC_16C(R1, R2) \
    VMULPS(ZMM(0), ZMM(R1), ZMM(2)) \
    VMULPS(ZMM(1), ZMM(R1), ZMM(R1)) \
    VMULPS(ZMM(0), ZMM(R2), ZMM(30)) \
    VMULPS(ZMM(1), ZMM(R2), ZMM(R2)) \
    PERMUTE(R1, R2) \
    ACC_COL(2, R1, 30, R2) \

/* Macro for scaling with alpha if it is complex
   in case of 1 load(8x? cases) */
#define ALPHA_GENERIC_8C(R1) \
    VMULPS(ZMM(0), ZMM(R1), ZMM(2)) \
    VMULPS(ZMM(1), ZMM(R1), ZMM(R1)) \
    PERMUTE(R1) \
    ACC_COL(2, R1) \

/* Macro to get the ALPHA_GENERIC_? signature from the list */
#define GET_ALPHA_GENERIC(_1, _2, _3, NAME, ...)  NAME

/* Overloaded macro ALPHA_GENERIC with variable arguments */
#define ALPHA_GENERIC(...) \
    GET_ALPHA_GENERIC(__VA_ARGS__, \
    ALPHA_GENERIC_24C, ALPHA_GENERIC_16C, ALPHA_GENERIC_8C)(__VA_ARGS__) \

/* Macro for scaling with beta if it is complex
   in case of 3 loads(24x? cases) */
#define BETA_GENERIC_24C(C, R1, I1, R2, I2, R3, I3)\
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3)) \
\
    ALPHA_GENERIC(R1, R2, R3) \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VADDPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128)) \

/* Macro for scaling with beta if it is complex
   in case of 2 loads(16x? cases) */
#define BETA_GENERIC_16C(C, R1, I1, R2, I2)\
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
\
    ALPHA_GENERIC(R1, R2) \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \

/* Macro for scaling with beta if it is complex
   in case of 1 load(8x? cases) */
#define BETA_GENERIC_8C(C, R1, I1)\
    VMOVUPS(MEM(C), ZMM(R1)) \
\
    ALPHA_GENERIC(R1) \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \

/* Macro to get the BETA_GENERIC_? signature from the list */
#define GET_BETA_GENERIC(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

/* Overloaded macro BETA_GENERIC with variable arguments */
#define BETA_GENERIC(...) \
    GET_BETA_GENERIC(__VA_ARGS__, \
    BETA_GENERIC_24C, _0, BETA_GENERIC_16C, _1, BETA_GENERIC_8C)(__VA_ARGS__) \

/* Macro for scaling with beta if it is complex
   in case of 1 load(fx? cases, f<8) */
#define BETA_GENERIC_fC(C, R1, I1)\
   VMOVUPS(MEM(C), ZMM(R1) MASK_(k(2))) \
\
   ALPHA_GENERIC(R1) \
   VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
\
   VMOVUPS(ZMM(I1), MEM(C) MASK_(k(2))) \

/* Macro to perform a 24x4 micro-tile computation */
#define MICRO_TILE_24x4 \
    /* Macro for 24x4 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13, 15) \
    FMA(31, 12, 14, 16) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(30, 23, 25, 27) \
    FMA(31, 24, 26, 28) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x4 micro-tile computation */
#define MICRO_TILE_16x4 \
    /* Macro for 16x4 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(1) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13) \
    FMA(31, 12, 14) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(30, 23, 25) \
    FMA(31, 24, 26) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x4 micro-tile computation */
#define MICRO_TILE_8x4 \
    /* Macro for 8x4 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 17) \
    FMA(4, 18) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 23) \
    FMA(31, 24) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx4 micro-tile computation(f<8) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx4 \
    /* Macro for fx4 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 17) \
    FMA(4, 18) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 23) \
    FMA(31, 24) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x3 micro-tile computation */
#define MICRO_TILE_24x3 \
    /* Macro for 24x3 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13, 15) \
    FMA(31, 12, 14, 16) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x3 micro-tile computation */
#define MICRO_TILE_16x3 \
    /* Macro for 16x3 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(1) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13) \
    FMA(31, 12, 14) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x3 micro-tile computation */
#define MICRO_TILE_8x3 \
    /* Macro for 8x3 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 17) \
    FMA(4, 18) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx3 micro-tile computation(f<8) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx3 \
    /* Macro for fx3 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 17) \
    FMA(4, 18) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \


/* Macro to perform a 24x2 micro-tile computation */
#define MICRO_TILE_24x2 \
    /* Macro for 24x2 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13, 15) \
    FMA(31, 12, 14, 16) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x2 micro-tile computation */
#define MICRO_TILE_16x2 \
    /* Macro for 16x2 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(1) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(30, 11, 13) \
    FMA(31, 12, 14) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x2 micro-tile computation */
#define MICRO_TILE_8x2 \
    /* Macro for 8x2 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx2 micro-tile computation(f<8) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx2 \
    /* Macro for fx2 micro-tile evaluation   */ \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(30, 11) \
    FMA(31, 12) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x1 micro-tile computation */
#define MICRO_TILE_24x1 \
    /* Macro for 24x1 micro-tile evaluation   */ \
    /* Broadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x1 micro-tile computation */
#define MICRO_TILE_16x1 \
    /* Macro for 16x1 micro-tile evaluation   */ \
    /* Broadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(1) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    /* 4 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x1 micro-tile computation */
#define MICRO_TILE_8x1 \
    /* Macro for 8x1 micro-tile evaluation   */ \
    /* Broadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx1 micro-tile computation(f<8) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx1 \
    /* Macro for fx1 micro-tile evaluation   */ \
    /* Broadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) */ \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    /* 2 FMAs over 2 broadcasts */ \
    FMA(3, 5) \
    FMA(4, 6) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* ===================================================================
 * Conjugation-aware MICRO_TILE variants.
 *
 * Mirror the encoding used by the canonical z12x4 zgemm family
 * (kernels/zen4/3/sup/bli_gemmsup_cv_zen4_asm_z12x4m.c):
 *
 *   CONJA           : ZMM(30) holds conja_arr = {+1,-1,+1,-1,...}.
 *                     Each loaded A panel is in-place multiplied by
 *                     ZMM(30) which flips the sign of the imaginary
 *                     halves of every interleaved (Re,Im) pair.
 *                     All B broadcasts go to ZMM(3)/ZMM(4) (since
 *                     ZMM(30) is now reserved for conja_arr).
 *
 *   CONJB           : ZMM(30) holds conjb_arr = {-1,-1,-1,-1,...}.
 *                     After each (B.re, B.im) broadcast pair the B.im
 *                     broadcast (in ZMM(4)) is multiplied by ZMM(30)
 *                     which negates the broadcasted imaginary value.
 *                     All B broadcasts go to ZMM(3)/ZMM(4).
 *
 *   CONJA_CONJB     : ZMM(30) = conja_arr, ZMM(31) = conjb_arr.
 *                     A is pre-multiplied by ZMM(30); the B.im
 *                     broadcast is pre-multiplied by ZMM(31).
 *                     All B broadcasts go to ZMM(3)/ZMM(4).
 * =================================================================== */

/* Macro to perform a 24x4 micro-tile computation (CONJA) */
#define MICRO_TILE_24x4_CONJA \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    /* Conjugate A: flip imag halves of each (Re,Im) pair */ \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(2), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x4 micro-tile computation (CONJB) */
#define MICRO_TILE_24x4_CONJB \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    /* Conjugate B: negate B.imag broadcast */ \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x4 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_24x4_CONJA_CONJB \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    /* Loading A using ZMM(0) - ZMM(2) */ \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    VMULPS(ZMM(2), ZMM(30), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */ \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    /* 6 FMAs over 2 broadcasts */ \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    /* Adjusting addresses for next micro tiles */ \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x4 micro-tile computation (CONJA) */
#define MICRO_TILE_16x4_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x4 micro-tile computation (CONJB) */
#define MICRO_TILE_16x4_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x4 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_16x4_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x4 micro-tile computation (CONJA) */
#define MICRO_TILE_8x4_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x4 micro-tile computation (CONJB) */
#define MICRO_TILE_8x4_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x4 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_8x4_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx4 micro-tile computation (CONJA, f<8) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx4_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx4 micro-tile computation (CONJB, f<8) */
#define MICRO_TILE_fx4_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx4 micro-tile computation (CONJA_CONJB, f<8) */
#define MICRO_TILE_fx4_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 23) \
    FMA(4, 24) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* ===================================================================
 * Masked MICRO_TILE variants for the 24x4 and 16x4 edge bodies that
 * cover m_left fringes inside CGEMM_24x4_MAIN_BODY (Stages C and D).
 *
 * The shape matches the corresponding non-masked variant exactly,
 * except the TOP A load (the one that would read past m_left rows
 * in memory) is replaced with a zero-masked load via MASK_KZ(2).
 * The k(2) mask register is set up by the .CMLEFT dispatcher from
 * the m_load_mask operand (formula: (1 << (2 * (m_left % 8))) - 1).
 *
 * No 8x4_MASK variant is provided here: it would be byte-identical
 * to MICRO_TILE_fx4 family which already exists and is what the
 * 8MASKx4 edge body uses.
 * =================================================================== */

/* MICRO_TILE_24x4_MASK : 3 zmms per A load, top zmm (ZMM(2)) masked */
#define MICRO_TILE_24x4_MASK \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(30, 11, 13, 15) \
    FMA(31, 12, 14, 16) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    FMA(30, 23, 25, 27) \
    FMA(31, 24, 26, 28) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_24x4_MASK (CONJA) - flip A imag halves, top zmm masked */
#define MICRO_TILE_24x4_MASK_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(2), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_24x4_MASK (CONJB) - negate B imag broadcast, top zmm masked */
#define MICRO_TILE_24x4_MASK_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_24x4_MASK (CONJA_CONJB) - both, top zmm masked */
#define MICRO_TILE_24x4_MASK_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    VMULPS(ZMM(2), ZMM(30), ZMM(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 23, 25, 27) \
    FMA(4, 24, 26, 28) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_16x4_MASK : 2 zmms per A load, top zmm (ZMM(1)) masked */
#define MICRO_TILE_16x4_MASK \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(31)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(30, 11, 13) \
    FMA(31, 12, 14) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(30)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(31)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    FMA(30, 23, 25) \
    FMA(31, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_16x4_MASK (CONJA) */
#define MICRO_TILE_16x4_MASK_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_16x4_MASK (CONJB) */
#define MICRO_TILE_16x4_MASK_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1) MASK_KZ(2)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* MICRO_TILE_16x4_MASK (CONJA_CONJB) */
#define MICRO_TILE_16x4_MASK_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    LEA(MEM(RBX, R15, 2), R9) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(R9), ZMM(3)) \
    VBROADCASTSS(MEM(R9, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    VBROADCASTSS(MEM(R9, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(R9, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 23, 25) \
    FMA(4, 24, 26) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \


/* ===================================================================
 * Masked BETA scaling helpers for col-stored C in the 16MASKx4 and
 * 24MASKx4 edge bodies. Only the TOP zmm of each store is masked
 * (it is the one that overruns m_left). The other zmms use the
 * standard non-masked load/store.
 *
 * 8MASKx4 reuses BETA_*_fC (already exists, single-zmm masked).
 * =================================================================== */

/* BETA_GENERIC for 16C with top zmm masked */
#define BETA_GENERIC_16C_MASK(C, R1, I1, R2, I2) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
\
    ALPHA_GENERIC(R1, R2) \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

/* BETA_MINUS_ONE for 16C with top zmm masked */
#define BETA_MINUS_ONE_16C_MASK(C, R1, I1, R2, I2) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
\
    VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VSUBPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

/* BETA_ONE for 16C with top zmm masked */
#define BETA_ONE_16C_MASK(C, R1, I1, R2, I2) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
\
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

/* BETA_GENERIC for 24C with top zmm masked */
#define BETA_GENERIC_24C_MASK(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3) MASK_(k(2))) \
\
    ALPHA_GENERIC(R1, R2, R3) \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VADDPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128) MASK_(k(2))) \

/* BETA_MINUS_ONE for 24C with top zmm masked */
#define BETA_MINUS_ONE_24C_MASK(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3) MASK_(k(2))) \
\
    VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VSUBPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VSUBPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128) MASK_(k(2))) \

/* BETA_ONE for 24C with top zmm masked */
#define BETA_ONE_24C_MASK(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3) MASK_(k(2))) \
\
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VADDPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
\
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128) MASK_(k(2))) \


/* ===================================================================
 * Conjugation-aware MICRO_TILE variants for N=3 (24x3, 16x3, 8x3, fx3).
 *
 * Encoding rules (same as x4 variants):
 *   CONJA       : ZMM(30) = conja_arr (+1,-1,+1,-1,...). All B
 *                 broadcasts go to ZMM(3)/ZMM(4). Each A load is
 *                 in-place multiplied by ZMM(30) to flip imag halves.
 *   CONJB       : ZMM(30) = conjb_arr (all -1). All B broadcasts go
 *                 to ZMM(3)/ZMM(4); ZMM(4) (B.imag) is multiplied by
 *                 ZMM(30) after each broadcast.
 *   CONJA_CONJB : ZMM(30) = conja_arr, ZMM(31) = conjb_arr.
 *                 A * ZMM(30), B.imag * ZMM(31).
 * =================================================================== */

/* Macro to perform a 24x3 micro-tile computation (CONJA) */
#define MICRO_TILE_24x3_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(2), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x3 micro-tile computation (CONJB) */
#define MICRO_TILE_24x3_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x3 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_24x3_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    VMULPS(ZMM(2), ZMM(30), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17, 19, 21) \
    FMA(4, 18, 20, 22) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x3 micro-tile computation (CONJA) */
#define MICRO_TILE_16x3_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x3 micro-tile computation (CONJB) */
#define MICRO_TILE_16x3_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x3 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_16x3_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17, 19) \
    FMA(4, 18, 20) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x3 micro-tile computation (CONJA) */
#define MICRO_TILE_8x3_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x3 micro-tile computation (CONJB) */
#define MICRO_TILE_8x3_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x3 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_8x3_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx3 micro-tile computation (CONJA, f<8) */
#define MICRO_TILE_fx3_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx3 micro-tile computation (CONJB, f<8) */
#define MICRO_TILE_fx3_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx3 micro-tile computation (CONJA_CONJB, f<8) */
#define MICRO_TILE_fx3_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    VBROADCASTSS(MEM(RBX, R15, 2), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 2, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 17) \
    FMA(4, 18) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* ===================================================================
 * Conjugation-aware MICRO_TILE variants for N=2 (24x2, 16x2, 8x2, fx2).
 * =================================================================== */

/* Macro to perform a 24x2 micro-tile computation (CONJA) */
#define MICRO_TILE_24x2_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(2), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x2 micro-tile computation (CONJB) */
#define MICRO_TILE_24x2_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x2 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_24x2_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    VMULPS(ZMM(2), ZMM(30), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13, 15) \
    FMA(4, 12, 14, 16) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x2 micro-tile computation (CONJA) */
#define MICRO_TILE_16x2_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x2 micro-tile computation (CONJB) */
#define MICRO_TILE_16x2_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x2 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_16x2_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11, 13) \
    FMA(4, 12, 14) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x2 micro-tile computation (CONJA) */
#define MICRO_TILE_8x2_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x2 micro-tile computation (CONJB) */
#define MICRO_TILE_8x2_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x2 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_8x2_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx2 micro-tile computation (CONJA, f<8) */
#define MICRO_TILE_fx2_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx2 micro-tile computation (CONJB, f<8) */
#define MICRO_TILE_fx2_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx2 micro-tile computation (CONJA_CONJB, f<8) */
#define MICRO_TILE_fx2_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    VBROADCASTSS(MEM(RBX, R15, 1), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, R15, 1, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    FMA(3, 11) \
    FMA(4, 12) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* ===================================================================
 * Conjugation-aware MICRO_TILE variants for N=1 (24x1, 16x1, 8x1, fx1).
 * =================================================================== */

/* Macro to perform a 24x1 micro-tile computation (CONJA) */
#define MICRO_TILE_24x1_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(2), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x1 micro-tile computation (CONJB) */
#define MICRO_TILE_24x1_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 24x1 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_24x1_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMOVUPS(MEM(RAX, 128), ZMM(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    VMULPS(ZMM(2), ZMM(30), ZMM(2)) \
    FMA(3, 5, 7, 9) \
    FMA(4, 6, 8, 10) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x1 micro-tile computation (CONJA) */
#define MICRO_TILE_16x1_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(1), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x1 micro-tile computation (CONJB) */
#define MICRO_TILE_16x1_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 16x1 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_16x1_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMOVUPS(MEM(RAX, 64), ZMM(1)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    VMULPS(ZMM(1), ZMM(30), ZMM(1)) \
    FMA(3, 5, 7) \
    FMA(4, 6, 8) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x1 micro-tile computation (CONJA) */
#define MICRO_TILE_8x1_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x1 micro-tile computation (CONJB) */
#define MICRO_TILE_8x1_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a 8x1 micro-tile computation (CONJA_CONJB) */
#define MICRO_TILE_8x1_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx1 micro-tile computation (CONJA, f<8) */
#define MICRO_TILE_fx1_CONJA \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(30), ZMM(0), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx1 micro-tile computation (CONJB, f<8) */
#define MICRO_TILE_fx1_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(30), ZMM(4), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro to perform a fx1 micro-tile computation (CONJA_CONJB, f<8) */
#define MICRO_TILE_fx1_CONJA_CONJB \
    VBROADCASTSS(MEM(RBX), ZMM(3)) \
    VBROADCASTSS(MEM(RBX, 4), ZMM(4)) \
    VMULPS(ZMM(4), ZMM(31), ZMM(4)) \
    VMOVUPS(MEM(RAX), ZMM(0) MASK_KZ(2)) \
    VMULPS(ZMM(0), ZMM(30), ZMM(0)) \
    FMA(3, 5) \
    FMA(4, 6) \
    ADD(R14, RBX) \
    ADD(R13, RAX) \

/* Macro for scaling with alpha if it is -1
   in case of 3 loads(24x? cases) */
#define ALPHA_MINUS_ONE_24C(R1, R2, R3) \
    VSUBPS(ZMM(R1), ZMM(2), ZMM(R1)) \
    VSUBPS(ZMM(R2), ZMM(2), ZMM(R2)) \
    VSUBPS(ZMM(R3), ZMM(2), ZMM(R3)) \

/* Macro for scaling with alpha if it is -1
   in case of 2 loads(16x? cases) */
#define ALPHA_MINUS_ONE_16C(R1, R2) \
    VSUBPS(ZMM(R1), ZMM(2), ZMM(R1)) \
    VSUBPS(ZMM(R2), ZMM(2), ZMM(R2)) \

/* Macro for scaling with alpha if it is -1
   in case of 1 loads(8x? cases) */
#define ALPHA_MINUS_ONE_8C(R1) \
    VSUBPS(ZMM(R1), ZMM(2), ZMM(R1)) \

/* Macro to get the ALPHA_MINUS_ONE_? signature from the list */
#define GET_ALPHA_MINUS_ONE(_1, _2, _3, NAME, ...)  NAME

/* Overloaded macro ALPHA_MINUS_ONE with variable arguments */
#define ALPHA_MINUS_ONE(...) \
    GET_ALPHA_MINUS_ONE(__VA_ARGS__, \
    ALPHA_MINUS_ONE_24C, ALPHA_MINUS_ONE_16C, ALPHA_MINUS_ONE_8C)(__VA_ARGS__) \

/* Macro for scaling with beta if it is -1
   in case of 3 loads(24x? cases) */
#define BETA_MINUS_ONE_24C(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3)) \
 \
    VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VSUBPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VSUBPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128)) \

/* Macro for scaling with beta if it is -1
   in case of 2 loads(16x? cases) */
#define BETA_MINUS_ONE_16C(C, R1, I1, R2, I2) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
 \
    VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VSUBPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \

/* Macro for scaling with beta if it is -1
   in case of 1 load(8x? cases) */
#define BETA_MINUS_ONE_8C(C, R1, I1) \
    VMOVUPS(MEM(C), ZMM(R1)) \
 \
    VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \

/* Macro to get the BETA_MINUS_ONE_? signature from the list */
#define GET_BETA_MINUS_ONE(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

/* Overloaded macro BETA_MINUS_ONE with variable arguments */
#define BETA_MINUS_ONE(...) \
    GET_BETA_MINUS_ONE(__VA_ARGS__, \
    BETA_MINUS_ONE_24C, _0, BETA_MINUS_ONE_16C, _1, BETA_MINUS_ONE_8C)(__VA_ARGS__) \

/* Macro for scaling with beta if it is -1
   in case of 1 load(fx? cases, f<8) */
   #define BETA_MINUS_ONE_fC(C, R1, I1) \
   VMOVUPS(MEM(C), ZMM(R1) MASK_(k(2))) \
\
   VSUBPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
\
   VMOVUPS(ZMM(I1), MEM(C) MASK_(k(2))) \

/* Macro for scaling with beta if it is 1
   in case of 3 loads(24x? cases) */
#define BETA_ONE_24C(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
    VMOVUPS(MEM(C, 128), ZMM(R3)) \
 \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VADDPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \
    VMOVUPS(ZMM(I3), MEM(C, 128)) \

/* Macro for scaling with beta if it is 1
   in case of 2 loads(16x? cases) */
#define BETA_ONE_16C(C, R1, I1, R2, I2) \
    VMOVUPS(MEM(C), ZMM(R1)) \
    VMOVUPS(MEM(C, 64), ZMM(R2)) \
 \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \
    VMOVUPS(ZMM(I2), MEM(C, 64)) \

/* Macro for scaling with beta if it is 1
   in case of 1 load(8x? cases) */
#define BETA_ONE_8C(C, R1, I1) \
    VMOVUPS(MEM(C), ZMM(R1)) \
 \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
 \
    VMOVUPS(ZMM(I1), MEM(C)) \

/* Macro to get the BETA_ONE_? signature from the list */
#define GET_BETA_ONE(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

/* Overloaded macro BETA_ONE with variable arguments */
#define BETA_ONE(...) \
    GET_BETA_MINUS_ONE(__VA_ARGS__, \
    BETA_ONE_24C, _0, BETA_ONE_16C, _1, BETA_ONE_8C)(__VA_ARGS__) \

/* Macro for scaling with beta if it is 1
   in case of 1 load(fx? cases, f<8) */
#define BETA_ONE_fC(C, R1, I1) \
    VMOVUPS(MEM(C), ZMM(R1) MASK_(k(2))) \
\
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
\
    VMOVUPS(ZMM(I1), MEM(C) MASK_(k(2))) \

/* Macro to perform 8x8 transpose of 64-bit elements */
/* Transpose is in place(R0...R7), T0...T7 are temporary registers */
#define TRANSPOSE_8x8(R0, R1, R2, R3, R4, R5, R6, R7, \
         T0, T1, T2, T3, T4, T5, T6, T7) \
  /*
    Let's consider the following case:
    ZMM(R0) = { 0, 1, 2, 3, 4, 5, 6, 7 }
    ZMM(R1) = { 8, 9, 10, 11, 12, 13, 14, 15 }
    .
    .
    .
    ZMM(R7) = { 56, 57, 58, 59, 60, 61, 62, 63 }

    Expected output:
    ZMM(R0) = { 0, 8, 16, 24, 32, 40, 48, 56 }
    ZMM(R1) = { 1, 9, 17, 25, 33, 41, 49, 57 }
    .
    .
    .
    ZMM(R7) = { 7, 15, 23, 31, 39, 47, 55, 63 }.
  */ \
  /* Inputs : ZMM(R0) = { 0, 1, 2, 3, 4, 5, 6, 7 }
 ZMM(R1) = { 8, 9, 10, 11, 12, 13, 14, 15 }
 ZMM(R2) = { 16, 17, 18, 19, 20, 21, 22, 23 }
 ZMM(R3) = { 24, 25, 26, 27, 28, 29, 30, 31 }
 ...
     Outputs: ZMM(T0) = { 0, 8, 2, 10, 4, 12, 6, 14 }
 ZMM(R1) = { 1, 9, 3, 11, 5, 13, 7, 15 }
 ZMM(T2) = { 16, 24, 18, 26, 20, 28, 22, 30 }
 ZMM(R3) = { 17, 25, 19, 27, 21, 29, 23, 31 }
 ... */ \
  VUNPCKLPD(ZMM(R1), ZMM(R0), ZMM(T0)) \
  VUNPCKHPD(ZMM(R1), ZMM(R0), ZMM(R1)) \
  VUNPCKLPD(ZMM(R3), ZMM(R2), ZMM(T1)) \
  VUNPCKHPD(ZMM(R3), ZMM(R2), ZMM(R3)) \
  VUNPCKLPD(ZMM(R5), ZMM(R4), ZMM(T2)) \
  VUNPCKHPD(ZMM(R5), ZMM(R4), ZMM(R5)) \
  VUNPCKLPD(ZMM(R7), ZMM(R6), ZMM(T3)) \
  VUNPCKHPD(ZMM(R7), ZMM(R6), ZMM(R7)) \
\
  /* Moving the contents of temporary registers
     to input registers for reuse */ \
  /* Output: ZMM(R0) = { 0, 8, 2, 10, 4, 12, 6, 14 }
             ZMM(R2) = { 16, 24, 18, 26, 20, 28, 22, 30 }
             ZMM(R4) = { 32, 40, 34, 42, 36, 44, 38, 46 }
             ZMM(R6) = { 48, 56, 50, 58, 52, 60, 54, 62 } */ \
  VMOVAPD(ZMM(T0), ZMM(R0)) \
  VMOVAPD(ZMM(T1), ZMM(R2)) \
  VMOVAPD(ZMM(T2), ZMM(R4)) \
  VMOVAPD(ZMM(T3), ZMM(R6)) \
\
  /* Inputs  : ZMM(R0) = { 0, 8, 2, 10, 4, 12, 6, 14 }
  ZMM(R2) = { 16, 24, 18, 26, 20, 28, 22, 30 }
  ZMM(R4) = { 32, 40, 34, 42, 36, 44, 38, 46 }
  ZMM(R6) = { 48, 56, 50, 58, 52, 60, 54, 62 }
     Outputs : ZMM(T0) = { 0, 8, 4, 12, 16, 24, 20, 28 }
  ZMM(T1) = { 32, 40, 36, 44, 48, 56, 52, 60 }
  ZMM(T2) = { 2, 10, 6, 14, 18, 26, 22, 30 }
  ZMM(T3) = { 34, 42, 38, 46, 50, 58, 54, 62 } */ \
  VSHUFF64X2(IMM(0x88), ZMM(R2), ZMM(R0), ZMM(T0)) \
  VSHUFF64X2(IMM(0x88), ZMM(R6), ZMM(R4), ZMM(T1)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R2), ZMM(R0), ZMM(T2)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R6), ZMM(R4), ZMM(T3)) \
\
  /* Inputs  : ZMM(R1) = { 1, 9, 3, 11, 5, 13, 7, 15 }
  ZMM(R3) = { 17, 25, 19, 27, 21, 29, 23, 31 }
  ZMM(R5) = { 33, 41, 35, 43, 37, 45, 39, 47 }
  ZMM(R7) = { 49, 57, 51, 59, 53, 61, 55, 63 }
     Outputs : ZMM(T4) = { 1, 9, 5, 13, 17, 25, 21, 29 }
  ZMM(T5) = { 33, 41, 37, 45, 49, 57, 53, 61 }
  ZMM(T6) = { 3, 11, 7, 15, 19, 27, 23, 31 }
  ZMM(T7) = { 35, 43, 39, 47, 51, 59, 55, 63 } */ \
  VSHUFF64X2(IMM(0x88), ZMM(R3), ZMM(R1), ZMM(T4)) \
  VSHUFF64X2(IMM(0x88), ZMM(R7), ZMM(R5), ZMM(T5)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R3), ZMM(R1), ZMM(T6)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R7), ZMM(R5), ZMM(T7)) \
\
  /* Inputs  : ZMM(T0) = { 0, 8, 4, 12, 16, 24, 20, 28 }
  ZMM(T1) = { 32, 40, 36, 44, 48, 56, 52, 60 }
  ZMM(T2) = { 2, 10, 6, 14, 18, 26, 22, 30 }
  ZMM(T3) = { 34, 42, 38, 46, 50, 58, 54, 62 }

    Outputs :  ZMM(R0) = { 0, 8, 16, 24, 32, 40, 48, 56 }
  ZMM(R2) = { 2, 10, 18, 26, 34, 42, 50, 58 }
  ZMM(R4) = { 4, 12, 20, 28, 36, 44, 52, 60 }
  ZMM(R6) = { 6, 14, 22, 30, 38, 46, 54, 62 } */ \
  VSHUFF64X2(IMM(0x88), ZMM(T1), ZMM(T0), ZMM(R0)) \
  VSHUFF64X2(IMM(0x88), ZMM(T3), ZMM(T2), ZMM(R2)) \
  VSHUFF64X2(IMM(0xDD), ZMM(T1), ZMM(T0), ZMM(R4)) \
  VSHUFF64X2(IMM(0xDD), ZMM(T3), ZMM(T2), ZMM(R6)) \
\
  /* Inputs : ZMM(T4) = { 1, 9, 5, 13, 17, 25, 21, 29 }
 ZMM(T5) = { 33, 41, 37, 45, 49, 57, 53, 61 }
 ZMM(T6) = { 3, 11, 7, 15, 19, 27, 23, 31 }
 ZMM(T7) = { 35, 43, 39, 47, 51, 59, 55, 63 }

    Outputs : ZMM(R1) = { 1, 9, 17, 25, 33, 41, 49, 57 }
 ZMM(R3) = { 3, 11, 19, 27, 35, 43, 51, 59 }
 ZMM(R5) = { 5, 13, 21, 29, 37, 45, 53, 61 }
 ZMM(R7) = { 7, 15, 23, 31, 39, 47, 55, 63 } */ \
  VSHUFF64X2(IMM(0x88), ZMM(T5), ZMM(T4), ZMM(R1)) \
  VSHUFF64X2(IMM(0x88), ZMM(T7), ZMM(T6), ZMM(R3)) \
  VSHUFF64X2(IMM(0xDD), ZMM(T5), ZMM(T4), ZMM(R5)) \
  VSHUFF64X2(IMM(0xDD), ZMM(T7), ZMM(T6), ZMM(R7)) \

/* Macro for beta scaling of a 4x4 micro-tile of C when row-stored */
/* Macro receives alpha*A*B in I1...I4. R1...R4 should be used for loading C */
/* Macro assumes that ZMM(0) and ZMM(1) have beta(real and imag) components
   already broadcasted */
/* Macro assumes R9 and RCX to have the address of C */
#define BETA_GEN_ROW_4x4(R1, I1, R2, I2, R3, I3, R4, I4) \
    /* Load C onto the registers */ \
    VMOVUPS(MEM(R9), YMM(R1)) \
    VMOVUPS(MEM(R9, RDI, 1), YMM(R2)) \
    LEA(MEM(R9, RDI, 2), R9) \
    VMOVUPS(MEM(R9), YMM(R3)) \
    VMOVUPS(MEM(R9, RDI, 1), YMM(R4)) \
\
    /* Reuse the alpha-scaling macro to perform beta-scaling */ \
    ALPHA_GENERIC(R1, R2) \
    ALPHA_GENERIC(R3, R4) \
\
    /* Add them to the result of alpha*A*B */ \
    VADDPS(YMM(R1), YMM(I1), YMM(I1)) \
    VADDPS(YMM(R2), YMM(I2), YMM(I2)) \
    VADDPS(YMM(R3), YMM(I3), YMM(I3)) \
    VADDPS(YMM(R4), YMM(I4), YMM(I4)) \
\
    /* Store the result back to C */ \
    VMOVUPS(YMM(I1), MEM(RCX)) \
    VMOVUPS(YMM(I2), MEM(RCX, RDI, 1)) \
    LEA(MEM(RCX, RDI, 2), RCX) \
    VMOVUPS(YMM(I3), MEM(RCX)) \
    VMOVUPS(YMM(I4), MEM(RCX, RDI, 1)) \

/* Macro for beta scaling of a 4xf(f < 4) micro-tile of C when row-stored */
/* Macro receives alpha*A*B in I1...I4. R1...R4 should be used for loading C */
/* Macro assumes that ZMM(0) and ZMM(1) have beta(real and imag) components
   already broadcasted */
/* Macro assumes R9 and RCX to have the address of C, and k(3) to have the mask */
#define BETA_GEN_ROW_4xf(R1, I1, R2, I2, R3, I3, R4, I4) \
    /* Load C onto the registers using the mask */ \
    VMOVUPS(MEM(R9), ZMM(R1) MASK_(k(3))) \
    VMOVUPS(MEM(R9, RDI, 1), ZMM(R2) MASK_(k(3))) \
    LEA(MEM(R9, RDI, 2), R9) \
    VMOVUPS(MEM(R9), ZMM(R3) MASK_(k(3))) \
    VMOVUPS(MEM(R9, RDI, 1), ZMM(R4) MASK_(k(3))) \
\
    /* Reuse the alpha-scaling macro to perform beta-scaling */ \
    ALPHA_GENERIC(R1, R2) \
    ALPHA_GENERIC(R3, R4) \
\
    /* Add them to the result of alpha*A*B */ \
    VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
    VADDPS(ZMM(R2), ZMM(I2), ZMM(I2)) \
    VADDPS(ZMM(R3), ZMM(I3), ZMM(I3)) \
    VADDPS(ZMM(R4), ZMM(I4), ZMM(I4)) \
\
    /* Store the result back to C using the mask */ \
    VMOVUPS(ZMM(I1), MEM(RCX) MASK_(k(3))) \
    VMOVUPS(ZMM(I2), MEM(RCX, RDI, 1) MASK_(k(3))) \
    LEA(MEM(RCX, RDI, 2), RCX) \
    VMOVUPS(ZMM(I3), MEM(RCX) MASK_(k(3))) \
    VMOVUPS(ZMM(I4), MEM(RCX, RDI, 1) MASK_(k(3))) \

/* Macro for beta scaling of a 4x4 micro-tile of C when beta == 0 */
/* Macro receives alpha*A*B in R1...R4 */
/* Macro assumes RCX to have the address of C */
#define BETA_ZERO_ROW_4x4(R1, R2, R3, R4) \
    /* Store the result back to C */ \
    VMOVUPS(YMM(R1), MEM(RCX)) \
    VMOVUPS(YMM(R2), MEM(RCX, RDI, 1)) \
    LEA(MEM(RCX, RDI, 2), RCX) \
    VMOVUPS(YMM(R3), MEM(RCX)) \
    VMOVUPS(YMM(R4), MEM(RCX, RDI, 1)) \

/* Macro for beta scaling of a 4xf(f < 4) micro-tile of C when beta == 0 */
/* Macro receives alpha*A*B in R1...R4 */
/* Macro assumes RCX to have the address of C, and k(3) to have the mask */
#define BETA_ZERO_ROW_4xf(R1, R2, R3, R4) \
    /* Store the result back to C using the mask */ \
    VMOVUPS(ZMM(R1), MEM(RCX) MASK_(k(3))) \
    VMOVUPS(ZMM(R2), MEM(RCX, RDI, 1) MASK_(k(3))) \
    LEA(MEM(RCX, RDI, 2), RCX) \
    VMOVUPS(ZMM(R3), MEM(RCX) MASK_(k(3))) \
    VMOVUPS(ZMM(R4), MEM(RCX, RDI, 1) MASK_(k(3))) \

/* Macro for beta scaling of a 1x4 micro-tile of C when row-stored */
/* Macro receives alpha*A*B in I1. R1 should be used for loading C */
/* Macro assumes that ZMM(0) and ZMM(1) have beta(real and imag) components
   already broadcasted */
/* Macro assumes RCX to have the address of C */
#define BETA_GEN_ROW_1x4(R1, I1) \
  /* Load C onto the registers */ \
  VMOVUPS(MEM(RCX), YMM(R1)) \
\
  /* Reuse the alpha-scaling macro to perform beta-scaling */ \
  ALPHA_GENERIC(R1) \
\
  /* Add them to the result of alpha*A*B */ \
  VADDPS(YMM(R1), YMM(I1), YMM(I1)) \
\
  /* Store the result back to C */ \
  VMOVUPS(YMM(I1), MEM(RCX)) \

/* Macro for beta scaling of a 1xf(f < 4) micro-tile of C when row-stored */
/* Macro receives alpha*A*B in I1. R1 should be used for loading C */
/* Macro assumes that ZMM(0) and ZMM(1) have beta(real and imag) components
already broadcasted */
/* Macro assumes RCX to have the address of C, and k(3) to have the mask */
#define BETA_GEN_ROW_1xf(R1, I1) \
  /* Load C onto the registers using the mask */ \
  VMOVUPS(MEM(RCX), ZMM(R1) MASK_(k(3))) \
\
  /* Reuse the alpha-scaling macro to perform beta-scaling */ \
  ALPHA_GENERIC(R1) \
\
  /* Add them to the result of alpha*A*B */ \
  VADDPS(ZMM(R1), ZMM(I1), ZMM(I1)) \
\
  /* Store the result back to C using the mask */ \
  VMOVUPS(ZMM(I1), MEM(RCX) MASK_(k(3))) \

/* Macro for beta scaling of a 1x4 micro-tile of C when beta == 0 */
/* Macro receives alpha*A*B in R1 */
/* Macro assumes RCX to have the address of C */
#define BETA_ZERO_ROW_1x4(R1) \
/* Store the result back to C */ \
VMOVUPS(YMM(R1), MEM(RCX)) \

/* Macro for beta scaling of a 1xf(f < 4) micro-tile of C when beta == 0 */
/* Macro receives alpha*A*B in R1 */
/* Macro assumes RCX to have the address of C, and k(3) to have the mask */
#define BETA_ZERO_ROW_1xf(R1) \
/* Store the result back to C using the mask */ \
VMOVUPS(ZMM(R1), MEM(RCX) MASK_(k(3))) \

/*
   ccc:
     | | | |         | | | |        | | | |
     | | | |   +=    | | | | ...    | | | | ...
     | | | |         | | | |        | | | |
     | | | |         | | | |        | | | |

   ccr:
     | | | |        | | | |       --------
     | | | |   +=   | | | | ...   --------
     | | | |        | | | |       --------
     | | | |        | | | |           :

   Assumptions:
   - A is column stored;
   - B is row-stored or column-stored;
   Therefore, this (c)olumn-preferential kernel is well-suited for contiguous
   (v)ector loads on A and single-element broadcasts from B.

   NOTE: These kernels explicitly support row-oriented IO, implemented
   via an in-register transpose. And thus they also support the rcc and
   rcr cases, though only rcc is ever utilized (because rcr is handled by
   transposing the operation and executing ccr, which does not incur the
   cost of the in-register transpose).

   rcc:
     ---------       | | | |      | | | |
     ---------  +=   | | | | ...  | | | | ...
     ---------       | | | |      | | | |
     ---------       | | | |      | | | |

*/

/*
   Arrays used for complex conjugate operations in CGEMM kernels.

   conja_arr: alternates between 1.0 and -1.0 to selectively negate
              the imaginary halves of every interleaved (Re,Im) pair
              when multiplying A by it. ZMM = 16 floats = 8 scomplex
              pairs.

   conjb_arr: all -1.0; multiplying it onto a B.imag broadcast (which
              has the same value replicated across all 16 lanes)
              uniformly negates the imaginary component of B.
*/
static float conja_arr[] = { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                             1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
static float conjb_arr[] = {-1.0f, -1.0f,-1.0f, -1.0f,-1.0f, -1.0f,-1.0f, -1.0f,
                            -1.0f, -1.0f,-1.0f, -1.0f,-1.0f, -1.0f,-1.0f, -1.0f };

/*
   Conj-array load sequences emitted inside the per-conj asm blocks,
   right after RESET_REGISTERS (which clobbers ZMM(30)/ZMM(31)) and
   before the k-loop. Empty for the NN case.

   - CONJ_LOAD_A   : ZMM(30) <- conja_arr (alt +1/-1 pattern)
   - CONJ_LOAD_B   : ZMM(30) <- conjb_arr (all -1)
   - CONJ_LOAD_AB  : ZMM(30) <- conja_arr, ZMM(31) <- conjb_arr
*/
#define CONJ_LOAD_NN   /* no-op */
#define CONJ_LOAD_A    MOV(VAR(conja_array), R9) \
                       VMOVUPS(MEM(R9), ZMM(30))
#define CONJ_LOAD_B    MOV(VAR(conjb_array), R9) \
                       VMOVUPS(MEM(R9), ZMM(30))
#define CONJ_LOAD_AB   MOV(VAR(conja_array), R9) \
                       VMOVUPS(MEM(R9), ZMM(30)) \
                       MOV(VAR(conjb_array), R9) \
                       VMOVUPS(MEM(R9), ZMM(31))

/*
   ====================================================================
   CGEMM edge-row body macros (inline equivalents of the standalone
   ?x4 kernels, mirroring the ZGEMM_{8x4,8MASKx4,4x4,4MASKx4,2x4} edge
   pattern in z12x4m exactly). Each body:
     1. RESET_REGISTERS to clear accumulators inherited from the
        m_iter slab(s) above, then re-emit CL to restore
        ZMM(30)/ZMM(31) for the current conj path.
     2. Runs its OWN dedicated k-loop (k_iter unrolled by 4, then
        k_left fringe) over the supplied MICRO_TILE variant MT.
     3. Accumulates (PERMUTE + ACC_COL) over its own register set.
     4. Alpha-scales with the alpha_mul_type fast paths.
     5. Beta-scales with the beta_mul_type fast paths, dispatching
        col-store vs row-store on cs_c == sizeof(scomplex).
     6. JMP(.CONCLUDE) at the end of every terminal block.

   Labels inside each body are suffixed (e.g. _E8M) so they do not
   collide with the corresponding labels in the main per-MR body or
   in sibling edge bodies. Each conj-variant expansion of the outer
   CGEMM_24x4_MAIN_BODY produces its OWN BEGIN_ASM/END_ASM block,
   so suffixes only need to be unique within one body, not across
   conj variants.
   ====================================================================
*/

/*
   8MASKx4 edge body : covers m_left in [1, 7].

   Uses MICRO_TILE_fx4 family (1 zmm masked) and BETA_*_fC scalers
   that already exist for the standalone fx4 kernel.

   Assumes on entry:
     - RAX  : current A row-slab base (caller copies from R10)
     - RBX  : current B base          (caller copies from RDX)
     - RCX  : current C row-slab base (caller copies from R12)
     - k(2) : m_load_mask (caller loads from VAR(m_load_mask))
     - R13/R14/R15/RDI/RSI : a/b/c stride bytes (untouched by outer
       m_iter loop body, still valid here)
*/
#define CGEMM_8MASKX4_BODY(MT, CL)                                                  \
    /* Reset all scratch + accum registers (also clobbers ZMM(30)/(31)) */          \
    RESET_REGISTERS                                                                 \
    /* Reload conja/conjb arrays into ZMM(30)/ZMM(31) for this conj path */         \
    CL                                                                              \
                                                                                    \
    /* Dedicated k-loop : k_iter unrolled by 4 */                                   \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT_E8M)                                                                 \
    LABEL(.CKMAINLOOP_E8M)                                                          \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP_E8M)                                                            \
                                                                                    \
    /* k_left fringe */                                                             \
    LABEL(.CKLEFT_E8M)                                                              \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE_E8M)                                                             \
    LABEL(.CKLEFTLOOP_E8M)                                                          \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP_E8M)                                                            \
                                                                                    \
    /* Accumulate A*B over 4 column registers (1 zmm/col) */                        \
    LABEL(.ACCUMULATE_E8M)                                                          \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    PERMUTE(24)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
    ACC_COL(23, 24)                                                                 \
                                                                                    \
    /* Alpha scaling */                                                             \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL_E8M)                                                         \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    ALPHA_MINUS_ONE(24)                                                             \
    JMP(.BETA_SCALE_E8M)                                                            \
                                                                                    \
    LABEL(.ALPHA_GENERAL_E8M)                                                       \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE_E8M)                                                            \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
    ALPHA_GENERIC(24)                                                               \
                                                                                    \
    /* Beta scaling - dispatch row-store vs col-store on cs_c */                    \
    LABEL(.BETA_SCALE_E8M)                                                          \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C_E8M)                                                          \
                                                                                    \
    /* Column-stored C : beta dispatch */                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_E8M)                                                                  \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD_E8M)                                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL_E8M)                                                          \
                                                                                    \
    /* beta == -1 : C = alpha*A*B - C, masked */                                    \
    BETA_MINUS_ONE_fC(RCX, 5, 6)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 11, 12)                                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 17, 18)                                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 23, 24)                                                  \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_E8M)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_fC(RCX, 5, 6)                                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 11, 12)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 17, 18)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 23, 24)                                                    \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 1 : C = alpha*A*B + C, masked */                                     \
    LABEL(.ADD_E8M)                                                                 \
    BETA_ONE_fC(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 11, 12)                                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 17, 18)                                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 23, 24)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 0 : C = alpha*A*B, masked */                                         \
    LABEL(.STORE_E8M)                                                               \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                  \
    VMOVUPS(ZMM(18), MEM(R9) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1) MASK_(k(2)))                                   \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* Row-stored C path : transpose then per-m_left scalar row store */            \
    LABEL(.ROW_STORAGE_C_E8M)                                                       \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW_E8M)                                                              \
                                                                                    \
    /* beta != 0 (general path - also used for +/-1 since row pathways */           \
    /* for those are not separately fast-pathed in the existing fx4) */             \
    LABEL(.BETA_GENERAL_ROW_E8M)                                                    \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(7), R8)                                                                 \
    JE(.SROW_GEN_E8M_7)                                                             \
    CMP(IMM(6), R8)                                                                 \
    JE(.SROW_GEN_E8M_6)                                                             \
    CMP(IMM(5), R8)                                                                 \
    JE(.SROW_GEN_E8M_5)                                                             \
    CMP(IMM(4), R8)                                                                 \
    JE(.SROW_GEN_E8M_4)                                                             \
    CMP(IMM(3), R8)                                                                 \
    JE(.SROW_GEN_E8M_3)                                                             \
    CMP(IMM(2), R8)                                                                 \
    JE(.SROW_GEN_E8M_2)                                                             \
    CMP(IMM(1), R8)                                                                 \
    JE(.SROW_GEN_E8M_1)                                                             \
                                                                                    \
    LABEL(.SROW_GEN_E8M_7)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(25, 17)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_6)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_5)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_4)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_3)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_2)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E8M_1)                                                          \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 0 : transpose then per-m_left scalar row store */                    \
    LABEL(.STORE_ROW_E8M)                                                           \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(7), R8)                                                                 \
    JE(.STROW_E8M_7)                                                                \
    CMP(IMM(6), R8)                                                                 \
    JE(.STROW_E8M_6)                                                                \
    CMP(IMM(5), R8)                                                                 \
    JE(.STROW_E8M_5)                                                                \
    CMP(IMM(4), R8)                                                                 \
    JE(.STROW_E8M_4)                                                                \
    CMP(IMM(3), R8)                                                                 \
    JE(.STROW_E8M_3)                                                                \
    CMP(IMM(2), R8)                                                                 \
    JE(.STROW_E8M_2)                                                                \
    CMP(IMM(1), R8)                                                                 \
    JE(.STROW_E8M_1)                                                                \
                                                                                    \
    LABEL(.STROW_E8M_7)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(17)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_6)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_5)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_4)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_3)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_2)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E8M_1)                                                             \
    BETA_ZERO_ROW_1x4(6)                                                            \
    JMP(.CONCLUDE)                                                                  \


/*
   8x4 edge body : covers m_left == 8 exactly.

   Uses MICRO_TILE_8x4 family (1 zmm, FULL/unmasked) and BETA_*_8C
   scalers (1-zmm column store, full register, no mask). Row store
   path emits exactly 8 rows so no per-m_left dispatch is needed.

   Assumes on entry (same contract as CGEMM_8MASKX4_BODY):
     - RAX  : current A row-slab base   (caller copies from R10)
     - RBX  : current B base            (caller copies from RDX)
     - RCX  : current C row-slab base   (caller copies from R12)
     - R13/R14/R15/RDI/RSI : a/b/c stride bytes (still valid)
*/
#define CGEMM_8X4_EDGE_BODY(MT, CL)                                                 \
    /* Reset all scratch + accum registers (also clobbers ZMM(30)/(31)) */          \
    RESET_REGISTERS                                                                 \
    /* Reload conja/conjb arrays into ZMM(30)/ZMM(31) for this conj path */         \
    CL                                                                              \
                                                                                    \
    /* Dedicated k-loop : k_iter unrolled by 4 */                                   \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT_E8)                                                                  \
    LABEL(.CKMAINLOOP_E8)                                                           \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP_E8)                                                             \
                                                                                    \
    /* k_left fringe */                                                             \
    LABEL(.CKLEFT_E8)                                                               \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE_E8)                                                              \
    LABEL(.CKLEFTLOOP_E8)                                                           \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP_E8)                                                             \
                                                                                    \
    /* Accumulate A*B over 4 column registers (1 zmm/col) */                        \
    LABEL(.ACCUMULATE_E8)                                                           \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    PERMUTE(24)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
    ACC_COL(23, 24)                                                                 \
                                                                                    \
    /* Alpha scaling */                                                             \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL_E8)                                                          \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    ALPHA_MINUS_ONE(24)                                                             \
    JMP(.BETA_SCALE_E8)                                                             \
                                                                                    \
    LABEL(.ALPHA_GENERAL_E8)                                                        \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE_E8)                                                             \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
    ALPHA_GENERIC(24)                                                               \
                                                                                    \
    /* Beta scaling - dispatch row-store vs col-store on cs_c */                    \
    LABEL(.BETA_SCALE_E8)                                                           \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C_E8)                                                           \
                                                                                    \
    /* Column-stored C : beta dispatch */                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_E8)                                                                   \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD_E8)                                                                     \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL_E8)                                                           \
                                                                                    \
    /* beta == -1 : C = alpha*A*B - C, full (no mask) */                            \
    BETA_MINUS_ONE(RCX, 5, 6)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12)                                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18)                                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 23, 24)                                                     \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_E8)                                                         \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6)                                                         \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 23, 24)                                                       \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 1 : C = alpha*A*B + C, full */                                       \
    LABEL(.ADD_E8)                                                                  \
    BETA_ONE(RCX, 5, 6)                                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12)                                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18)                                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 23, 24)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 0 : C = alpha*A*B, full */                                           \
    LABEL(.STORE_E8)                                                                \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* Row-stored C path : transpose then store all 8 rows in 4x4 blocks */         \
    LABEL(.ROW_STORAGE_C_E8)                                                        \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW_E8)                                                               \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW_E8)                                                     \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STORE_ROW_E8)                                                            \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    JMP(.CONCLUDE)                                                                  \


/*
   16MASKx4 edge body : covers m_left in [9, 15].

   Uses MICRO_TILE_16x4_MASK family (2 zmm/col, top zmm masked) and
   BETA_*_16C_MASK helpers (top zmm load/store masked). Row store
   path : 8 full rows from 1st transpose + per-(m_left-8) dispatch
   from 2nd transpose for the partial 1..7 rows in the top half.
*/
#define CGEMM_16MASKX4_BODY(MT, CL)                                                 \
    RESET_REGISTERS                                                                 \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT_E16M)                                                                \
    LABEL(.CKMAINLOOP_E16M)                                                         \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP_E16M)                                                           \
                                                                                    \
    LABEL(.CKLEFT_E16M)                                                             \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE_E16M)                                                            \
    LABEL(.CKLEFTLOOP_E16M)                                                         \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP_E16M)                                                           \
                                                                                    \
    LABEL(.ACCUMULATE_E16M)                                                         \
    PERMUTE(6, 8)                                                                   \
    PERMUTE(12, 14)                                                                 \
    PERMUTE(18, 20)                                                                 \
    PERMUTE(24, 26)                                                                 \
    ACC_COL(5, 6, 7, 8)                                                             \
    ACC_COL(11, 12, 13, 14)                                                         \
    ACC_COL(17, 18, 19, 20)                                                         \
    ACC_COL(23, 24, 25, 26)                                                         \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL_E16M)                                                        \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    ALPHA_MINUS_ONE(12, 14)                                                         \
    ALPHA_MINUS_ONE(18, 20)                                                         \
    ALPHA_MINUS_ONE(24, 26)                                                         \
    JMP(.BETA_SCALE_E16M)                                                           \
                                                                                    \
    LABEL(.ALPHA_GENERAL_E16M)                                                      \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE_E16M)                                                           \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
    ALPHA_GENERIC(12, 14)                                                           \
    ALPHA_GENERIC(18, 20)                                                           \
    ALPHA_GENERIC(24, 26)                                                           \
                                                                                    \
    LABEL(.BETA_SCALE_E16M)                                                         \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C_E16M)                                                         \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_E16M)                                                                 \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD_E16M)                                                                   \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL_E16M)                                                         \
                                                                                    \
    BETA_MINUS_ONE_16C_MASK(RCX, 5, 6, 7, 8)                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_16C_MASK(RCX, 11, 12, 13, 14)                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_16C_MASK(RCX, 17, 18, 19, 20)                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_16C_MASK(RCX, 23, 24, 25, 26)                                    \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_E16M)                                                       \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_16C_MASK(RCX, 5, 6, 7, 8)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_16C_MASK(RCX, 11, 12, 13, 14)                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_16C_MASK(RCX, 17, 18, 19, 20)                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_16C_MASK(RCX, 23, 24, 25, 26)                                      \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.ADD_E16M)                                                                \
    BETA_ONE_16C_MASK(RCX, 5, 6, 7, 8)                                              \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_16C_MASK(RCX, 11, 12, 13, 14)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_16C_MASK(RCX, 17, 18, 19, 20)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_16C_MASK(RCX, 23, 24, 25, 26)                                          \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STORE_E16M)                                                              \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64) MASK_(k(2)))                                       \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64) MASK_(k(2)))                              \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64) MASK_(k(2)))                                       \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64) MASK_(k(2)))                               \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* Row-stored C path : transpose top 8 rows full, dispatch partial */           \
    /* rows from 2nd transpose by (m_left - 8) in [1, 7].                */         \
    LABEL(.ROW_STORAGE_C_E16M)                                                      \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW_E16M)                                                             \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW_E16M)                                                   \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    /* Top 8 rows : full 4x4 stores */                                              \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    /* Partial top : up to 7 rows in {8, 14, 20, 26, 5, 11, 17} */                  \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(15), R8)                                                                \
    JE(.SROW_GEN_E16M_15)                                                           \
    CMP(IMM(14), R8)                                                                \
    JE(.SROW_GEN_E16M_14)                                                           \
    CMP(IMM(13), R8)                                                                \
    JE(.SROW_GEN_E16M_13)                                                           \
    CMP(IMM(12), R8)                                                                \
    JE(.SROW_GEN_E16M_12)                                                           \
    CMP(IMM(11), R8)                                                                \
    JE(.SROW_GEN_E16M_11)                                                           \
    CMP(IMM(10), R8)                                                                \
    JE(.SROW_GEN_E16M_10)                                                           \
    CMP(IMM(9), R8)                                                                 \
    JE(.SROW_GEN_E16M_9)                                                            \
                                                                                    \
    LABEL(.SROW_GEN_E16M_15)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 20)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 26)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(25, 17)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_14)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 20)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 26)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_13)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 20)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 26)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_12)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 20)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 26)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_11)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 20)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_10)                                                        \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 14)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E16M_9)                                                         \
    BETA_GEN_ROW_1x4(7, 8)                                                          \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 0 row-store path */                                                  \
    LABEL(.STORE_ROW_E16M)                                                          \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(15), R8)                                                                \
    JE(.STROW_E16M_15)                                                              \
    CMP(IMM(14), R8)                                                                \
    JE(.STROW_E16M_14)                                                              \
    CMP(IMM(13), R8)                                                                \
    JE(.STROW_E16M_13)                                                              \
    CMP(IMM(12), R8)                                                                \
    JE(.STROW_E16M_12)                                                              \
    CMP(IMM(11), R8)                                                                \
    JE(.STROW_E16M_11)                                                              \
    CMP(IMM(10), R8)                                                                \
    JE(.STROW_E16M_10)                                                              \
    CMP(IMM(9), R8)                                                                 \
    JE(.STROW_E16M_9)                                                               \
                                                                                    \
    LABEL(.STROW_E16M_15)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(20)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(26)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(17)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_14)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(20)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(26)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_13)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(20)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(26)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_12)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(20)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(26)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_11)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(20)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_10)                                                           \
    BETA_ZERO_ROW_1x4(8)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(14)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E16M_9)                                                            \
    BETA_ZERO_ROW_1x4(8)                                                            \
    JMP(.CONCLUDE)                                                                  \


/*
   16x4 edge body : covers m_left == 16 exactly.

   Uses MICRO_TILE_16x4 family (2 zmm/col, FULL/unmasked) and
   BETA_*_16C scalers (no mask). Row store emits exactly 16 rows
   in 4x4 blocks (4 BETA_GEN_ROW_4x4 calls).
*/
#define CGEMM_16X4_EDGE_BODY(MT, CL)                                                \
    RESET_REGISTERS                                                                 \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT_E16)                                                                 \
    LABEL(.CKMAINLOOP_E16)                                                          \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP_E16)                                                            \
                                                                                    \
    LABEL(.CKLEFT_E16)                                                              \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE_E16)                                                             \
    LABEL(.CKLEFTLOOP_E16)                                                          \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP_E16)                                                            \
                                                                                    \
    LABEL(.ACCUMULATE_E16)                                                          \
    PERMUTE(6, 8)                                                                   \
    PERMUTE(12, 14)                                                                 \
    PERMUTE(18, 20)                                                                 \
    PERMUTE(24, 26)                                                                 \
    ACC_COL(5, 6, 7, 8)                                                             \
    ACC_COL(11, 12, 13, 14)                                                         \
    ACC_COL(17, 18, 19, 20)                                                         \
    ACC_COL(23, 24, 25, 26)                                                         \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL_E16)                                                         \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    ALPHA_MINUS_ONE(12, 14)                                                         \
    ALPHA_MINUS_ONE(18, 20)                                                         \
    ALPHA_MINUS_ONE(24, 26)                                                         \
    JMP(.BETA_SCALE_E16)                                                            \
                                                                                    \
    LABEL(.ALPHA_GENERAL_E16)                                                       \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE_E16)                                                            \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
    ALPHA_GENERIC(12, 14)                                                           \
    ALPHA_GENERIC(18, 20)                                                           \
    ALPHA_GENERIC(24, 26)                                                           \
                                                                                    \
    LABEL(.BETA_SCALE_E16)                                                          \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C_E16)                                                          \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_E16)                                                                  \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD_E16)                                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL_E16)                                                          \
                                                                                    \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)                                                 \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20)                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 23, 24, 25, 26)                                             \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_E16)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                               \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18, 19, 20)                                               \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 23, 24, 25, 26)                                               \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.ADD_E16)                                                                 \
    BETA_ONE(RCX, 5, 6, 7, 8)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18, 19, 20)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 23, 24, 25, 26)                                                   \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STORE_E16)                                                               \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.ROW_STORAGE_C_E16)                                                       \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW_E16)                                                              \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW_E16)                                                    \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STORE_ROW_E16)                                                           \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    JMP(.CONCLUDE)                                                                  \


/*
   24MASKx4 edge body : covers m_left in [17, 23].

   Uses MICRO_TILE_24x4_MASK family (3 zmm/col, top zmm masked) and
   BETA_*_24C_MASK helpers (top zmm load/store masked). Row store
   path : 16 full rows from 1st + 2nd transposes + per-(m_left-16)
   dispatch from 3rd transpose for the partial 1..7 rows in the top.
*/
#define CGEMM_24MASKX4_BODY(MT, CL)                                                 \
    RESET_REGISTERS                                                                 \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT_E24M)                                                                \
    LABEL(.CKMAINLOOP_E24M)                                                         \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP_E24M)                                                           \
                                                                                    \
    LABEL(.CKLEFT_E24M)                                                             \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE_E24M)                                                            \
    LABEL(.CKLEFTLOOP_E24M)                                                         \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP_E24M)                                                           \
                                                                                    \
    LABEL(.ACCUMULATE_E24M)                                                         \
    PERMUTE(6, 8, 10)                                                               \
    PERMUTE(12, 14, 16)                                                             \
    PERMUTE(18, 20, 22)                                                             \
    PERMUTE(24, 26, 28)                                                             \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                      \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                 \
    ACC_COL(17, 18, 19, 20, 21, 22)                                                 \
    ACC_COL(23, 24, 25, 26, 27, 28)                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL_E24M)                                                        \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8, 10)                                                       \
    ALPHA_MINUS_ONE(12, 14, 16)                                                     \
    ALPHA_MINUS_ONE(18, 20, 22)                                                     \
    ALPHA_MINUS_ONE(24, 26, 28)                                                     \
    JMP(.BETA_SCALE_E24M)                                                           \
                                                                                    \
    LABEL(.ALPHA_GENERAL_E24M)                                                      \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE_E24M)                                                           \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8, 10)                                                         \
    ALPHA_GENERIC(12, 14, 16)                                                       \
    ALPHA_GENERIC(18, 20, 22)                                                       \
    ALPHA_GENERIC(24, 26, 28)                                                       \
                                                                                    \
    LABEL(.BETA_SCALE_E24M)                                                         \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C_E24M)                                                         \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_E24M)                                                                 \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD_E24M)                                                                   \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL_E24M)                                                         \
                                                                                    \
    BETA_MINUS_ONE_24C_MASK(RCX, 5, 6, 7, 8, 9, 10)                                 \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_24C_MASK(RCX, 11, 12, 13, 14, 15, 16)                            \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_24C_MASK(RCX, 17, 18, 19, 20, 21, 22)                            \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_24C_MASK(RCX, 23, 24, 25, 26, 27, 28)                            \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_E24M)                                                       \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_24C_MASK(RCX, 5, 6, 7, 8, 9, 10)                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_24C_MASK(RCX, 11, 12, 13, 14, 15, 16)                              \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_24C_MASK(RCX, 17, 18, 19, 20, 21, 22)                              \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_24C_MASK(RCX, 23, 24, 25, 26, 27, 28)                              \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.ADD_E24M)                                                                \
    BETA_ONE_24C_MASK(RCX, 5, 6, 7, 8, 9, 10)                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_24C_MASK(RCX, 11, 12, 13, 14, 15, 16)                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_24C_MASK(RCX, 17, 18, 19, 20, 21, 22)                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_24C_MASK(RCX, 23, 24, 25, 26, 27, 28)                                  \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STORE_E24M)                                                              \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(10), MEM(RCX, 128) MASK_(k(2)))                                     \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128) MASK_(k(2)))                             \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    VMOVUPS(ZMM(22), MEM(R9, 128) MASK_(k(2)))                                      \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))                                           \
    VMOVUPS(ZMM(28), MEM(R9, RSI, 1, 128) MASK_(k(2)))                              \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* Row-stored C path : transpose top 16 rows full + partial 3rd block */        \
    LABEL(.ROW_STORAGE_C_E24M)                                                      \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW_E24M)                                                             \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW_E24M)                                                   \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    /* Top 16 rows : 2 full transposes */                                           \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    /* Partial top : up to 7 rows from 3rd transpose */                             \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(23), R8)                                                                \
    JE(.SROW_GEN_E24M_23)                                                           \
    CMP(IMM(22), R8)                                                                \
    JE(.SROW_GEN_E24M_22)                                                           \
    CMP(IMM(21), R8)                                                                \
    JE(.SROW_GEN_E24M_21)                                                           \
    CMP(IMM(20), R8)                                                                \
    JE(.SROW_GEN_E24M_20)                                                           \
    CMP(IMM(19), R8)                                                                \
    JE(.SROW_GEN_E24M_19)                                                           \
    CMP(IMM(18), R8)                                                                \
    JE(.SROW_GEN_E24M_18)                                                           \
    CMP(IMM(17), R8)                                                                \
    JE(.SROW_GEN_E24M_17)                                                           \
                                                                                    \
    LABEL(.SROW_GEN_E24M_23)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 22)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 28)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(25, 17)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_22)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 22)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 28)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_21)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 22)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 28)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_20)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 22)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 28)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_19)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 22)                                                        \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_18)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 16)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.SROW_GEN_E24M_17)                                                        \
    BETA_GEN_ROW_1x4(7, 10)                                                         \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    /* beta == 0 row-store path */                                                  \
    LABEL(.STORE_ROW_E24M)                                                          \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_left), R8)                                                            \
    CMP(IMM(23), R8)                                                                \
    JE(.STROW_E24M_23)                                                              \
    CMP(IMM(22), R8)                                                                \
    JE(.STROW_E24M_22)                                                              \
    CMP(IMM(21), R8)                                                                \
    JE(.STROW_E24M_21)                                                              \
    CMP(IMM(20), R8)                                                                \
    JE(.STROW_E24M_20)                                                              \
    CMP(IMM(19), R8)                                                                \
    JE(.STROW_E24M_19)                                                              \
    CMP(IMM(18), R8)                                                                \
    JE(.STROW_E24M_18)                                                              \
    CMP(IMM(17), R8)                                                                \
    JE(.STROW_E24M_17)                                                              \
                                                                                    \
    LABEL(.STROW_E24M_23)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(22)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(28)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(17)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_22)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(22)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(28)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_21)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(22)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(28)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_20)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(22)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(28)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_19)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(22)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_18)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(16)                                                           \
    JMP(.CONCLUDE)                                                                  \
                                                                                    \
    LABEL(.STROW_E24M_17)                                                           \
    BETA_ZERO_ROW_1x4(10)                                                           \
    JMP(.CONCLUDE)                                                                  \


/*
   Main per-MR-iteration body of bli_cgemmsup_cv_zen4_asm_24x4m,
   parameterised by:

     MT        : MICRO_TILE_24x4 variant used in the m_iter k-loop
                 (one of MICRO_TILE_24x4, _CONJA, _CONJB, _CONJA_CONJB)
     CL        : Conj-array load sequence emitted after RESET_REGISTERS
                 (one of CONJ_LOAD_NN, _A, _B, _AB)
     MT_8MASK  : MICRO_TILE_fx4 variant for the 8MASKx4 edge body
                 (m_left in [1,7])
     MT_8      : MICRO_TILE_8x4 variant for the 8x4 edge body
                 (m_left == 8)
     MT_16MASK : MICRO_TILE_16x4_MASK variant for the 16MASKx4 edge
                 body (m_left in [9,15])
     MT_16     : MICRO_TILE_16x4 variant for the 16x4 edge body
                 (m_left == 16)
     MT_24MASK : MICRO_TILE_24x4_MASK variant for the 24MASKx4 edge
                 body (m_left in [17,23])

   This is the byte-equivalent of having four hand-duplicated asm
   bodies (one per conj quadrant) as the canonical z12x4m kernel
   does. We use a parameterised macro purely to keep the source
   maintainable; the preprocessor expands into the same four full
   asm bodies the duplicated form would produce.

   Trailing m_left handling (full ZGEMM_12x4m mirror):
     m_left in [1,7]   : INLINE via CGEMM_8MASKX4_BODY
                         (mirrors ZGEMM_4MASKx4 in z12x4m)
     m_left == 8       : INLINE via CGEMM_8X4_EDGE_BODY
                         (mirrors ZGEMM_8x4 in z12x4m)
     m_left in [9,15]  : INLINE via CGEMM_16MASKX4_BODY
                         (mirrors ZGEMM_8MASKx4 in z12x4m)
     m_left == 16      : INLINE via CGEMM_16X4_EDGE_BODY
                         (no z12x4m direct analog since MR_z = 12)
     m_left in [17,23] : INLINE via CGEMM_24MASKX4_BODY
                         (mirrors ZGEMM_12MASKx4 in z12x4m)

   All five edge bodies have their OWN dedicated k-loop, accumulate,
   alpha-scale, beta-scale, and column-store + row-store paths,
   following the canonical ZGEMM 12x4m pattern exactly.
*/
#define CGEMM_24x4_MAIN_BODY(MT, CL, MT_8MASK, MT_8, MT_16MASK, MT_16, MT_24MASK)   \
    MOV(VAR(a), R10)               /* R10 = base addr of A (MCXKC block) */         \
    MOV(VAR(b), RDX)               /* RDX = base addr of B (KCXNR block) */         \
    MOV(VAR(c), R12)               /* R12 = base addr of C (MCxNR block) */         \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)        /* R13 = sizeof(scomplex)*cs_a */                \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)        /* R14 = sizeof(scomplex)*rs_b */                \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)        /* R15 = sizeof(scomplex)*cs_b */                \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)        /* RDI = sizeof(scomplex)*rs_c */                \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)        /* RSI = sizeof(scomplex)*cs_c */                \
                                                                                    \
    /* Intermediate register for complex arithmetic */                              \
    MOV(VAR(v), R9)                /* Used in fmaddsub instruction */               \
    VBROADCASTSS(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */              \
                                                                                    \
    MOV(VAR(m_iter), R11)          /* Iterating in steps of MR, until MC */         \
    TEST(R11, R11)                                                                  \
    JZ(.CMLEFT)                    /* m_iter == 0 : skip directly to m_left edge */ \
    LABEL(.CMLOOP)                                                                  \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    /* Reset all scratch + accum registers (also clobbers ZMM(30)/ZMM(31)) */       \
    RESET_REGISTERS                                                                 \
                                                                                    \
    /* Reload conja/conjb arrays into ZMM(30)/ZMM(31) for the current conj path */  \
    CL                                                                              \
                                                                                    \
    /* Setting iterator for k */                                                    \
    MOV(VAR(k_iter), R8)                                                            \
                                                                                    \
    LABEL(.CK_BP) /* Computation before prefetching */                              \
    SUB(IMM(4 + PREFETCH_DIST_C), R8)                                               \
    JLE(.CK_DP)                                                                     \
    LABEL(.CKITERLOOP_BP)                                                           \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKITERLOOP_BP)                                                             \
                                                                                    \
    LABEL(.CK_DP) /* Computation during prefetching */                              \
    ADD(IMM(4), R8)                                                                 \
    JLE(.CK_AP)                                                                     \
    MOV(RCX, R9)                                                                    \
    LABEL(.CKITERLOOP_DP)                                                           \
    PREFETCH(1, MEM(R9))                                                            \
    MT                                                                              \
    PREFETCH(1, MEM(R9, 64))                                                        \
    MT                                                                              \
    PREFETCH(1, MEM(R9, 128))                                                       \
    MT                                                                              \
    MT                                                                              \
    ADD(RSI, R9)                                                                    \
    DEC(R8)                                                                         \
    JNZ(.CKITERLOOP_DP)                                                             \
                                                                                    \
    LABEL(.CK_AP) /* Computation after prefetching */                               \
    ADD(IMM(0 + PREFETCH_DIST_C), R8)                                               \
    JLE(.CKLEFT)                                                                    \
    LABEL(.CKITERLOOP_AP)                                                           \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKITERLOOP_AP)                                                             \
                                                                                    \
    /* Remainder loop for k (k_fringe) */                                           \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    /* Accumulating A*B over 12 registers */                                        \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8, 10)                                                               \
    PERMUTE(12, 14, 16)                                                             \
    PERMUTE(18, 20, 22)                                                             \
    PERMUTE(24, 26, 28)                                                             \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                      \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                 \
    ACC_COL(17, 18, 19, 20, 21, 22)                                                 \
    ACC_COL(23, 24, 25, 26, 27, 28)                                                 \
                                                                                    \
    /* Alpha scaling */                                                             \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8, 10)                                                       \
    ALPHA_MINUS_ONE(12, 14, 16)                                                     \
    ALPHA_MINUS_ONE(18, 20, 22)                                                     \
    ALPHA_MINUS_ONE(24, 26, 28)                                                     \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8, 10)                                                         \
    ALPHA_GENERIC(12, 14, 16)                                                       \
    ALPHA_GENERIC(18, 20, 22)                                                       \
    ALPHA_GENERIC(24, 26, 28)                                                       \
                                                                                    \
    /* Beta scaling */                                                              \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20, 21, 22)                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 23, 24, 25, 26, 27, 28)                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)                                            \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18, 19, 20, 21, 22)                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 23, 24, 25, 26, 27, 28)                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)                                                \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18, 19, 20, 21, 22)                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 23, 24, 25, 26, 27, 28)                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(10), MEM(RCX, 128))                                                 \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))                                         \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    VMOVUPS(ZMM(22), MEM(R9, 128))                                                  \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))                                           \
    VMOVUPS(ZMM(28), MEM(R9, RSI, 1, 128))                                          \
    JMP(.END)                                                                       \
                                                                                    \
    /* Beta scaling when C is row stored */                                         \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 10, 9, 16, 13, 22, 15, 28)                                  \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(10, 16, 22, 28)                                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
                                                                                    \
    LABEL(.END)                                                                     \
    /* Advance to next MR slab */                                                   \
    MOV(VAR(ps_a8), RBX)                                                            \
    ADD(RBX, R10)                                                                   \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    DEC(R11)                                                                        \
    JNE(.CMLOOP)                                                                    \
                                                                                    \
    /* =================================================================== */      \
    /* m_left dispatch ladder (full ZGEMM_12x4m mirror). At entry R10 points */    \
    /* at the start of the m_left A row-slab (last m_iter step added ps_a8) */     \
    /* and R12 points at the matching C row-slab. */                                \
    /* */                                                                           \
    /* All 5 cases (1..7, 8, 9..15, 16, 17..23) are handled INLINE here. */         \
    /* */                                                                           \
    /* m_left == 0   : nothing to do, JMP CONCLUDE. */                              \
    /* m_left in 1..7  -> CGEMM_8MASKX4_BODY  (INLINE, masked 1 zmm/col). */        \
    /* m_left == 8     -> CGEMM_8X4_EDGE_BODY  (INLINE, full   1 zmm/col). */       \
    /* m_left in 9..15 -> CGEMM_16MASKX4_BODY (INLINE, masked 2 zmm/col). */        \
    /* m_left == 16    -> CGEMM_16X4_EDGE_BODY (INLINE, full   2 zmm/col). */       \
    /* m_left in 17..23-> CGEMM_24MASKX4_BODY (INLINE, masked 3 zmm/col). */        \
    /* =================================================================== */      \
    LABEL(.CMLEFT)                                                                  \
    MOV(VAR(m_left), R11)                                                           \
    TEST(R11, R11)                                                                  \
    JZ(.CONCLUDE)                                                                   \
                                                                                    \
    /* Reset per-iteration pointers for the edge body (shared by all edges) */     \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    /* Load the 1..7-row mask into k(2); it's harmless when unused. */              \
    /* Use R8D here - using EDI would clobber RDI (= sizeof(scomplex)*rs_c) set */  \
    /* by the outer CGEMM_24x4_MAIN_BODY, which the inline edge bodies still need */\
    /* for their row-store ADD(RDI, RCX) advances. R8 is a scratch register that */ \
    /* is re-initialised inside every edge body's k-loop, so it is safe here. */    \
    MOV(VAR(m_load_mask), R8D)                                                      \
    KMOVW(R8D, k(2))                                                                \
                                                                                    \
    /* Ladder : compare against {8, 9, 16, 17} thresholds. */                       \
    CMP(IMM(0x08), R11)                                                             \
    JZ(.EDGE8XN)                                                                    \
    JG(.EDGE_OVER_8)             /* m_left >= 9 */                                  \
                                                                                    \
    /* m_left in [1,7] : 8MASKx4 with k(2) zero-mask on the top zmm. */             \
    CGEMM_8MASKX4_BODY(MT_8MASK, CL)                                                \
                                                                                    \
    LABEL(.EDGE8XN)                                                                 \
    /* m_left == 8 : full 8x4 (no mask needed). */                                  \
    CGEMM_8X4_EDGE_BODY(MT_8, CL)                                                   \
                                                                                    \
    LABEL(.EDGE_OVER_8)                                                             \
    CMP(IMM(0x10), R11)                                                             \
    JZ(.EDGE16XN)                                                                   \
    JG(.EDGE_OVER_16)            /* m_left >= 17 */                                 \
                                                                                    \
    /* m_left in [9,15] : 16MASKx4 with k(2) zero-mask on the top zmm. */           \
    CGEMM_16MASKX4_BODY(MT_16MASK, CL)                                              \
                                                                                    \
    LABEL(.EDGE16XN)                                                                \
    /* m_left == 16 : full 16x4 (no mask needed). */                                \
    CGEMM_16X4_EDGE_BODY(MT_16, CL)                                                 \
                                                                                    \
    LABEL(.EDGE_OVER_16)                                                            \
    /* m_left in [17,23] : 24MASKx4 with k(2) zero-mask on the top zmm. */          \
    CGEMM_24MASKX4_BODY(MT_24MASK, CL)                                              \
                                                                                    \
    LABEL(.CONCLUDE)                                                                \

/*
   All m_left fringe cases (1..23) are now handled INLINE inside
   CGEMM_24x4_MAIN_BODY's .CMLEFT dispatch (8MASKx4, 8x4, 16MASKx4,
   16x4, 24MASKx4 bodies). No C-level m_left dispatch helper is
   needed anymore - this is the canonical ZGEMM 12x4m mirror.
*/

void bli_cgemmsup_cv_zen4_asm_24x4m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Obtaining the panel stride for A, in case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    /*
       Mask used by the masked inline edge bodies (8MASKx4 for
       m_left in [1, 7], 16MASKx4 for [9, 15], 24MASKx4 for [17, 23]).
       Mirrors the formula used by the standalone cgemm fx4 kernel
       (2 mask bits per complex float; mod 8 since each ZMM holds 8
       complex floats and only the TOP zmm of each edge body needs
       masking). Defaults to zero for the m_left == 0 / m_left a
       multiple of 8 cases (then asm uses unmasked stores via the
       8x4/16x4 full edge bodies, so the value is unused).
    */
    uint16_t m_load_mask =
        ( m_left == 0 )
            ? (uint16_t)0
            : (uint16_t)( ( (uint32_t)1 << ( 2 * ( m_left % 8 ) ) ) - 1u );

    const float value = 1.0f; // To be broadcasted and used for complex arithmetic
    const float *v = &value;

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0) // (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0) // (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    /*
       Pointers into the file-scope conj_arr tables. Exposed to the
       inline asm via the [conja_array] / [conjb_array] operand names
       in the END_ASM operand list of each conj-specific branch below.
       Mirrors the (double *)conja_array pattern used in z12x4m.
    */
    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    /*
       JR loop setup. n_iter NR-wide blocks are processed by the asm body
       inside each conj-variant branch below; n_left columns are dispatched
       to the 24x{1,2,3}m edge kernels after all four conj branches.
       Mirrors the canonical zgemm_12x4m layout.
    */
    uint64_t n_iter = (uint64_t)n0 / NR;
    scomplex *b_buf = b;
    scomplex *c_buf = c;
    scomplex *a_ref = a;
    dim_t iter = 0;
    uint64_t m_iter_ref = (uint64_t)m0 / MR;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
      for(; iter < n_iter; iter++)
      {
        b = b_buf + iter * NR * cs_b0;
        c = c_buf + iter * NR * cs_c0;
        a = a_ref;
        m_iter = m_iter_ref;

        if ( m_iter > 0 || m_left > 0 )
        {
        BEGIN_ASM()
        CGEMM_24x4_MAIN_BODY(MICRO_TILE_24x4_CONJA_CONJB, CONJ_LOAD_AB,
                             MICRO_TILE_fx4_CONJA_CONJB,
                             MICRO_TILE_8x4_CONJA_CONJB,
                             MICRO_TILE_16x4_MASK_CONJA_CONJB,
                             MICRO_TILE_16x4_CONJA_CONJB,
                             MICRO_TILE_24x4_MASK_CONJA_CONJB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [m_load_mask] "m" (m_load_mask),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2",
          "memory"
        )
        }
      }
    }
    else if ( bli_is_conj( conja ) )
    {
      for(; iter < n_iter; iter++)
      {
        b = b_buf + iter * NR * cs_b0;
        c = c_buf + iter * NR * cs_c0;
        a = a_ref;
        m_iter = m_iter_ref;

        if ( m_iter > 0 || m_left > 0 )
        {
        BEGIN_ASM()
        CGEMM_24x4_MAIN_BODY(MICRO_TILE_24x4_CONJA, CONJ_LOAD_A,
                             MICRO_TILE_fx4_CONJA,
                             MICRO_TILE_8x4_CONJA,
                             MICRO_TILE_16x4_MASK_CONJA,
                             MICRO_TILE_16x4_CONJA,
                             MICRO_TILE_24x4_MASK_CONJA)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [m_load_mask] "m" (m_load_mask),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2",
          "memory"
        )
        }
      }
    }
    else if ( bli_is_conj( conjb ) )
    {
      for(; iter < n_iter; iter++)
      {
        b = b_buf + iter * NR * cs_b0;
        c = c_buf + iter * NR * cs_c0;
        a = a_ref;
        m_iter = m_iter_ref;

        if ( m_iter > 0 || m_left > 0 )
        {
        BEGIN_ASM()
        CGEMM_24x4_MAIN_BODY(MICRO_TILE_24x4_CONJB, CONJ_LOAD_B,
                             MICRO_TILE_fx4_CONJB,
                             MICRO_TILE_8x4_CONJB,
                             MICRO_TILE_16x4_MASK_CONJB,
                             MICRO_TILE_16x4_CONJB,
                             MICRO_TILE_24x4_MASK_CONJB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [m_load_mask] "m" (m_load_mask),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2",
          "memory"
        )
        }
      }
    }
    else
    {
      for(; iter < n_iter; iter++)
      {
        b = b_buf + iter * NR * cs_b0;
        c = c_buf + iter * NR * cs_c0;
        a = a_ref;
        m_iter = m_iter_ref;

        if ( m_iter > 0 || m_left > 0 )
        {
        BEGIN_ASM()
        CGEMM_24x4_MAIN_BODY(MICRO_TILE_24x4, CONJ_LOAD_NN,
                             MICRO_TILE_fx4,
                             MICRO_TILE_8x4,
                             MICRO_TILE_16x4_MASK,
                             MICRO_TILE_16x4,
                             MICRO_TILE_24x4_MASK)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [m_load_mask] "m" (m_load_mask),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2",
          "memory"
        )
        }
      }
    }

    /*
       n_left dispatch. After the JR loop above, iter == n_iter and
       iter * NR is the column offset of the n-fringe. Forwards
       conja/conjb to the 24x{3,2,1}m edge kernels exactly as
       zgemm_12x4m does at the bottom of its function.
    */
    uint64_t n_left = (uint64_t)n0 % NR;
    if ( n_left )
    {
      scomplex* cij = c_buf + iter * NR * cs_c0;
      scomplex* bj  = b_buf + iter * NR * cs_b0;
      scomplex* ai  = a_ref;

      if ( 3 == n_left )
      {
        const dim_t nr_cur = 3;
        bli_cgemmsup_cv_zen4_asm_24x3m(conja, conjb, m0, nr_cur, k0,
             alpha, ai, rs_a0, cs_a0,
             bj, rs_b0, cs_b0, beta,
             cij, rs_c0, cs_c0,
             data, cntx);
      }
      if ( 2 == n_left )
      {
        const dim_t nr_cur = 2;
        bli_cgemmsup_cv_zen4_asm_24x2m(conja, conjb, m0, nr_cur, k0,
             alpha, ai, rs_a0, cs_a0,
             bj, rs_b0, cs_b0, beta,
             cij, rs_c0, cs_c0,
             data, cntx);
      }
      if ( 1 == n_left )
      {
        const dim_t nr_cur = 1;
        bli_cgemmsup_cv_zen4_asm_24x1m(conja, conjb, m0, nr_cur, k0,
             alpha, ai, rs_a0, cs_a0,
             bj, rs_b0, cs_b0, beta,
             cij, rs_c0, cs_c0,
             data, cntx);
      }
    }
}

/*
   CGEMM_24x3_MAIN_BODY(MT, CL)

   Parameterized inline asm body for the 24x3 single-precision complex
   GEMM sup kernel. MT is the per-conj micro-tile macro
   (MICRO_TILE_24x3, MICRO_TILE_24x3_CONJA, _CONJB, _CONJA_CONJB).
   CL is the conj-array reload macro (CONJ_LOAD_NN/A/B/AB) injected
   after RESET_REGISTERS so ZMM(30)/ZMM(31) hold the appropriate
   sign-flip pattern for the imaginary lanes.

   Layout matches the original monolithic 24x3 body so the four conj
   variants are bytewise siblings.
*/
#define CGEMM_24x3_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(VAR(m_iter), R11)                                                           \
    LABEL(.CMLOOP)                                                                  \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8, 10)                                                               \
    PERMUTE(12, 14, 16)                                                             \
    PERMUTE(18, 20, 22)                                                             \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                      \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                 \
    ACC_COL(17, 18, 19, 20, 21, 22)                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8, 10)                                                       \
    ALPHA_MINUS_ONE(12, 14, 16)                                                     \
    ALPHA_MINUS_ONE(18, 20, 22)                                                     \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8, 10)                                                         \
    ALPHA_GENERIC(12, 14, 16)                                                       \
    ALPHA_GENERIC(18, 20, 22)                                                       \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20, 21, 22)                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)                                            \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18, 19, 20, 21, 22)                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)                                                \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18, 19, 20, 21, 22)                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(10), MEM(RCX, 128))                                                 \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))                                         \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    VMOVUPS(ZMM(22), MEM(R9, 128))                                                  \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)                                  \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)                                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
                                                                                    \
    LABEL(.END)                                                                     \
    MOV(VAR(ps_a8), RBX)                                                            \
    ADD(RBX, R10)                                                                   \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    DEC(R11)                                                                        \
    JNE(.CMLOOP)                                                                    \

void bli_cgemmsup_cv_zen4_asm_24x3m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Obtaining the panel stride for A, in case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x3 kernels

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
    */
    uint16_t trans_load_mask = 0x3F;
    if ( m_iter == 0 ) goto consider_edge_cases;

    const float value = 1.0f; // To be broadcasted and used for complex arithmetic
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    /* Pointers into the file-scope conj_arr tables. */
    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x3_MAIN_BODY(MICRO_TILE_24x3_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_24x3_MAIN_BODY(MICRO_TILE_24x3_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x3_MAIN_BODY(MICRO_TILE_24x3_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_24x3_MAIN_BODY(MICRO_TILE_24x3, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }

    consider_edge_cases:;
    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
      const dim_t      i_edge = m0 - ( dim_t )m_left;

      scomplex* restrict cij = c + i_edge * rs_c;
      scomplex* restrict ai  = a + m_iter * ps_a;
      scomplex* restrict bj  = b;

      if (16 <= m_left)
      {
        const dim_t      mr_cur = 16;
        bli_cgemmsup_cv_zen4_asm_16x3(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (8 <= m_left)
      {
        const dim_t      mr_cur = 8;
        bli_cgemmsup_cv_zen4_asm_8x3(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (1 <= m_left)
      {
        const dim_t      mr_cur = m_left;
        bli_cgemmsup_cv_zen4_asm_fx3(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
      }
    }
}

/*
   CGEMM_24x2_MAIN_BODY(MT, CL)

   Parameterized inline asm body for the 24x2 single-precision complex
   GEMM sup kernel. Mirrors the 24x3 layout but with 6 accumulator
   registers split as two columns and three row chunks of 8 elements
   each, and a k(3) opmask of 0xF for the 2-column row-store path
   (BETA_GEN_ROW_4xf masks the unused lanes).
*/
#define CGEMM_24x2_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(VAR(m_iter), R11)                                                           \
    LABEL(.CMLOOP)                                                                  \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8, 10)                                                               \
    PERMUTE(12, 14, 16)                                                             \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                      \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8, 10)                                                       \
    ALPHA_MINUS_ONE(12, 14, 16)                                                     \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8, 10)                                                         \
    ALPHA_GENERIC(12, 14, 16)                                                       \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)                                            \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)                                                \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(10), MEM(RCX, 128))                                                 \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)                                  \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)                                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
                                                                                    \
    LABEL(.END)                                                                     \
    MOV(VAR(ps_a8), RBX)                                                            \
    ADD(RBX, R10)                                                                   \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    DEC(R11)                                                                        \
    JNE(.CMLOOP)                                                                    \

void bli_cgemmsup_cv_zen4_asm_24x2m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR;
    uint64_t m_left = m0 % MR;

    uint16_t trans_load_mask = 0xF;
    if ( m_iter == 0 ) goto consider_edge_cases;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x2_MAIN_BODY(MICRO_TILE_24x2_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_24x2_MAIN_BODY(MICRO_TILE_24x2_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x2_MAIN_BODY(MICRO_TILE_24x2_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_24x2_MAIN_BODY(MICRO_TILE_24x2, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }

    consider_edge_cases:;
    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
      const dim_t      i_edge = m0 - ( dim_t )m_left;

      scomplex* restrict cij = c + i_edge * rs_c;
      scomplex* restrict ai  = a + m_iter * ps_a;
      scomplex* restrict bj  = b;

      if (16 <= m_left)
      {
        const dim_t      mr_cur = 16;
        bli_cgemmsup_cv_zen4_asm_16x2(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (8 <= m_left)
      {
        const dim_t      mr_cur = 8;
        bli_cgemmsup_cv_zen4_asm_8x2(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (1 <= m_left)
      {
        const dim_t      mr_cur = m_left;
        bli_cgemmsup_cv_zen4_asm_fx2(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
      }
    }
}

/*
   CGEMM_24x1_MAIN_BODY(MT, CL)

   24x1 single-precision complex GEMM sup body. Only one column of B,
   so 3 accumulators split into three row chunks of 8. Row-store
   transposes three 8x1 chunks (extended to 8x8) and writes one row
   per pair under k(3)=0x3.
*/
#define CGEMM_24x1_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(VAR(m_iter), R11)                                                           \
    LABEL(.CMLOOP)                                                                  \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8, 10)                                                               \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                      \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8, 10)                                                       \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8, 10)                                                         \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)                                            \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)                                                \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(10), MEM(RCX, 128))                                                 \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)                                  \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,                                    \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)                                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
                                                                                    \
    LABEL(.END)                                                                     \
    MOV(VAR(ps_a8), RBX)                                                            \
    ADD(RBX, R10)                                                                   \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    LEA(MEM(R12, RDI, 8), R12)                                                      \
    DEC(R11)                                                                        \
    JNE(.CMLOOP)                                                                    \

void bli_cgemmsup_cv_zen4_asm_24x1m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR;
    uint64_t m_left = m0 % MR;

    uint16_t trans_load_mask = 0x3;
    if ( m_iter == 0 ) goto consider_edge_cases;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x1_MAIN_BODY(MICRO_TILE_24x1_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_24x1_MAIN_BODY(MICRO_TILE_24x1_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_24x1_MAIN_BODY(MICRO_TILE_24x1_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_24x1_MAIN_BODY(MICRO_TILE_24x1, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_left]  "m" (m_left),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [ps_a8]   "m" (ps_a8),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }

    consider_edge_cases:;
    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
      const dim_t      i_edge = m0 - ( dim_t )m_left;

      scomplex* restrict cij = c + i_edge * rs_c;
      scomplex* restrict ai  = a + m_iter * ps_a;
      scomplex* restrict bj  = b;

      if (16 <= m_left)
      {
        const dim_t      mr_cur = 16;
        bli_cgemmsup_cv_zen4_asm_16x1(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (8 <= m_left)
      {
        const dim_t      mr_cur = 8;
        bli_cgemmsup_cv_zen4_asm_8x1(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
        cij += mr_cur * rs_c; ai += mr_cur * rs_a;
        m_left -= mr_cur;
      }
      if (1 <= m_left)
      {
        const dim_t      mr_cur = m_left;
        bli_cgemmsup_cv_zen4_asm_fx1(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
      }
    }
}

/*
   CGEMM_16x4_MAIN_BODY(MT, CL)

   Parameterized inline asm body for the 16x4 single-precision complex
   GEMM sup kernel. MT is the per-conj micro-tile macro
   (MICRO_TILE_16x4, MICRO_TILE_16x4_CONJA, _CONJB, _CONJA_CONJB).
   CL is the conj-array reload macro (CONJ_LOAD_NN/A/B/AB) injected
   after RESET_REGISTERS so ZMM(30)/ZMM(31) hold the appropriate
   sign-flip pattern for the imaginary lanes.

   Layout is intentionally identical to the original monolithic 16x4
   body so the four conj variants are bytewise siblings.
*/
#define CGEMM_16x4_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)               /* R10 = base addr of A (MCXKC block) */         \
    MOV(VAR(b), RDX)               /* RDX = base addr of B (KCXNR block) */         \
    MOV(VAR(c), R12)               /* R12 = base addr of C (MCxNR block) */         \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)        /* R13 = sizeof(scomplex)*cs_a */                \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)        /* R14 = sizeof(scomplex)*rs_b */                \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)        /* R15 = sizeof(scomplex)*cs_b */                \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)        /* RDI = sizeof(scomplex)*rs_c */                \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)        /* RSI = sizeof(scomplex)*cs_c */                \
                                                                                    \
    /* Intermediate register for complex arithmetic */                              \
    MOV(VAR(v), R9)                /* Used in fmaddsub instruction */               \
    VBROADCASTSS(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */              \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    /* Reset all scratch + accum registers (also clobbers ZMM(30)/ZMM(31)) */       \
    RESET_REGISTERS                                                                 \
                                                                                    \
    /* Reload conja/conjb arrays into ZMM(30)/ZMM(31) for the current conj path */  \
    CL                                                                              \
                                                                                    \
    /* Setting iterator for k */                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
                                                                                    \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    /* Remainder loop for k */                                                      \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    /* Accumulating A*B over 8 registers */                                         \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8)                                                                   \
    PERMUTE(12, 14)                                                                 \
    PERMUTE(18, 20)                                                                 \
    PERMUTE(24, 26)                                                                 \
    ACC_COL(5, 6, 7, 8)                                                             \
    ACC_COL(11, 12, 13, 14)                                                         \
    ACC_COL(17, 18, 19, 20)                                                         \
    ACC_COL(23, 24, 25, 26)                                                         \
                                                                                    \
    /* Alpha scaling */                                                             \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    ALPHA_MINUS_ONE(12, 14)                                                         \
    ALPHA_MINUS_ONE(18, 20)                                                         \
    ALPHA_MINUS_ONE(24, 26)                                                         \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
    ALPHA_GENERIC(12, 14)                                                           \
    ALPHA_GENERIC(18, 20)                                                           \
    ALPHA_GENERIC(24, 26)                                                           \
                                                                                    \
    /* Beta scaling */                                                              \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)                                                 \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20)                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 23, 24, 25, 26)                                             \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                               \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18, 19, 20)                                               \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 23, 24, 25, 26)                                               \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18, 19, 20)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 23, 24, 25, 26)                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))                                           \
    JMP(.END)                                                                       \
                                                                                    \
    /* Beta scaling when C is row stored */                                         \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_16x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    const float value = 1.0f; // To be broadcasted and used for complex arithmetic
    const float *v = &value;

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0) // (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0) // (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    /*
       Pointers into the file-scope conj_arr tables. Exposed to the
       inline asm via the [conja_array] / [conjb_array] operand names
       in the END_ASM operand list of each conj-specific branch below.
       Mirrors the (double *)conja_array pattern used in z12x4m.
    */
    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x4_MAIN_BODY(MICRO_TILE_16x4_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_16x4_MAIN_BODY(MICRO_TILE_16x4_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x4_MAIN_BODY(MICRO_TILE_16x4_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_16x4_MAIN_BODY(MICRO_TILE_16x4, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
}

/*
   CGEMM_16x3_MAIN_BODY(MT, CL)
   16x3 single-precision complex GEMM sup body, parameterized by
   conj variant. Single CKMAINLOOP + CKLEFT, 6 accum regs, row-store
   uses 2 chunks of TRANSPOSE_8x8 + BETA_GEN_ROW_4xf (k(3) masks the
   3-col store).
*/
#define CGEMM_16x3_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8)                                                                   \
    PERMUTE(12, 14)                                                                 \
    PERMUTE(18, 20)                                                                 \
    ACC_COL(5, 6, 7, 8)                                                             \
    ACC_COL(11, 12, 13, 14)                                                         \
    ACC_COL(17, 18, 19, 20)                                                         \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    ALPHA_MINUS_ONE(12, 14)                                                         \
    ALPHA_MINUS_ONE(18, 20)                                                         \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
    ALPHA_GENERIC(12, 14)                                                           \
    ALPHA_GENERIC(18, 20)                                                           \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)                                                 \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20)                                             \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                               \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18, 19, 20)                                               \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18, 19, 20)                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(20), MEM(R9, 64))                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

/* 4-way conj dispatch shim for the 16x3 sup kernel. */
#define CGEMM_16x3_END_OPERANDS_NN                                                  \
        : /* output operands (none) */                                              \
        : /* input operands */                                                      \
          [v]  "m" (v),                                                             \
          [k_iter]  "m" (k_iter),                                                   \
          [k_left]  "m" (k_left),                                                   \
          [trans_load_mask]  "m" (trans_load_mask),                                 \
          [alpha_mul_type]  "m" (alpha_mul_type),                                   \
          [beta_mul_type]   "m" (beta_mul_type),                                    \
          [alpha]  "m" (alpha),                                                     \
          [a]      "m" (a),                                                         \
          [b]      "m" (b),                                                         \
          [beta]   "m" (beta),                                                      \
          [c]      "m" (c),                                                         \
          [cs_a]   "m" (cs_a),                                                      \
          [rs_b]   "m" (rs_b),                                                      \
          [cs_b]   "m" (cs_b),                                                      \
          [rs_c]   "m" (rs_c),                                                      \
          [cs_c]   "m" (cs_c)

#define CGEMM_x3_CLOBBERS                                                           \
        : /* register clobber list */                                               \
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al", \
          "zmm0", "zmm1", "zmm2", "zmm3",                                           \
          "zmm4", "zmm5", "zmm6", "zmm7",                                           \
          "zmm8", "zmm9", "zmm10", "zmm11",                                         \
          "zmm12", "zmm13", "zmm14", "zmm15",                                       \
          "zmm16", "zmm17", "zmm18", "zmm19",                                       \
          "zmm20", "zmm21", "zmm22", "zmm23",                                       \
          "zmm24", "zmm25", "zmm26", "zmm27",                                       \
          "zmm28", "zmm29", "zmm30", "zmm31",                                       \
          "k3", "memory"

void bli_cgemmsup_cv_zen4_asm_16x3
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0x3F;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x3_MAIN_BODY(MICRO_TILE_16x3_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_16x3_MAIN_BODY(MICRO_TILE_16x3_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x3_MAIN_BODY(MICRO_TILE_16x3_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_16x3_MAIN_BODY(MICRO_TILE_16x3, CONJ_LOAD_NN)
        END_ASM(
        CGEMM_16x3_END_OPERANDS_NN
        CGEMM_x3_CLOBBERS
        )
    }
}

/*
   CGEMM_16x2_MAIN_BODY(MT, CL)
   16x2 single-precision complex GEMM sup body. 4 accum regs in two
   columns. Row-store uses two TRANSPOSE_8x8 + four BETA_GEN_ROW_4xf
   calls under k(3)=0xF.
*/
#define CGEMM_16x2_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8)                                                                   \
    PERMUTE(12, 14)                                                                 \
    ACC_COL(5, 6, 7, 8)                                                             \
    ACC_COL(11, 12, 13, 14)                                                         \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    ALPHA_MINUS_ONE(12, 14)                                                         \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
    ALPHA_GENERIC(12, 14)                                                           \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)                                                 \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)                                             \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                                   \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                               \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12, 13, 14)                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_16x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0xF;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x2_MAIN_BODY(MICRO_TILE_16x2_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_16x2_MAIN_BODY(MICRO_TILE_16x2_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x2_MAIN_BODY(MICRO_TILE_16x2_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_16x2_MAIN_BODY(MICRO_TILE_16x2, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
}

/*
   CGEMM_16x1_MAIN_BODY(MT, CL)
   16x1 single-precision complex GEMM sup body. 2 accum regs in a single
   column. Row-store uses two TRANSPOSE_8x8 + four BETA_GEN_ROW_4xf
   calls under k(3)=0x3.
*/
#define CGEMM_16x1_MAIN_BODY(MT, CL)                                                \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6, 8)                                                                   \
    ACC_COL(5, 6, 7, 8)                                                             \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6, 8)                                                           \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6, 8)                                                             \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)                                                 \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6, 7, 8)                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(8), MEM(RCX, 64))                                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_16x1
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0x3;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x1_MAIN_BODY(MICRO_TILE_16x1_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_16x1_MAIN_BODY(MICRO_TILE_16x1_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_16x1_MAIN_BODY(MICRO_TILE_16x1_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_16x1_MAIN_BODY(MICRO_TILE_16x1, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
}

/*
   CGEMM_8x4_MAIN_BODY(MT, CL)

   Parameterized inline asm body for the 8x4 single-precision complex
   GEMM sup kernel. MT is the per-conj micro-tile macro
   (MICRO_TILE_8x4, MICRO_TILE_8x4_CONJA, _CONJB, _CONJA_CONJB).
   CL is the conj-array reload macro (CONJ_LOAD_NN/A/B/AB) injected
   after RESET_REGISTERS so ZMM(30)/ZMM(31) hold the appropriate
   sign-flip pattern for the imaginary lanes.
*/
#define CGEMM_8x4_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    PERMUTE(24)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
    ACC_COL(23, 24)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    ALPHA_MINUS_ONE(24)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
    ALPHA_GENERIC(24)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12)                                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18)                                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 23, 24)                                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6)                                                         \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 23, 24)                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6)                                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12)                                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18)                                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 23, 24)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))                                               \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_8x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    const float value = 1.0f; // To be broadcasted and used for complex arithmetic
    const float *v = &value;

    // Assigning the type of alpha and beta scaling
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0) // (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0) // (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    /*
       Pointers into the file-scope conj_arr tables. Exposed to the
       inline asm via the [conja_array] / [conjb_array] operand names
       in the END_ASM operand list of each conj-specific branch below.
    */
    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x4_MAIN_BODY(MICRO_TILE_8x4_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_8x4_MAIN_BODY(MICRO_TILE_8x4_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x4_MAIN_BODY(MICRO_TILE_8x4_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_8x4_MAIN_BODY(MICRO_TILE_8x4, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm22",
          "ymm23", "ymm24", "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "memory"
        )
    }
}

/*
   CGEMM_8x3_MAIN_BODY(MT, CL)
   8x3 single-precision complex GEMM sup body. 3 accum regs, row-store
   uses a single TRANSPOSE_8x8 + two BETA_GEN_ROW_4xf calls under k(3).
*/
#define CGEMM_8x3_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12)                                                     \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 17, 18)                                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6)                                                         \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 17, 18)                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6)                                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12)                                                           \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 17, 18)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    VMOVUPS(ZMM(18), MEM(R9))                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_8x3
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0x3F;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x3_MAIN_BODY(MICRO_TILE_8x3_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_8x3_MAIN_BODY(MICRO_TILE_8x3_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x3_MAIN_BODY(MICRO_TILE_8x3_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_8x3_MAIN_BODY(MICRO_TILE_8x3, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
}

/*
   CGEMM_8x2_MAIN_BODY(MT, CL)
   8x2 single-precision complex GEMM sup body. 2 accum regs; row-store
   does one TRANSPOSE_8x8 + two BETA_GEN_ROW_4xf under k(3)=0xF.
*/
#define CGEMM_8x2_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6)                                                       \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE(RCX, 11, 12)                                                     \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6)                                                         \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC(RCX, 11, 12)                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6)                                                             \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE(RCX, 11, 12)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))                                              \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_8x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0xF;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x2_MAIN_BODY(MICRO_TILE_8x2_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_8x2_MAIN_BODY(MICRO_TILE_8x2_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x2_MAIN_BODY(MICRO_TILE_8x2_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_8x2_MAIN_BODY(MICRO_TILE_8x2, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
}

/*
   CGEMM_8x1_MAIN_BODY(MT, CL)
   8x1 single-precision complex GEMM sup body. 1 accum reg in a single
   column. Row-store uses one TRANSPOSE_8x8 + two BETA_GEN_ROW_4xf
   calls under k(3)=0x3.
*/
#define CGEMM_8x1_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    ACC_COL(5, 6)                                                                   \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE(RCX, 5, 6)                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC(RCX, 5, 6)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE(RCX, 5, 6)                                                             \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    VMOVUPS(ZMM(6), MEM(RCX))                                                       \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    MOV(RCX, R9)                                                                    \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)                                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    LEA(MEM(R9, RDI, 2), R9)                                                        \
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                      \
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)                                                \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_8x1
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint16_t trans_load_mask = 0x3;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x1_MAIN_BODY(MICRO_TILE_8x1_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_8x1_MAIN_BODY(MICRO_TILE_8x1_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_8x1_MAIN_BODY(MICRO_TILE_8x1_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_8x1_MAIN_BODY(MICRO_TILE_8x1, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k3", "memory"
        )
    }
}

/*
   CGEMM_fx4_MAIN_BODY(MT, CL)

   Parameterized inline asm body for the fx4 single-precision complex
   GEMM sup kernel which handles m_left in [1,7] via opmask k(2).
   MT is the per-conj micro-tile macro (MICRO_TILE_fx4, _CONJA,
   _CONJB, _CONJA_CONJB). CL is the conj-array reload macro injected
   after RESET_REGISTERS so ZMM(30)/ZMM(31) hold the appropriate
   sign-flip pattern for the imaginary lanes.

   Row-storage epilogue dispatches on m_store_row (= m0) to write
   exactly m0 transposed rows back to C via BETA_GEN_ROW_1x4 /
   BETA_ZERO_ROW_1x4.
*/
#define CGEMM_fx4_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(m_load_mask), EBX)                                                      \
    KMOVW(EBX, k(2))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    PERMUTE(24)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
    ACC_COL(23, 24)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    ALPHA_MINUS_ONE(24)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
    ALPHA_GENERIC(24)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE_fC(RCX, 5, 6)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 11, 12)                                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 17, 18)                                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 23, 24)                                                  \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_fC(RCX, 5, 6)                                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 11, 12)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 17, 18)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 23, 24)                                                    \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE_fC(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 11, 12)                                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 17, 18)                                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 23, 24)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                  \
    VMOVUPS(ZMM(18), MEM(R9) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(24), MEM(R9, RSI, 1) MASK_(k(2)))                                   \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.SCALE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.SCALE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.SCALE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.SCALE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.SCALE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.SCALE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.SCALE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_7)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(25, 17)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_6)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(23, 11)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_5)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(21, 5)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_4)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(15, 24)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_3)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(13, 18)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_2)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1x4(9, 12)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_1)                                                         \
    BETA_GEN_ROW_1x4(7, 6)                                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.STORE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.STORE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.STORE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.STORE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.STORE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.STORE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.STORE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.STORE_ROW_GEN_7)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(17)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_6)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(11)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_5)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(5)                                                            \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_4)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(24)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_3)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(18)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_2)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1x4(12)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_1)                                                         \
    BETA_ZERO_ROW_1x4(6)                                                            \
                                                                                    \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_fx4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    const float value = 1.0f; // To be broadcasted and used for complex arithmetic
    const float *v = &value;

    /*
      The mask bits below are set for ensuring fx4 compatability
      while transposing, and loading/storing C(k(2) mask register).
      This mask is set based on the m-value(m0) that the kernel receives.
      m0 is guaranteed to be less than 8.
    */
    uint64_t m_store_row = m0; // Also used when handling row-storage of C(post transpose)
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    // Assigning the type of alpha and beta scaling
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0) // (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0) // (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    /*
       Pointers into the file-scope conj_arr tables. Exposed to the
       inline asm via the [conja_array] / [conjb_array] operand names
       in the END_ASM operand list of each conj-specific branch below.
    */
    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx4_MAIN_BODY(MICRO_TILE_fx4_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_load_mask] "m" (m_load_mask),
          [m_store_row] "m" (m_store_row),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm21",
          "ymm22", "ymm23", "ymm24", "ymm25",
          "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_fx4_MAIN_BODY(MICRO_TILE_fx4_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_load_mask] "m" (m_load_mask),
          [m_store_row] "m" (m_store_row),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm21",
          "ymm22", "ymm23", "ymm24", "ymm25",
          "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx4_MAIN_BODY(MICRO_TILE_fx4_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_load_mask] "m" (m_load_mask),
          [m_store_row] "m" (m_store_row),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm21",
          "ymm22", "ymm23", "ymm24", "ymm25",
          "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_fx4_MAIN_BODY(MICRO_TILE_fx4, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_load_mask] "m" (m_load_mask),
          [m_store_row] "m" (m_store_row),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "al",
          "ymm5", "ymm6", "ymm7", "ymm8",
          "ymm9", "ymm10", "ymm11", "ymm12",
          "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm20", "ymm21",
          "ymm22", "ymm23", "ymm24", "ymm25",
          "ymm26", "ymm28",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "memory"
        )
    }
}

/*
   CGEMM_fx3_MAIN_BODY(MT, CL)
   fx3 single-precision complex GEMM sup body, parameterized by conj
   variant. Column-store path uses k(2) for the m_load_mask. Row-store
   path has a 7-way ladder of fall-through labels keyed on m_store_row.
*/
#define CGEMM_fx3_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(m_load_mask), EBX)                                                      \
    KMOVW(EBX, k(2))                                                                \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    PERMUTE(18)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
    ACC_COL(17, 18)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    ALPHA_MINUS_ONE(18)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
    ALPHA_GENERIC(18)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE_fC(RCX, 5, 6)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 11, 12)                                                  \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 17, 18)                                                  \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_fC(RCX, 5, 6)                                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 11, 12)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 17, 18)                                                    \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE_fC(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 11, 12)                                                        \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 17, 18)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                  \
    VMOVUPS(ZMM(18), MEM(R9) MASK_(k(2)))                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.SCALE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.SCALE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.SCALE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.SCALE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.SCALE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.SCALE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.SCALE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_7)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(25, 17)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_6)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_5)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_4)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_3)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_2)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_1)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.STORE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.STORE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.STORE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.STORE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.STORE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.STORE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.STORE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.STORE_ROW_GEN_7)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(17)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_6)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_5)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_4)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_3)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_2)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_1)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
                                                                                    \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_fx3
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_store_row = m0;
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    uint16_t trans_load_mask = 0x3F;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx3_MAIN_BODY(MICRO_TILE_fx3_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_fx3_MAIN_BODY(MICRO_TILE_fx3_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx3_MAIN_BODY(MICRO_TILE_fx3_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_fx3_MAIN_BODY(MICRO_TILE_fx3, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
}

/*
   CGEMM_fx2_MAIN_BODY(MT, CL)
   fx2 single-precision complex GEMM sup body. Column-store uses k(2)
   m-mask; row-store transposes to 8 lanes and ladder-stores
   m_store_row of them.
*/
#define CGEMM_fx2_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(m_load_mask), EBX)                                                      \
    KMOVW(EBX, k(2))                                                                \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    PERMUTE(12)                                                                     \
    ACC_COL(5, 6)                                                                   \
    ACC_COL(11, 12)                                                                 \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    ALPHA_MINUS_ONE(12)                                                             \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
    ALPHA_GENERIC(12)                                                               \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE_fC(RCX, 5, 6)                                                    \
    ADD(RSI, RCX)                                                                   \
    BETA_MINUS_ONE_fC(RCX, 11, 12)                                                  \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_fC(RCX, 5, 6)                                                      \
    ADD(RSI, RCX)                                                                   \
    BETA_GENERIC_fC(RCX, 11, 12)                                                    \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE_fC(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                   \
    BETA_ONE_fC(RCX, 11, 12)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    LEA(MEM(RCX, RSI, 2), R9)                                                       \
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))                                           \
    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                  \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.SCALE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.SCALE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.SCALE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.SCALE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.SCALE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.SCALE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.SCALE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_7)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(25, 17)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_6)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_5)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_4)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_3)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_2)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_1)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.STORE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.STORE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.STORE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.STORE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.STORE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.STORE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.STORE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.STORE_ROW_GEN_7)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(17)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_6)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_5)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_4)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_3)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_2)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_1)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_fx2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_store_row = m0;
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    uint16_t trans_load_mask = 0xF;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx2_MAIN_BODY(MICRO_TILE_fx2_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_fx2_MAIN_BODY(MICRO_TILE_fx2_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx2_MAIN_BODY(MICRO_TILE_fx2_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_fx2_MAIN_BODY(MICRO_TILE_fx2, CONJ_LOAD_NN)
        END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter]  "m" (k_iter),
      [k_left]  "m" (k_left),
      [m_store_row]  "m" (m_store_row),
      [m_load_mask]  "m" (m_load_mask),
      [trans_load_mask]  "m" (trans_load_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [cs_a]   "m" (cs_a),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7",
      "zmm8", "zmm9", "zmm10", "zmm11",
      "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23",
      "zmm24", "zmm25", "zmm26", "zmm27",
      "zmm28", "zmm29", "zmm30", "zmm31",
      "k2", "k3", "memory"
    )
    }
}

/*
   CGEMM_fx1_MAIN_BODY(MT, CL)
   fx1 single-precision complex GEMM sup body. Column-store uses k(2)
   m-mask; row-store transposes to 8 lanes and ladder-stores
   m_store_row of them.
*/
#define CGEMM_fx1_MAIN_BODY(MT, CL)                                                 \
    MOV(VAR(a), R10)                                                                \
    MOV(VAR(b), RDX)                                                                \
    MOV(VAR(c), R12)                                                                \
                                                                                    \
    MOV(VAR(cs_a), R13)                                                             \
    LEA(MEM(, R13, 8), R13)                                                         \
                                                                                    \
    MOV(VAR(rs_b), R14)                                                             \
    LEA(MEM(, R14, 8), R14)                                                         \
                                                                                    \
    MOV(VAR(cs_b), R15)                                                             \
    LEA(MEM(, R15, 8), R15)                                                         \
                                                                                    \
    MOV(VAR(rs_c), RDI)                                                             \
    LEA(MEM(, RDI, 8), RDI)                                                         \
                                                                                    \
    MOV(VAR(cs_c), RSI)                                                             \
    LEA(MEM(, RSI, 8), RSI)                                                         \
                                                                                    \
    MOV(VAR(m_load_mask), EBX)                                                      \
    KMOVW(EBX, k(2))                                                                \
                                                                                    \
    MOV(VAR(trans_load_mask), EAX)                                                  \
    KMOVW(EAX, k(3))                                                                \
                                                                                    \
    MOV(VAR(v), R9)                                                                 \
    VBROADCASTSS(MEM(R9), ZMM(29))                                                  \
                                                                                    \
    MOV(R10, RAX)                                                                   \
    MOV(RDX, RBX)                                                                   \
    MOV(R12, RCX)                                                                   \
                                                                                    \
    RESET_REGISTERS                                                                 \
                                                                                    \
    CL                                                                              \
                                                                                    \
    MOV(VAR(k_iter), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.CKLEFT)                                                                     \
    LABEL(.CKMAINLOOP)                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKMAINLOOP)                                                                \
                                                                                    \
    LABEL(.CKLEFT)                                                                  \
    MOV(VAR(k_left), R8)                                                            \
    TEST(R8, R8)                                                                    \
    JE(.ACCUMULATE)                                                                 \
    LABEL(.CKLEFTLOOP)                                                              \
    MT                                                                              \
    DEC(R8)                                                                         \
    JNZ(.CKLEFTLOOP)                                                                \
                                                                                    \
    LABEL(.ACCUMULATE)                                                              \
    PERMUTE(6)                                                                      \
    ACC_COL(5, 6)                                                                   \
                                                                                    \
    MOV(VAR(alpha_mul_type), AL)                                                    \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.ALPHA_GENERAL)                                                             \
    VXORPS(ZMM(2), ZMM(2), ZMM(2))                                                  \
    ALPHA_MINUS_ONE(6)                                                              \
    JMP(.BETA_SCALE)                                                                \
                                                                                    \
    LABEL(.ALPHA_GENERAL)                                                           \
    CMP(IMM(2), AL)                                                                 \
    JNE(.BETA_SCALE)                                                                \
    MOV(VAR(alpha), RAX)                                                            \
    VBROADCASTSS(MEM(RAX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RAX, 4), ZMM(1))                                               \
    ALPHA_GENERIC(6)                                                                \
                                                                                    \
    LABEL(.BETA_SCALE)                                                              \
    CMP(IMM(8), RSI)                                                                \
    JE(.ROW_STORAGE_C)                                                              \
                                                                                    \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE)                                                                      \
    CMP(IMM(0x01), AL)                                                              \
    JE(.ADD)                                                                        \
    CMP(IMM(0xFF), AL)                                                              \
    JNE(.BETA_GENERAL)                                                              \
    BETA_MINUS_ONE_fC(RCX, 5, 6)                                                    \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.BETA_GENERAL)                                                            \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    BETA_GENERIC_fC(RCX, 5, 6)                                                      \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ADD)                                                                     \
    BETA_ONE_fC(RCX, 5, 6)                                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE)                                                                   \
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.ROW_STORAGE_C)                                                           \
    MOV(VAR(beta_mul_type), AL)                                                     \
    CMP(IMM(0), AL)                                                                 \
    JE(.STORE_ROW)                                                                  \
                                                                                    \
    LABEL(.BETA_GENERAL_ROW)                                                        \
    MOV(VAR(beta), RBX)                                                             \
    VBROADCASTSS(MEM(RBX), ZMM(0))                                                  \
    VBROADCASTSS(MEM(RBX, 4), ZMM(1))                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.SCALE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.SCALE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.SCALE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.SCALE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.SCALE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.SCALE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.SCALE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_7)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(25, 17)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_6)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(23, 11)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_5)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(21, 5)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_4)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(15, 24)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_3)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(13, 18)                                                        \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_2)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    ADD(RDI, RCX)                                                                   \
    BETA_GEN_ROW_1xf(9, 12)                                                         \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.SCALE_ROW_GEN_1)                                                         \
    BETA_GEN_ROW_1xf(7, 6)                                                          \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW)                                                               \
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,                                     \
                  7, 9, 13, 15, 21, 23, 25, 27)                                     \
    MOV(VAR(m_store_row), R8)                                                       \
    CMP(IMM(7), R8)                                                                 \
    JE(.STORE_ROW_GEN_7)                                                            \
    CMP(IMM(6), R8)                                                                 \
    JE(.STORE_ROW_GEN_6)                                                            \
    CMP(IMM(5), R8)                                                                 \
    JE(.STORE_ROW_GEN_5)                                                            \
    CMP(IMM(4), R8)                                                                 \
    JE(.STORE_ROW_GEN_4)                                                            \
    CMP(IMM(3), R8)                                                                 \
    JE(.STORE_ROW_GEN_3)                                                            \
    CMP(IMM(2), R8)                                                                 \
    JE(.STORE_ROW_GEN_2)                                                            \
    CMP(IMM(1), R8)                                                                 \
    JE(.STORE_ROW_GEN_1)                                                            \
                                                                                    \
    LABEL(.STORE_ROW_GEN_7)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(17)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_6)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(11)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_5)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(5)                                                            \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_4)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(24)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_3)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(18)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_2)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    ADD(RDI, RCX)                                                                   \
    BETA_ZERO_ROW_1xf(12)                                                           \
    JMP(.END)                                                                       \
                                                                                    \
    LABEL(.STORE_ROW_GEN_1)                                                         \
    BETA_ZERO_ROW_1xf(6)                                                            \
    LABEL(.END)                                                                     \

void bli_cgemmsup_cv_zen4_asm_fx1
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_store_row = m0;
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    uint16_t trans_load_mask = 0x3;

    const float value = 1.0f;
    const float *v = &value;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    float *conja_array = conja_arr;
    float *conjb_array = conjb_arr;

    if ( bli_is_conj( conja ) && bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx1_MAIN_BODY(MICRO_TILE_fx1_CONJA_CONJB, CONJ_LOAD_AB)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conja ) )
    {
        BEGIN_ASM()
        CGEMM_fx1_MAIN_BODY(MICRO_TILE_fx1_CONJA, CONJ_LOAD_A)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conja_array] "m" (conja_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else if ( bli_is_conj( conjb ) )
    {
        BEGIN_ASM()
        CGEMM_fx1_MAIN_BODY(MICRO_TILE_fx1_CONJB, CONJ_LOAD_B)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c),
          [conjb_array] "m" (conjb_array)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
    else
    {
        BEGIN_ASM()
        CGEMM_fx1_MAIN_BODY(MICRO_TILE_fx1, CONJ_LOAD_NN)
        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [k_iter]  "m" (k_iter),
          [k_left]  "m" (k_left),
          [m_store_row]  "m" (m_store_row),
          [m_load_mask]  "m" (m_load_mask),
          [trans_load_mask]  "m" (trans_load_mask),
          [alpha_mul_type]  "m" (alpha_mul_type),
          [beta_mul_type]   "m" (beta_mul_type),
          [alpha]  "m" (alpha),
          [a]      "m" (a),
          [b]      "m" (b),
          [beta]   "m" (beta),
          [c]      "m" (c),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "ebx", "eax", "al",
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31",
          "k2", "k3", "memory"
        )
    }
}
