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
    uint64_t n_left = n0 % NR;
    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 24x?m kernels, as needed.
    if ( n_left )
    {
      scomplex*  cij = c;
      scomplex*  bj  = b;
      scomplex*  ai  = a;

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
      return;
    }
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
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    LABEL(.CMLOOP)
    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)

    /* The k-loop is divided into 4 parts, to have a fixed distance for prefetching C */
    /* The k-loop is divided as follows :
        1. .CK_BP  : k/4 - 4 - PREFETCH_DIST_C((B)efore (P)refetch of C)
        2. .CK_DP  : 4(prefetches C)((D)uring (P)refetch of C)
        3. .CK_AP  : PREFETCH_DIST_C((A)fter (P)refetch of C)
        4. .CKLEFT : Fringe case of k loop
    */
   
    LABEL(.CK_BP) // Computation before prefetching
    /* Check for entering the k-loop(before prefetch) */
    SUB(IMM(4 + PREFETCH_DIST_C), R8)
    /* Jump to k-loop(prefetch) if k/4 <= 4 + PREFETCH_DIST_C */
    JLE(.CK_DP)
    LABEL(.CKITERLOOP_BP) // k-loop (B)efore (P)refetch of C

    /* Performing rank-1 update 4 times(based on unroll) */
    MICRO_TILE_24x4
    MICRO_TILE_24x4
    MICRO_TILE_24x4
    MICRO_TILE_24x4

    DEC(R8) // k_iter -= 1
    JNZ(.CKITERLOOP_BP)

    LABEL(.CK_DP) // Computation dring prefetching
    /* Check for entering the k-loop(for C prefetch) */
    ADD(IMM(4), R8)
    /* Jump to k-loop(after prefetch) if k/4 <= PREFETCH_DIST_C */
    JLE(.CK_AP)
    MOV(RCX, R9)
    LABEL(.CKITERLOOP_DP) // k-loop (D)uring (P)refetch of C

    /* Performing rank-1 update 4 times(based on unroll) */
    /* Also prefetch C */
    PREFETCH(1, MEM(R9))
    MICRO_TILE_24x4
    PREFETCH(1, MEM(R9, 64))
    MICRO_TILE_24x4
    PREFETCH(1, MEM(R9, 128))
    MICRO_TILE_24x4
    MICRO_TILE_24x4

    ADD(RSI, R9)

    DEC(R8)             // k_iter -= 1
    JNZ(.CKITERLOOP_DP)

    LABEL(.CK_AP) // Computation after prefetching
    /* Check for entering the k-loop(for C prefetch) */
    ADD(IMM(0 + PREFETCH_DIST_C), R8)
    /* Jump to k-loop(after prefetch) if k/4 <= 0 */
    JLE(.CKLEFT)
    LABEL(.CKITERLOOP_AP) // k-loop (A)fter (P)refetch of C

    /* Performing rank-1 update 4 times(based on unroll) */
    MICRO_TILE_24x4
    MICRO_TILE_24x4
    MICRO_TILE_24x4
    MICRO_TILE_24x4

    DEC(R8)             // k_iter -= 1
    JNZ(.CKITERLOOP_AP)

    // Remainder loop for k(k_fringe)
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    /* Performing rank-1 update */
    MICRO_TILE_24x4

    DEC(R8)             // k_left -= 1
    JNZ(.CKLEFTLOOP)

    /*
      ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to
      real components broadcasted from B.

      ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to
      imaginary components broadcasted from B.
    */

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)
    PERMUTE(12, 14, 16)
    PERMUTE(18, 20, 22)
    PERMUTE(24, 26, 28)

    // Final accumulation for A*B on 12 reg using the 24 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)
    ACC_COL(11, 12, 13, 14, 15, 16)
    ACC_COL(17, 18, 19, 20, 21, 22)
    ACC_COL(23, 24, 25, 26, 27, 28)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3   Col-4
      Rows(1-8)    ZMM6   ZMM12   ZMM18   ZMM24
      Rows(9-15)   ZMM8   ZMM14   ZMM20   ZMM26
      Rows(16-24)  ZMM10  ZMM16   ZMM22   ZMM28
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    ALPHA_MINUS_ONE(12, 14, 16)
    ALPHA_MINUS_ONE(18, 20, 22)
    ALPHA_MINUS_ONE(24, 26, 28)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)
    ALPHA_GENERIC(12, 14, 16)
    ALPHA_GENERIC(18, 20, 22)
    ALPHA_GENERIC(24, 26, 28)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20, 21, 22)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 23, 24, 25, 26, 27, 28)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18, 19, 20, 21, 22)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 23, 24, 25, 26, 27, 28)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18, 19, 20, 21, 22)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 23, 24, 25, 26, 27, 28)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))
    VMOVUPS(ZMM(10), MEM(RCX, 128))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))

    VMOVUPS(ZMM(18), MEM(R9))
    VMOVUPS(ZMM(20), MEM(R9, 64))
    VMOVUPS(ZMM(22), MEM(R9, 128))

    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))
    VMOVUPS(ZMM(28), MEM(R9, RSI, 1, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 24x4 micro-tile
      in blocks of 8x4.
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 10, 9, 16, 13, 22, 15, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 24x4 micro-tile
      in blocks of 8x4.
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(10, 16, 22, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a8), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)

    DEC(R11)
    JNE(.CMLOOP)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_left]  "m" (m_left),
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
      "memory"
    )

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
        bli_cgemmsup_cv_zen4_asm_16x4(conja, conjb, mr_cur, n0, k0, alpha,
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
        bli_cgemmsup_cv_zen4_asm_8x4(conja, conjb, mr_cur, n0, k0, alpha,
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
        bli_cgemmsup_cv_zen4_asm_fx4(conja, conjb, mr_cur, n0, k0, alpha,
            ai, rs_a0, cs_a0,
            bj, rs_b0, cs_b0,
            beta,
            cij, rs_c0, cs_c0,
            data, cntx);
      }
    }
}

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
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0011 1111
    uint16_t trans_load_mask = 0x3F;
    if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    LABEL(.CMLOOP)
    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_24x3
    MICRO_TILE_24x3
    MICRO_TILE_24x3
    MICRO_TILE_24x3

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_24x3

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 9 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)
    PERMUTE(12, 14, 16)
    PERMUTE(18, 20, 22)

    // Final accumulation for A*B on 9 reg using the 18 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)
    ACC_COL(11, 12, 13, 14, 15, 16)
    ACC_COL(17, 18, 19, 20, 21, 22)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3
      Rows(1-8)    ZMM6   ZMM12   ZMM18
      Rows(9-15)   ZMM8   ZMM14   ZMM20
      Rows(16-24)  ZMM10  ZMM16   ZMM22
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    ALPHA_MINUS_ONE(12, 14, 16)
    ALPHA_MINUS_ONE(18, 20, 22)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)
    ALPHA_GENERIC(12, 14, 16)
    ALPHA_GENERIC(18, 20, 22)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20, 21, 22)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18, 19, 20, 21, 22)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18, 19, 20, 21, 22)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))
    VMOVUPS(ZMM(10), MEM(RCX, 128))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))

    VMOVUPS(ZMM(18), MEM(R9))
    VMOVUPS(ZMM(20), MEM(R9, 64))
    VMOVUPS(ZMM(22), MEM(R9, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 24x3 micro-tile
      in blocks of 8x3.
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 24x3 micro-tile
      in blocks of 8x3.
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a8), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)

    DEC(R11)
    JNE(.CMLOOP)

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

    // Obtaining the panel stride for A, in case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 1111
    uint16_t trans_load_mask = 0xF;
    if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    LABEL(.CMLOOP)
    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_24x2
    MICRO_TILE_24x2
    MICRO_TILE_24x2
    MICRO_TILE_24x2

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_24x2

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 6 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)
    PERMUTE(12, 14, 16)

    // Final accumulation for A*B on 6 reg using the 12 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)
    ACC_COL(11, 12, 13, 14, 15, 16)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2
      Rows(1-8)    ZMM6   ZMM12
      Rows(9-15)   ZMM8   ZMM14
      Rows(16-24)  ZMM10  ZMM16
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    ALPHA_MINUS_ONE(12, 14, 16)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)
    ALPHA_GENERIC(12, 14, 16)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14, 15, 16)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14, 15, 16)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14, 15, 16)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))
    VMOVUPS(ZMM(10), MEM(RCX, 128))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))
    VMOVUPS(ZMM(16), MEM(RCX, RSI, 1, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 24x2 micro-tile
      in blocks of 8x2.
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for loading/storing C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(22), ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for loading/storing C
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 24x2 micro-tile
      in blocks of 8x2.
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(22), ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a8), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)

    DEC(R11)
    JNE(.CMLOOP)

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

    // Obtaining the panel stride for A, in case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( scomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 0011
    uint16_t trans_load_mask = 0x3;
    if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    LABEL(.CMLOOP)
    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_24x1
    MICRO_TILE_24x1
    MICRO_TILE_24x1
    MICRO_TILE_24x1

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_24x1

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 3 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)

    // Final accumulation for A*B on 3 reg using the 6 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1
      Rows(1-8)    ZMM6
      Rows(9-15)   ZMM8
      Rows(16-24)  ZMM10
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8, 9, 10)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))
    VMOVUPS(ZMM(10), MEM(RCX, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 24x1 micro-tile
      in blocks of 8x1.
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(14), ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(16), ZMM(22), ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 10, 9, 16, 13, 22, 15, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 24x1 micro-tile
      in blocks of 8x1.
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(14), ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(17-24)  ZMM(10)  ZMM(16)  ZMM(22)  ZMM(28)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(16), ZMM(22), ZMM(28), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(10, 16, 22, 28, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(10, 16, 22, 28)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a8), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 8), R12)

    DEC(R11)
    JNE(.CMLOOP)

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_16x4
    MICRO_TILE_16x4
    MICRO_TILE_16x4
    MICRO_TILE_16x4

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_16x4

    DEC(R8)
    JNZ(.CKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 8 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8)
    PERMUTE(12, 14)
    PERMUTE(18, 20)
    PERMUTE(24, 26)

    // Final accumulation for A*B on 8 reg using the 16 reg.
    ACC_COL(5, 6, 7, 8)
    ACC_COL(11, 12, 13, 14)
    ACC_COL(17, 18, 19, 20)
    ACC_COL(23, 24, 25, 26)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3   Col-4
      Rows(1-8)    ZMM6   ZMM12   ZMM18   ZMM24
      Rows(9-15)   ZMM8   ZMM14   ZMM20   ZMM26
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8)
    ALPHA_MINUS_ONE(12, 14)
    ALPHA_MINUS_ONE(18, 20)
    ALPHA_MINUS_ONE(24, 26)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8)
    ALPHA_GENERIC(12, 14)
    ALPHA_GENERIC(18, 20)
    ALPHA_GENERIC(24, 26)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 23, 24, 25, 26)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18, 19, 20)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 23, 24, 25, 26)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18, 19, 20)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 23, 24, 25, 26)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))

    VMOVUPS(ZMM(18), MEM(R9))
    VMOVUPS(ZMM(20), MEM(R9, 64))

    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))
    VMOVUPS(ZMM(26), MEM(R9, RSI, 1, 64))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 16x4 micro-tile
      in blocks of 8x4.
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for loading/storing C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 16x4 micro-tile
      in blocks of 8x4.
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)
    LABEL(.END)

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

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0011 1111
    uint16_t trans_load_mask = 0x3F;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_16x3
    MICRO_TILE_16x3
    MICRO_TILE_16x3
    MICRO_TILE_16x3

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_16x3

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 6 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8)
    PERMUTE(12, 14)
    PERMUTE(18, 20)

    // Final accumulation for A*B on 6 reg using the 12 reg.
    ACC_COL(5, 6, 7, 8)
    ACC_COL(11, 12, 13, 14)
    ACC_COL(17, 18, 19, 20)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3
      Rows(1-8)    ZMM6   ZMM12   ZMM18
      Rows(9-15)   ZMM8   ZMM14   ZMM20
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8)
    ALPHA_MINUS_ONE(12, 14)
    ALPHA_MINUS_ONE(18, 20)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8)
    ALPHA_GENERIC(12, 14)
    ALPHA_GENERIC(18, 20)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18, 19, 20)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18, 19, 20)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18, 19, 20)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))

    VMOVUPS(ZMM(18), MEM(R9))
    VMOVUPS(ZMM(20), MEM(R9, 64))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 16x3 micro-tile
      in blocks of 8x3.
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 16x3 micro-tile
      in blocks of 8x3.
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 1111
    uint16_t trans_load_mask = 0xF;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_16x2
    MICRO_TILE_16x2
    MICRO_TILE_16x2
    MICRO_TILE_16x2

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_16x2

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 4 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8)
    PERMUTE(12, 14)

    // Final accumulation for A*B on 4 reg using the 8 reg.
    ACC_COL(5, 6, 7, 8)
    ACC_COL(11, 12, 13, 14)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2
      Rows(1-8)    ZMM6   ZMM12
      Rows(9-15)   ZMM8   ZMM14
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8)
    ALPHA_MINUS_ONE(12, 14)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8)
    ALPHA_GENERIC(12, 14)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12, 13, 14)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12, 13, 14)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12, 13, 14)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPS(ZMM(14), MEM(RCX, RSI, 1, 64))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 16x2 micro-tile
      in blocks of 8x2.
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for loading/storing C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 16x2 micro-tile
      in blocks of 8x2.
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 0011
    uint16_t trans_load_mask = 0x3;
    // if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_16x1
    MICRO_TILE_16x1
    MICRO_TILE_16x1
    MICRO_TILE_16x1

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_16x1

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 2 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8)

    // Final accumulation for A*B on 2 reg using the 4 reg.
    ACC_COL(5, 6, 7, 8)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1
      Rows(1-8)    ZMM6
      Rows(9-15)   ZMM8
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6, 7, 8)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))
    VMOVUPS(ZMM(8), MEM(RCX, 64))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 16x1 micro-tile
      in blocks of 8x1.
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(14), ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 8, 9, 14, 13, 20, 15, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      In-register transposition happens over the 16x1 micro-tile
      in blocks of 8x1.
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LEA(MEM(RCX, RDI, 2), RCX)

    /*
      Input for transpose:
                   Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-16)    ZMM(8)   ZMM(14)  ZMM(20)  ZMM(26)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(14), ZMM(20), ZMM(26), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(8, 14, 20, 26, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(8, 14, 20, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_8x4
    MICRO_TILE_8x4
    MICRO_TILE_8x4
    MICRO_TILE_8x4

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_8x4

    DEC(R8)
    JNZ(.CKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 4 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)
    PERMUTE(18)
    PERMUTE(24)

    // Final accumulation for A*B on 4 reg using the 8 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)
    ACC_COL(17, 18)
    ACC_COL(23, 24)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3   Col-4
      Rows(1-8)    ZMM6   ZMM12   ZMM18   ZMM24
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    ALPHA_MINUS_ONE(18)
    ALPHA_MINUS_ONE(24)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)
    ALPHA_GENERIC(18)
    ALPHA_GENERIC(24)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 23, 24)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 23, 24)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 23, 24)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))

    VMOVUPS(ZMM(18), MEM(R9))

    VMOVUPS(ZMM(24), MEM(R9, RSI, 1))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4x4(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4x4(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4x4(5, 11, 17, 23)
    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0011 1111
    uint16_t trans_load_mask = 0x3F;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_8x3
    MICRO_TILE_8x3
    MICRO_TILE_8x3
    MICRO_TILE_8x3

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_8x3

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 6 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)
    PERMUTE(18)

    // Final accumulation for A*B on 6 reg using the 12 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)
    ACC_COL(17, 18)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3
      Rows(1-8)    ZMM6   ZMM12   ZMM18
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    ALPHA_MINUS_ONE(18)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)
    ALPHA_GENERIC(18)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 17, 18)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 17, 18)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 17, 18)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))

    VMOVUPS(ZMM(18), MEM(R9))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This 8x3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 1111
    uint16_t trans_load_mask = 0xF;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_8x2
    MICRO_TILE_8x2
    MICRO_TILE_8x2
    MICRO_TILE_8x2

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_8x2

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 4 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)

    // Final accumulation for A*B on 4 reg using the 8 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2
      Rows(1-8)    ZMM6   ZMM12
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE(RCX, 11, 12)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC(RCX, 11, 12)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE(RCX, 11, 12)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This 8x2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)
    LABEL(.END)

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
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 0011
    uint16_t trans_load_mask = 0x3;
    // if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_8x1
    MICRO_TILE_8x1
    MICRO_TILE_8x1
    MICRO_TILE_8x1

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_8x1

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 2 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)

    // Final accumulation for A*B on 2 reg using the 4 reg.
    ACC_COL(5, 6)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1
      Rows(1-8)    ZMM6
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE(RCX, 5, 6)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    VMOVUPS(ZMM(6), MEM(RCX))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Copy the address of C */
    MOV(RCX, R9)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_GEN_ROW_4xf(7, 6, 9, 12, 13, 18, 15, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4xf(7, 5, 9, 11, 13, 17, 15, 23)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This 8x1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(12), ZMM(18), ZMM(24), ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for
      transposing(part of the input) and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    BETA_ZERO_ROW_4xf(6, 12, 18, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    BETA_ZERO_ROW_4xf(5, 11, 17, 23)

    LABEL(.END)

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(m_load_mask), EBX)
    KMOVW(EBX, k(2))        // k(2) = m_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_fx4
    MICRO_TILE_fx4
    MICRO_TILE_fx4
    MICRO_TILE_fx4

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_fx4

    DEC(R8)
    JNZ(.CKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 4 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)
    PERMUTE(18)
    PERMUTE(24)

    // Final accumulation for A*B on 4 reg using the 8 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)
    ACC_COL(17, 18)
    ACC_COL(23, 24)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3   Col-4
      Rows(1-8)    ZMM6   ZMM12   ZMM18   ZMM24
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    ALPHA_MINUS_ONE(18)
    ALPHA_MINUS_ONE(24)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)
    ALPHA_GENERIC(18)
    ALPHA_GENERIC(24)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 23, 24)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 23, 24)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 17, 18)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 23, 24)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))

    VMOVUPS(ZMM(18), MEM(R9) MASK_(k(2)))

    VMOVUPS(ZMM(24), MEM(R9, RSI, 1) MASK_(k(2)))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      In-register transposition happens over the 24x4 micro-tile
      in blocks of 8x4.
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.SCALE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.SCALE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.SCALE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.SCALE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.SCALE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.SCALE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.SCALE_ROW_GEN_1)

    LABEL(.SCALE_ROW_GEN_7)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(23, 11)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(25, 17)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_6)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(23, 11)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_5)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(21, 5)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_4)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(15, 24)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_3)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(13, 18)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_2)
    BETA_GEN_ROW_1x4(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1x4(9, 12)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_1)
    BETA_GEN_ROW_1x4(7, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This 8x4 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
     7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.STORE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.STORE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.STORE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.STORE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.STORE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.STORE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.STORE_ROW_GEN_1)

    LABEL(.STORE_ROW_GEN_7)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(11)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(17)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_6)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(11)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_5)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(5)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_4)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(24)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_3)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(18)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_2)
    BETA_ZERO_ROW_1x4(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1x4(12)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_1)
    BETA_ZERO_ROW_1x4(6)

    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring fx3 compatability
      while transposing, and loading/storing C(k(2) mask register).
      This mask is set based on the m-value(m0) that the kernel receives.
      m0 is guaranteed to be less than 8.
    */
    uint64_t m_store_row = m0; // Also used when handling row-storage of C(post transpose)
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0011 1111
    uint16_t trans_load_mask = 0x3F;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(m_load_mask), EBX)
    KMOVW(EBX, k(2))               // k(2) = m_load_mask

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_fx3
    MICRO_TILE_fx3
    MICRO_TILE_fx3
    MICRO_TILE_fx3

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_fx3

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 6 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)
    PERMUTE(18)

    // Final accumulation for A*B on 6 reg using the 12 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)
    ACC_COL(17, 18)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2   Col-3
      Rows(1-8)    ZMM6   ZMM12   ZMM18
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    ALPHA_MINUS_ONE(18)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)
    ALPHA_GENERIC(18)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 17, 18)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 17, 18)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 11, 12)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 17, 18)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))

    VMOVUPS(ZMM(18), MEM(R9) MASK_(k(2)))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
        Check for beta_mul_type, to jump to the required code-section
        Intermediate C = beta*C + IR, where IR = alpha*A*B
        If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
        C = IR, skip beta-scaling
        else => BLIS_MUL_DEFAULT
        C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This fx3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.SCALE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.SCALE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.SCALE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.SCALE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.SCALE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.SCALE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.SCALE_ROW_GEN_1)

    LABEL(.SCALE_ROW_GEN_7)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(25, 17)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_6)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_5)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_4)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_3)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_2)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_1)
    BETA_GEN_ROW_1xf(7, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This fx3 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.STORE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.STORE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.STORE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.STORE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.STORE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.STORE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.STORE_ROW_GEN_1)

    LABEL(.STORE_ROW_GEN_7)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(17)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_6)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_5)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_4)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_3)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_2)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_1)
    BETA_ZERO_ROW_1xf(6)

    LABEL(.END)

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

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring fx3 compatability
      while transposing, and loading/storing C(k(2) mask register).
      This mask is set based on the m-value(m0) that the kernel receives.
      m0 is guaranteed to be less than 8.
    */
    uint64_t m_store_row = m0; // Also used when handling row-storage of C(post transpose)
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 1111
    uint16_t trans_load_mask = 0xF;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(m_load_mask), EBX)
    KMOVW(EBX, k(2))               // k(2) = m_load_mask

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_fx2
    MICRO_TILE_fx2
    MICRO_TILE_fx2
    MICRO_TILE_fx2

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_fx2

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 4 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)
    PERMUTE(12)

    // Final accumulation for A*B on 4 reg using the 8 reg.
    ACC_COL(5, 6)
    ACC_COL(11, 12)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1  Col-2
      Rows(1-8)    ZMM6   ZMM12
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    ALPHA_MINUS_ONE(12)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)
    ALPHA_GENERIC(12)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_MINUS_ONE_fC(RCX, 11, 12)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_GENERIC_fC(RCX, 11, 12)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE_fC(RCX, 5, 6)
    ADD(RSI, RCX)
    BETA_ONE_fC(RCX, 11, 12)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    LEA(MEM(RCX, RSI, 2), R9)
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))

    VMOVUPS(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
      C = IR, skip beta-scaling
      else => BLIS_MUL_DEFAULT
      C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This fx2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.SCALE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.SCALE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.SCALE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.SCALE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.SCALE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.SCALE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.SCALE_ROW_GEN_1)

    LABEL(.SCALE_ROW_GEN_7)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(25, 17)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_6)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_5)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_4)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_3)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_2)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_1)
    BETA_GEN_ROW_1xf(7, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This fx2 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.STORE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.STORE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.STORE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.STORE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.STORE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.STORE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.STORE_ROW_GEN_1)

    LABEL(.STORE_ROW_GEN_7)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(17)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_6)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_5)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_4)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_3)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_2)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_1)
    BETA_ZERO_ROW_1xf(6)
    LABEL(.END)

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
    // Main kernel
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;

    /*
      The mask bits below are set for ensuring fx3 compatability
      while transposing, and loading/storing C(k(2) mask register).
      This mask is set based on the m-value(m0) that the kernel receives.
      m0 is guaranteed to be less than 8.
    */
    uint64_t m_store_row = m0; // Also used when handling row-storage of C(post transpose)
    uint16_t m_load_mask = ( (uint16_t)1 << ( 2 * m_store_row ) ) - (uint16_t)1;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 16 single precision elements.
    */
    // Mask for transposing and loading = 0b 0000 0000 0000 0011
    uint16_t trans_load_mask = 0x3;
    // if ( m_iter == 0 ) goto consider_edge_cases;

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

    BEGIN_ASM()
    MOV(VAR(a), R10) // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX) // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12) // R12 = base addr of C (MCxNR block)

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13) // R13 = sizeof(scomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14) // R14 = sizeof(scomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15) // R15 = sizeof(scomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI) // RDI = sizeof(scomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI) // RSI = sizeof(scomplex)*cs_c

    MOV(VAR(m_load_mask), EBX)
    KMOVW(EBX, k(2))               // k(2) = m_load_mask

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9) // Used in fmaddsub instruction
    VBROADCASTSS(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(R10, RAX) // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX) // RBX = addr of B for the KCxNR block
    MOV(R12, RCX) // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.CKLEFT)
    // Main loop for k
    LABEL(.CKMAINLOOP)

    MICRO_TILE_fx1
    MICRO_TILE_fx1
    MICRO_TILE_fx1
    MICRO_TILE_fx1

    DEC(R8)
    JNZ(.CKMAINLOOP)

    // Remainder loop for k
    LABEL(.CKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.CKLEFTLOOP)

    MICRO_TILE_fx1

    DEC(R8)
    JNZ(.CKLEFTLOOP)


    LABEL(.ACCUMULATE) // Accumulating A*B over 2 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6)

    // Final accumulation for A*B on 2 reg using the 4 reg.
    ACC_COL(5, 6)

    // A*B is accumulated over the ZMM registers as follows :
    /*
                   Col-1
      Rows(1-8)    ZMM6
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    /*
      Check for alpha_mul_type, to jump to the required code-section
      Intermediate result(IR) = alpha*(A*B)
      If alpha == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       IR = A*B
      else if, alpha != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       IR = alpha*(A*B), using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       IR = 0.0 - A*B, using subtraction
   */
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSS(MEM(RAX), ZMM(0)) // Alpha->real
    VBROADCASTSS(MEM(RAX, 4), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(8), RSI) // cs_c = 1*sizeof(scomplex) = 8
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
       C = IR, skip beta-scaling
      else if beta == ( 1.0, 0.0 ) => BLIS_MUL_ONE
       C = C + IR, using addition
      else if, beta != ( -1.0, 0.0 ) => BLIS_MUL_DEFAULT
       C = beta*C + IR, using complex multiplication
      else => BLIS_MUL_MINUS_ONE
       C = ( 0.0 - C ) + IR, using subtraction
    */

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    JE(.STORE)
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Subtracting C from alpha*A*B, one column at a time
    BETA_MINUS_ONE_fC(RCX, 5, 6)
    JMP(.END)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC_fC(RCX, 5, 6)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE_fC(RCX, 5, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE)
    VMOVUPS(ZMM(6), MEM(RCX) MASK_(k(2)))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
      Check for beta_mul_type, to jump to the required code-section
      Intermediate C = beta*C + IR, where IR = alpha*A*B
      If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
      C = IR, skip beta-scaling
      else => BLIS_MUL_DEFAULT
      C = beta*C + IR, using complex multiplication
    */
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL) // Checking if beta == 0
    /* Skip beta scaling and jump to store */
    JE(.STORE_ROW)

    LABEL(.BETA_GENERAL_ROW)
    /* Load beta onto a ZMM register */
    MOV(VAR(beta), RBX)
    /* Broadcast the real and imag components of beta onto the registers */
    VBROADCASTSS(MEM(RBX), ZMM(0)) // Beta->real
    VBROADCASTSS(MEM(RBX, 4), ZMM(1)) // Beta->imag
    /*
      This fx1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing(part of the input)
      Registers ZMM(21), ZMM(23), ZMM(25), ZMM(27) are used for transposing(temporary registers)
      Registers ZMM(7), ZMM(9), ZMM(13) and ZMM(15) are used for transposing(temporary registers)
      and for loading/storing to C.
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.SCALE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.SCALE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.SCALE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.SCALE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.SCALE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.SCALE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.SCALE_ROW_GEN_1)

    LABEL(.SCALE_ROW_GEN_7)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(25, 17)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_6)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(23, 11)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_5)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(21, 5)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_4)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(15, 24)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_3)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(13, 18)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_2)
    BETA_GEN_ROW_1xf(7, 6)
    ADD(RDI, RCX)
    BETA_GEN_ROW_1xf(9, 12)
    JMP(.END)

    LABEL(.SCALE_ROW_GEN_1)
    BETA_GEN_ROW_1xf(7, 6)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    /*
      This fx1 block is further extended into an 8x8 block to perform
      the transpose operation.
      Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(6)   ZMM(12)  ZMM(18)  ZMM(24)  ZMM(5)  ZMM(11)  ZMM(17)  ZMM(23)

      Registers ZMM(5), ZMM(11), ZMM(17), ZMM(23) are used for transposing and storing to C
    */
    TRANSPOSE_8x8(6, 12, 18, 24, 5, 11, 17, 23,
      7, 9, 13, 15, 21, 23, 25, 27)
    /* Store the appropriate number of registers based on m0 */
    MOV(var(m_store_row), R8)
    CMP(IMM(7), R8)
    JE(.STORE_ROW_GEN_7)
    CMP(IMM(6), R8)
    JE(.STORE_ROW_GEN_6)
    CMP(IMM(5), R8)
    JE(.STORE_ROW_GEN_5)
    CMP(IMM(4), R8)
    JE(.STORE_ROW_GEN_4)
    CMP(IMM(3), R8)
    JE(.STORE_ROW_GEN_3)
    CMP(IMM(2), R8)
    JE(.STORE_ROW_GEN_2)
    CMP(IMM(1), R8)
    JE(.STORE_ROW_GEN_1)

    LABEL(.STORE_ROW_GEN_7)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(17)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_6)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(11)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_5)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(5)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_4)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(24)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_3)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(18)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_2)
    BETA_ZERO_ROW_1xf(6)
    ADD(RDI, RCX)
    BETA_ZERO_ROW_1xf(12)
    JMP(.END)

    LABEL(.STORE_ROW_GEN_1)
    BETA_ZERO_ROW_1xf(6)

    LABEL(.END)

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
