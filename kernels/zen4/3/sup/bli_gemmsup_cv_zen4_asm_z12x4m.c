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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
#define PREFETCH_DIST_C 4
#define MR 12
#define NR 4

// Macro for resetting the registers for accumulation
#define RESET_REGISTERS \
    VXORPD(ZMM(5), ZMM(5), ZMM(5))      \
    VXORPD(ZMM(6), ZMM(6), ZMM(6))      \
    VXORPD(ZMM(7), ZMM(7), ZMM(7))      \
    VXORPD(ZMM(8), ZMM(8), ZMM(8))      \
    VXORPD(ZMM(9), ZMM(9), ZMM(9))      \
    VXORPD(ZMM(10), ZMM(10), ZMM(10))   \
    VXORPD(ZMM(11), ZMM(11), ZMM(11))   \
    VXORPD(ZMM(12), ZMM(12), ZMM(12))   \
    VXORPD(ZMM(13), ZMM(13), ZMM(13))   \
    VXORPD(ZMM(14), ZMM(14), ZMM(14))   \
    VXORPD(ZMM(15), ZMM(15), ZMM(15))   \
    VXORPD(ZMM(16), ZMM(16), ZMM(16))   \
    VXORPD(ZMM(17), ZMM(17), ZMM(17))   \
    VXORPD(ZMM(18), ZMM(18), ZMM(18))   \
    VXORPD(ZMM(19), ZMM(19), ZMM(19))   \
    VXORPD(ZMM(20), ZMM(20), ZMM(20))   \
    VXORPD(ZMM(21), ZMM(21), ZMM(21))   \
    VXORPD(ZMM(22), ZMM(22), ZMM(22))   \
    VXORPD(ZMM(23), ZMM(23), ZMM(23))   \
    VXORPD(ZMM(24), ZMM(24), ZMM(24))   \
    VXORPD(ZMM(25), ZMM(25), ZMM(25))   \
    VXORPD(ZMM(26), ZMM(26), ZMM(26))   \
    VXORPD(ZMM(27), ZMM(27), ZMM(27))   \
    VXORPD(ZMM(28), ZMM(28), ZMM(28))   \
    VXORPD(ZMM(30), ZMM(30), ZMM(30))   \
    VXORPD(ZMM(31), ZMM(31), ZMM(31))   \

// Macro to permute in case of 3 loads(12x? cases)
#define PERMUTE_12Z(R1, R2, R3)  \
    VPERMILPD(IMM(0x55), ZMM(R1), ZMM(R1))  \
    VPERMILPD(IMM(0x55), ZMM(R2), ZMM(R2))  \
    VPERMILPD(IMM(0x55), ZMM(R3), ZMM(R3))  \

// Macro to permute in case of 2 loads(8x? cases)
#define PERMUTE_8Z(R1, R2)  \
    VPERMILPD(IMM(0x55), ZMM(R1), ZMM(R1))  \
    VPERMILPD(IMM(0x55), ZMM(R2), ZMM(R2))  \

// Macro to permute in case of 1 loads(4x? cases)
#define PERMUTE_4Z(R1)  \
    VPERMILPD(IMM(0x55), ZMM(R1), ZMM(R1))  \

// Macro to get the PERMUTE_? signature from the list
#define GET_PERMUTE(_1, _2, _3, NAME, ...)  NAME

// Overloaded macro PERMUTE with variable arguments
#define PERMUTE(...)  \
    GET_PERMUTE(__VA_ARGS__,  \
    PERMUTE_12Z, PERMUTE_8Z, PERMUTE_4Z)(__VA_ARGS__) \

// Macro for fma op in case of 3 loads(12x? cases)
#define FMA_12Z(B, R1, R2, R3)  \
    VFMADD231PD(ZMM(0), ZMM(B), ZMM(R1))  \
    VFMADD231PD(ZMM(1), ZMM(B), ZMM(R2))  \
    VFMADD231PD(ZMM(2), ZMM(B), ZMM(R3))  \

// Macro for fma op in case of 2 loads(8x? cases)
#define FMA_8Z(B, R1, R2)  \
    VFMADD231PD(ZMM(0), ZMM(B), ZMM(R1))  \
    VFMADD231PD(ZMM(1), ZMM(B), ZMM(R2))  \

// Macro for fma op in case of 1 load(4x? cases)
#define FMA_4Z(B, R1)  \
    VFMADD231PD(ZMM(0), ZMM(B), ZMM(R1))  \

// Macro to get the FMA_? signature from the list
#define GET_FMA(_1, _2, _3, _4, NAME, ...)  NAME

// Overloaded macro FMA with variable arguments
#define FMA(...)  \
    GET_FMA(__VA_ARGS__,  \
    FMA_12Z, FMA_8Z, FMA_4Z)(__VA_ARGS__) \

// Macro for accumalation in case of 3 loads(12x? cases)
#define ACC_COL_12Z(R1, I1, R2, I2, R3, I3)  \
    VFMADDSUB231PD(ZMM(R1), ZMM(29), ZMM(I1))  \
    VFMADDSUB231PD(ZMM(R2), ZMM(29), ZMM(I2))  \
    VFMADDSUB231PD(ZMM(R3), ZMM(29), ZMM(I3))  \

// Macro for accumalation in case of 2 loads(8x? cases)
#define ACC_COL_8Z(R1, I1, R2, I2)  \
    VFMADDSUB231PD(ZMM(R1), ZMM(29), ZMM(I1))  \
    VFMADDSUB231PD(ZMM(R2), ZMM(29), ZMM(I2))  \

// Macro for accumalation in case of 1 load(4x? cases)
#define ACC_COL_4Z(R1, I1)  \
    VFMADDSUB231PD(ZMM(R1), ZMM(29), ZMM(I1))  \

// Macro to get the ACC_COL_? signature from the list
#define GET_ACC_COL(_1, _2, _3, _4, _5, _6, NAME, ...)  NAME

// Overloaded macro ACC_COL with variable arguments
#define ACC_COL(...)  \
    GET_ACC_COL(__VA_ARGS__,  \
    ACC_COL_12Z, _0, ACC_COL_8Z, _1, ACC_COL_4Z)(__VA_ARGS__) \

// Macro for scaling with alpha if it is complex
// in case of 3 loads(12x? cases)
#define ALPHA_GENERIC_12Z(R1, R2, R3) \
    VMULPD(ZMM(0), ZMM(R1), ZMM(2))  \
    VMULPD(ZMM(1), ZMM(R1), ZMM(R1))  \
    VMULPD(ZMM(0), ZMM(R2), ZMM(30))  \
    VMULPD(ZMM(1), ZMM(R2), ZMM(R2))  \
    VMULPD(ZMM(0), ZMM(R3), ZMM(31))  \
    VMULPD(ZMM(1), ZMM(R3), ZMM(R3))  \
    PERMUTE(R1, R2, R3) \
    ACC_COL(2, R1, 30, R2, 31, R3)  \

// Macro for scaling with alpha if it is complex
// in case of 2 loads(8x? cases)
#define ALPHA_GENERIC_8Z(R1, R2) \
    VMULPD(ZMM(0), ZMM(R1), ZMM(2))  \
    VMULPD(ZMM(1), ZMM(R1), ZMM(R1))  \
    VMULPD(ZMM(0), ZMM(R2), ZMM(30))  \
    VMULPD(ZMM(1), ZMM(R2), ZMM(R2))  \
    PERMUTE(R1, R2) \
    ACC_COL(2, R1, 30, R2)  \

// Macro for scaling with alpha if it is complex
// in case of 1 load(4x? cases)
#define ALPHA_GENERIC_4Z(R1) \
    VMULPD(ZMM(0), ZMM(R1), ZMM(2))  \
    VMULPD(ZMM(1), ZMM(R1), ZMM(R1))  \
    PERMUTE(R1) \
    ACC_COL(2, R1)  \

// Macro to get the ALPHA_GENERIC_? signature from the list
#define GET_ALPHA_GENERIC(_1, _2, _3, NAME, ...)  NAME

// Overloaded macro ALPHA_GENERIC with variable arguments
#define ALPHA_GENERIC(...)  \
    GET_ALPHA_GENERIC(__VA_ARGS__,  \
    ALPHA_GENERIC_12Z, ALPHA_GENERIC_8Z, ALPHA_GENERIC_4Z)(__VA_ARGS__) \

// Macro for scaling with beta if it is complex
// in case of 3 loads(12x? cases)
#define BETA_GENERIC_12Z(C, R1, I1, R2, I2, R3, I3)\
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3))  \
    \
    ALPHA_GENERIC(R1, R2, R3) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128))  \

// Macro for scaling with beta if it is complex
// in case of 2 loads(8x? cases)
#define BETA_GENERIC_8Z(C, R1, I1, R2, I2)\
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    \
    ALPHA_GENERIC(R1, R2) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \

// Macro for scaling with beta if it is complex
// in case of 1 load(4x? cases)
#define BETA_GENERIC_4Z(C, R1, I1)\
    VMOVUPD(MEM(C), ZMM(R1)) \
    \
    ALPHA_GENERIC(R1) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \

// Macro for scaling with beta if it is complex
// in case of 3 loads(12x? cases)
#define BETA_GENERIC_12Z_MASK(C, R1, I1, R2, I2, R3, I3)\
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3) MASK_(k(2)))  \
    \
    ALPHA_GENERIC(R1, R2, R3) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128) MASK_(k(2))  )  \

// Macro for scaling with beta if it is complex
// in case of 2 loads(8x? cases)
#define BETA_GENERIC_8Z_MASK(C, R1, I1, R2, I2)\
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
    \
    ALPHA_GENERIC(R1, R2) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

// Macro for scaling with beta if it is complex
// in case of 1 load(4x? cases)
#define BETA_GENERIC_4Z_MASK(C, R1, I1)\
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(2))) \
    \
    ALPHA_GENERIC(R1) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C) MASK_(k(2))) \

// Macro to get the BETA_GENERIC_? signature from the list
#define GET_BETA_GENERIC(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_GENERIC with variable arguments
#define BETA_GENERIC(...)  \
    GET_BETA_GENERIC(__VA_ARGS__,  \
    BETA_GENERIC_12Z, _0, BETA_GENERIC_8Z, _1, BETA_GENERIC_4Z)(__VA_ARGS__) \

// Macro to get the BETA_GENERIC_? signature from the list
#define GET_BETA_GENERIC_MASK(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_GENERIC with variable arguments
#define BETA_GENERIC_MASK(...)  \
    GET_BETA_GENERIC_MASK(__VA_ARGS__,  \
    BETA_GENERIC_12Z_MASK, _0, BETA_GENERIC_8Z_MASK, _1, BETA_GENERIC_4Z_MASK)(__VA_ARGS__) \


// Macro for scaling with beta if it is complex
// in case of 1 load(fx? cases, f<4)
#define BETA_GENERIC_fZ(C, R1, I1) \
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(2))) \
\
    ALPHA_GENERIC(R1) \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
\
    VMOVUPD(ZMM(I1), MEM(C) MASK_(k(2))) \

#define MICRO_TILE_12x4                             \
    /* Macro for 12x4 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    LEA(MEM(RBX, R15, 2), R9)                       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(R9), ZMM(3))                   \
    VBROADCASTSD(MEM(R9, 8), ZMM(4))                \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(R9, R15, 1), ZMM(30))          \
    VBROADCASTSD(MEM(R9, R15, 1, 8), ZMM(31))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19, 21)                              \
    FMA(4, 18, 20, 22)                              \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 23, 25, 27)                             \
    FMA(31, 24, 26, 28)                             \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x4                              \
    /* Macro for 8x4 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    LEA(MEM(RBX, R15, 2), R9)                       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(R9), ZMM(3))                   \
    VBROADCASTSD(MEM(R9, 8), ZMM(4))                \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(R9, R15, 1), ZMM(30))          \
    VBROADCASTSD(MEM(R9, R15, 1, 8), ZMM(31))       \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19)                                  \
    FMA(4, 18, 20)                                  \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 23, 25)                                 \
    FMA(31, 24, 26)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x4                                      \
    /* Macro for 4x4 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    LEA(MEM(RBX, R15, 2), R9)                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(17))              \
    VFMADD231PD(mem_1to8(R9,  8), ZMM(0), ZMM(18))          \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(23))      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(24))   \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

/* Macro to perform an fx4 micro-tile computation(f<4) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx4                                      \
    /* Macro for fx4 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    LEA(MEM(RBX, R15, 2), R9)                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(17))              \
    VFMADD231PD(mem_1to8(R9,  8), ZMM(0), ZMM(18))          \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(23))      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(24))   \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x4_MASK                        \
    /* Macro for 12x4 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    LEA(MEM(RBX, R15, 2), R9)                       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(R9), ZMM(3))                   \
    VBROADCASTSD(MEM(R9, 8), ZMM(4))                \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(R9, R15, 1), ZMM(30))          \
    VBROADCASTSD(MEM(R9, R15, 1, 8), ZMM(31))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19, 21)                              \
    FMA(4, 18, 20, 22)                              \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 23, 25, 27)                             \
    FMA(31, 24, 26, 28)                             \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x4_MASK                         \
    /* Macro for 8x4 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    LEA(MEM(RBX, R15, 2), R9)                       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(R9), ZMM(3))                   \
    VBROADCASTSD(MEM(R9, 8), ZMM(4))                \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(R9, R15, 1), ZMM(30))          \
    VBROADCASTSD(MEM(R9, R15, 1, 8), ZMM(31))       \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19)                                  \
    FMA(4, 18, 20)                                  \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 23, 25)                                 \
    FMA(31, 24, 26)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x4_MASK_SET1                            \
    /* Macro for 4x4 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    LEA(MEM(RBX, R15, 2), R9)                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(17))              \
    VFMADD231PD(mem_1to8(R9,  8), ZMM(0), ZMM(18))          \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(23))      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(24))   \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_4x4_MASK_SET2                            \
    /* Macro for 4x4 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    LEA(MEM(RBX, R15, 2), R9)                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1), ZMM(8))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(14))  \
    VFMADD231PD(mem_1to8(R9), ZMM(1), ZMM(19))              \
    VFMADD231PD(mem_1to8(R9,  8), ZMM(1), ZMM(20))          \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(1), ZMM(25))      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(1), ZMM(26))   \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x3                             \
    /* Macro for 12x3 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX, R15, 2), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 2, 8), ZMM(4))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19, 21)                              \
    FMA(4, 18, 20, 22)                              \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x3                              \
    /* Macro for 8x3 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX, R15, 2), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 2, 8), ZMM(4))       \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19)                                  \
    FMA(4, 18, 20)                                  \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x3                                      \
    /* Macro for 4x3 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(RBX, R15, 2), ZMM(0), ZMM(17))     \
    VFMADD231PD(mem_1to8(RBX, R15, 2, 8), ZMM(0), ZMM(18))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x3_MASK                        \
    /* Macro for 12x3 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX, R15, 2), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 2, 8), ZMM(4))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19, 21)                              \
    FMA(4, 18, 20, 22)                              \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x3_MASK                         \
    /* Macro for 8x3 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))                   \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX, R15, 2), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 2, 8), ZMM(4))       \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 17, 19)                                  \
    FMA(4, 18, 20)                                  \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x3_MASK_SET1                            \
    /* Macro for 4x3 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(RBX, R15, 2), ZMM(0), ZMM(17))     \
    VFMADD231PD(mem_1to8(RBX, R15, 2, 8), ZMM(0), ZMM(18))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


#define MICRO_TILE_4x3_MASK_SET2                            \
    /* Macro for 4x3 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1), ZMM(8))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(14))  \
    VFMADD231PD(mem_1to8(RBX, R15, 2), ZMM(1), ZMM(19))     \
    VFMADD231PD(mem_1to8(RBX, R15, 2, 8), ZMM(1), ZMM(20))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


/* Macro to perform an fx3 micro-tile computation(f<4) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx3                                      \
    /* Macro for fx3 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    VFMADD231PD(mem_1to8(RBX, R15, 2), ZMM(0), ZMM(17))     \
    VFMADD231PD(mem_1to8(RBX, R15, 2, 8), ZMM(0), ZMM(18))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x2_MASK                        \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x2_MASK_SET1                    \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_8x2_MASK_SET2                    \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(23))                 \
    VBROADCASTSD(MEM(RBX, 8), ZMM(24))              \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(3))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(4) MASK_KZ(2))        \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    VFMADD231PD(ZMM(3), ZMM(23), ZMM(15))           \
    VFMADD231PD(ZMM(4), ZMM(23), ZMM(17))           \
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(16))           \
    VFMADD231PD(ZMM(4), ZMM(24), ZMM(18))           \
    /* 4 FMAs over 2 broadcasts */                  \
    VFMADD231PD(ZMM(3), ZMM(30), ZMM(19))           \
    VFMADD231PD(ZMM(4), ZMM(30), ZMM(21))           \
    VFMADD231PD(ZMM(3), ZMM(31), ZMM(20))           \
    VFMADD231PD(ZMM(4), ZMM(31), ZMM(22))           \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x2_MASK_SET1                            \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_4x2_MASK_SET2                            \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1), ZMM(8))           \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(14))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x2                             \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13, 15)                             \
    FMA(31, 12, 14, 16)                             \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x2                              \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x2                                      \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

/* Macro to perform an fx2 micro-tile computation(f<4) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx2                                      \
    /* Macro for fx2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_12x1                             \
    /* Macro for 12x1 micro-tile evaluation   */    \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x1_SET1                         \
    /* Macro for 8x1 micro-tile evaluation   */     \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x1_SET2                         \
    /* Macro for 8x1 micro-tile evaluation   */     \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(13))                 \
    VBROADCASTSD(MEM(RBX, 8), ZMM(14))              \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(15))                      \
    VMOVUPD(MEM(RAX, 64), ZMM(16))                  \
    /* 4 FMAs over 2 broadcasts */                  \
    VFMADD231PD(ZMM(15), ZMM(13), ZMM(9))           \
    VFMADD231PD(ZMM(16), ZMM(13), ZMM(11))          \
    VFMADD231PD(ZMM(15), ZMM(14), ZMM(10))          \
    VFMADD231PD(ZMM(16), ZMM(14), ZMM(12))          \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x1                              \
    /* Macro for 4x1 micro-tile evaluation   */     \
    /* Loading A using ZMM(0) */                    \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))      \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))   \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_12x1_MASK                        \
    /* Macro for 12x1 micro-tile evaluation   */    \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x1_MASK_SET1                    \
    /* Macro for 8x1 micro-tile evaluation   */     \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x1_MASK_SET2                    \
    /* Macro for 8x1 micro-tile evaluation   */     \
    /* Broadcasting B on ZMM(3) and ZMM(4) */       \
    VBROADCASTSD(MEM(RBX), ZMM(30))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(31))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(13))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(14) MASK_KZ(2))        \
    /* 4 FMAs over 2 broadcasts */                  \
    VFMADD231PD(ZMM(13), ZMM(30), ZMM(9))           \
    VFMADD231PD(ZMM(14), ZMM(30), ZMM(11))           \
    VFMADD231PD(ZMM(13), ZMM(31), ZMM(10))           \
    VFMADD231PD(ZMM(14), ZMM(31), ZMM(12))           \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x1_MASK_SET1                     \
    /* Macro for 4x1 micro-tile evaluation   */     \
    /* Loading A using ZMM(0) */                    \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))            \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))      \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))   \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_4x1_MASK_SET2                    \
    /* Macro for 4x1 micro-tile evaluation   */     \
    /* Loading A using ZMM(0) */                    \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))            \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))      \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1), ZMM(8))   \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

/* Macro to perform an fx1 micro-tile computation(f<4) */
/* Macro assumes k(2) to have the mask for loading A */
#define MICRO_TILE_fx1                              \
    /* Macro for fx1 micro-tile evaluation   */     \
    /* Loading A using ZMM(0) */                    \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))            \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))      \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))   \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

// Macro for scaling with alpha if it is -1
// in case of 3 loads(12x? cases)
#define ALPHA_MINUS_ONE_12Z(R1, R2, R3) \
    VSUBPD(ZMM(R1), ZMM(2), ZMM(R1)) \
    VSUBPD(ZMM(R2), ZMM(2), ZMM(R2)) \
    VSUBPD(ZMM(R3), ZMM(2), ZMM(R3)) \

// Macro for scaling with alpha if it is -1
// in case of 2 loads(8x? cases)
#define ALPHA_MINUS_ONE_8Z(R1, R2) \
    VSUBPD(ZMM(R1), ZMM(2), ZMM(R1)) \
    VSUBPD(ZMM(R2), ZMM(2), ZMM(R2)) \

// Macro for scaling with alpha if it is -1
// in case of 1 loads(4x? cases)
#define ALPHA_MINUS_ONE_4Z(R1) \
    VSUBPD(ZMM(R1), ZMM(2), ZMM(R1)) \

// Macro to get the ALPHA_MINUS_ONE_? signature from the list
#define GET_ALPHA_MINUS_ONE(_1, _2, _3, NAME, ...)  NAME

// Overloaded macro ALPHA_MINUS_ONE with variable arguments
#define ALPHA_MINUS_ONE(...)  \
    GET_ALPHA_MINUS_ONE(__VA_ARGS__,  \
    ALPHA_MINUS_ONE_12Z, ALPHA_MINUS_ONE_8Z, ALPHA_MINUS_ONE_4Z)(__VA_ARGS__) \

// Macro for scaling with beta if it is -1
// in case of 3 loads(12x? cases)
#define BETA_MINUS_ONE_12Z(C, R1, I1, R2, I2, R3, I3)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3))  \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VSUBPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VSUBPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128))  \

// Macro for scaling with beta if it is -1
// in case of 2 loads(8x? cases)
#define BETA_MINUS_ONE_8Z(C, R1, I1, R2, I2)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VSUBPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \

// Macro for scaling with beta if it is -1
// in case of 1 load(4x? cases)
#define BETA_MINUS_ONE_4Z(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \

// Macro to get the BETA_MINUS_ONE_? signature from the list
#define GET_BETA_MINUS_ONE(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_MINUS_ONE with variable arguments
#define BETA_MINUS_ONE(...)  \
    GET_BETA_MINUS_ONE(__VA_ARGS__,  \
    BETA_MINUS_ONE_12Z, _0, BETA_MINUS_ONE_8Z, _1, BETA_MINUS_ONE_4Z)(__VA_ARGS__) \

// Macro for scaling with beta if it is -1
// in case of 3 loads(12x? cases)
#define BETA_MINUS_ONE_12Z_MASK(C, R1, I1, R2, I2, R3, I3)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3) MASK_(k(2)))  \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VSUBPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VSUBPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128) MASK_(k(2)))  \

// Macro for scaling with beta if it is -1
// in case of 2 loads(8x? cases)
#define BETA_MINUS_ONE_8Z_MASK(C, R1, I1, R2, I2)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VSUBPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

// Macro for scaling with beta if it is -1
// in case of 1 load(4x? cases)
#define BETA_MINUS_ONE_4Z_MASK(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(2))) \
    \
    VSUBPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C) MASK_(k(2))) \

// Macro to get the BETA_MINUS_ONE_? signature from the list
#define GET_BETA_MINUS_ONE_MASK(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_MINUS_ONE with variable arguments
#define BETA_MINUS_ONE_MASK(...)  \
    GET_BETA_MINUS_ONE_MASK(__VA_ARGS__,  \
    BETA_MINUS_ONE_12Z_MASK, _0, BETA_MINUS_ONE_8Z_MASK, _1, BETA_MINUS_ONE_4Z_MASK)(__VA_ARGS__) \

// Macro for scaling with beta if it is 1
// in case of 3 loads(12x? cases)
#define BETA_ONE_12Z(C, R1, I1, R2, I2, R3, I3)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3))  \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128))  \

// Macro for scaling with beta if it is 1
// in case of 2 loads(8x? cases)
#define BETA_ONE_8Z(C, R1, I1, R2, I2)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \

// Macro for scaling with beta if it is 1
// in case of 1 load(4x? cases)
#define BETA_ONE_4Z(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \

// Macro to get the BETA_ONE_? signature from the list
#define GET_BETA_ONE(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_ONE with variable arguments
#define BETA_ONE(...)  \
    GET_BETA_MINUS_ONE(__VA_ARGS__,  \
    BETA_ONE_12Z, _0, BETA_ONE_8Z, _1, BETA_ONE_4Z)(__VA_ARGS__) \

// Macro for scaling with beta if it is 1
// in case of 3 loads(12x? cases)
#define BETA_ONE_12Z_MASK(C, R1, I1, R2, I2, R3, I3)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2)) \
    VMOVUPD(MEM(C, 128), ZMM(R3) MASK_(k(2)))  \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64)) \
    VMOVUPD(ZMM(I3), MEM(C, 128) MASK_(k(2)))  \

// Macro for scaling with beta if it is 1
// in case of 2 loads(8x? cases)
#define BETA_ONE_8Z_MASK(C, R1, I1, R2, I2)  \
    VMOVUPD(MEM(C), ZMM(R1)) \
    VMOVUPD(MEM(C, 64), ZMM(R2) MASK_(k(2))) \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    \
    VMOVUPD(ZMM(I1), MEM(C)) \
    VMOVUPD(ZMM(I2), MEM(C, 64) MASK_(k(2))) \

// Macro for scaling with beta if it is 1
// in case of 1 load(4x? cases)
#define BETA_ONE_4Z_MASK(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(2))) \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(C) MASK_(k(2))) \

// Macro to get the BETA_ONE_? signature from the list
#define GET_BETA_ONE_MASK(_1, _2, _3, _4, _5, _6, _7, NAME, ...)  NAME

// Overloaded macro BETA_ONE with variable arguments
#define BETA_ONE_MASK(...)  \
    GET_BETA_MINUS_ONE_MASK(__VA_ARGS__,  \
    BETA_ONE_12Z_MASK, _0, BETA_ONE_8Z_MASK, _1, BETA_ONE_4Z_MASK)(__VA_ARGS__) \

// Macro for providing in-register transposition of a 4x4 block
#define TRANSPOSE_4x4(R1, R2, R3, R4) \
    VSHUFF64X2(IMM(0x88), ZMM(R2), ZMM(R1), ZMM(0)) \
    VSHUFF64X2(IMM(0x88), ZMM(R4), ZMM(R3), ZMM(2)) \
    VSHUFF64X2(IMM(0xDD), ZMM(R2), ZMM(R1), ZMM(1)) \
    VSHUFF64X2(IMM(0xDD), ZMM(R4), ZMM(R3), ZMM(3)) \
    VSHUFF64X2(IMM(0x88), ZMM(2), ZMM(0), ZMM(R1))  \
    VSHUFF64X2(IMM(0x88), ZMM(3), ZMM(1), ZMM(R2))  \
    VSHUFF64X2(IMM(0xDD), ZMM(2), ZMM(0), ZMM(R3))  \
    VSHUFF64X2(IMM(0xDD), ZMM(3), ZMM(1), ZMM(R4))  \

// Macro for providing in-register transposition of a 4x2 block
#define TRANSPOSE_4x2(R1, R2) \
    VSHUFF64X2(IMM(0x88), ZMM(R2), ZMM(R1), ZMM(0)) \
    VSHUFF64X2(IMM(0xDD), ZMM(R2), ZMM(R1), ZMM(1)) \
    VSHUFF64X2(IMM(0x88), ZMM(1), ZMM(0), ZMM(R1))  \
    VSHUFF64X2(IMM(0xDD), ZMM(1), ZMM(0), ZMM(R2))  \

// Macro for beta scaling of a 4x4 micro-tile of C when row-stored
#define BETA_GEN_ROW_4x4(C, R1, I1, R2, I2, R3, I3, R4, I4)  \
    VMOVUPD(MEM(C), ZMM(R1))    \
    VMOVUPD(MEM(C, RDI, 1), ZMM(R2))    \
    LEA(MEM(C, RDI, 2), C)  \
    VMOVUPD(MEM(C), ZMM(R3))  \
    VMOVUPD(MEM(C, RDI, 1), ZMM(R4))  \
    \
    ALPHA_GENERIC(R1, R2)   \
    ALPHA_GENERIC(R3, R4)   \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    VADDPD(ZMM(R4), ZMM(I4), ZMM(I4))  \
    \
    VMOVUPD(ZMM(I1), MEM(RCX))    \
    VMOVUPD(ZMM(I2), MEM(RCX, RDI, 1))    \
    LEA(MEM(RCX, RDI, 2), RCX)  \
    VMOVUPD(ZMM(I3), MEM(RCX))  \
    VMOVUPD(ZMM(I4), MEM(RCX, RDI, 1))  \

// Macro for beta scaling of a 1x4 micro-tile of C when row-stored
#define BETA_GEN_ROW_1x4(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1))    \
    \
    ALPHA_GENERIC(R1)   \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(RCX))    \

// Macro for beta scaling of a 4x? micro-tile of C when row-stored, using mask register
#define BETA_GEN_ROW_4x4_MASK(C, R1, I1, R2, I2, R3, I3, R4, I4)  \
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(3)))    \
    VMOVUPD(MEM(C, RDI, 1), ZMM(R2) MASK_(k(3)))    \
    LEA(MEM(C, RDI, 2), C)  \
    VMOVUPD(MEM(C), ZMM(R3) MASK_(k(3)))  \
    VMOVUPD(MEM(C, RDI, 1), ZMM(R4) MASK_(k(3)))  \
    \
    ALPHA_GENERIC(R1, R2)   \
    ALPHA_GENERIC(R3, R4)   \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    VADDPD(ZMM(R2), ZMM(I2), ZMM(I2))  \
    VADDPD(ZMM(R3), ZMM(I3), ZMM(I3))  \
    VADDPD(ZMM(R4), ZMM(I4), ZMM(I4))  \
    \
    VMOVUPD(ZMM(I1), MEM(RCX) MASK_(k(3)))    \
    VMOVUPD(ZMM(I2), MEM(RCX, RDI, 1) MASK_(k(3)))    \
    LEA(MEM(RCX, RDI, 2), RCX)  \
    VMOVUPD(ZMM(I3), MEM(RCX) MASK_(k(3)))  \
    VMOVUPD(ZMM(I4), MEM(RCX, RDI, 1) MASK_(k(3)))  \

// Macro for beta scaling of a fx4(f<4) micro-tile of C when row-stored
#define BETA_GEN_ROW_1x4_MASK(C, R1, I1)  \
    VMOVUPD(MEM(C), ZMM(R1) MASK_(k(3)))    \
    \
    ALPHA_GENERIC(R1)   \
    \
    VADDPD(ZMM(R1), ZMM(I1), ZMM(I1))  \
    \
    VMOVUPD(ZMM(I1), MEM(RCX) MASK_(k(3)))    \

// Macro for providing in-register transposition of a 2x2 block
#define TRANSPOSE_2x2(R1, R2) \
    VUNPCKLPD(YMM(R2), YMM(R1), YMM(2)) \
    VUNPCKHPD(YMM(R2), YMM(R1), YMM(3)) \
    VPERMPD(IMM(0xD8), YMM(2), YMM(2))  \
    VPERMPD(IMM(0xD8), YMM(3), YMM(3))  \
    VUNPCKLPD(YMM(3), YMM(2), YMM(R1))  \
    VUNPCKHPD(YMM(3), YMM(2), YMM(R2))  \

// Macro for beta scaling of a 2x4 micro-tile of C when row-stored
#define BETA_GEN_ROW_2x4(C, R1, I1, R2, I2) \
    VMOVUPD(MEM(C), YMM(R1))  \
    VMOVUPD(MEM(C, RSI, 2), YMM(R2))  \
    \
    VMULPD(YMM(0), YMM(R1), YMM(2))  \
    VMULPD(YMM(1), YMM(R1), YMM(R1)) \
    VMULPD(YMM(0), YMM(R2), YMM(3))  \
    VMULPD(YMM(1), YMM(R2), YMM(R2)) \
    \
    VPERMILPD(IMM(0x5), YMM(R1), YMM(R1)) \
    VPERMILPD(IMM(0x5), YMM(R2), YMM(R2)) \
    \
    VADDSUBPD(YMM(R1), YMM(2), YMM(R1)) \
    VADDSUBPD(YMM(R2), YMM(3), YMM(R2)) \
    \
    VADDPD(YMM(R1), YMM(I1), YMM(I1))   \
    VADDPD(YMM(R2), YMM(I2), YMM(I2))   \
    \
    VMOVUPD(YMM(I1), MEM(C))  \
    VMOVUPD(YMM(I2), MEM(C, RSI, 2))  \

// Macro for beta scaling of a 2x3 micro-tile of C when row-stored
#define BETA_GEN_ROW_2x3(C, R1, I1, R2, I2, R3, I3) \
    VMOVUPD(MEM(C), YMM(R1))  \
    VMOVUPD(MEM(C, RSI, 2), XMM(11))  \
    ADD(RDI, C) \
    VMOVUPD(MEM(C), YMM(R2))  \
    VMOVUPD(MEM(C, RSI, 2), XMM(12))  \
    \
    VMULPD(YMM(0), YMM(R1), YMM(2))  \
    VMULPD(YMM(1), YMM(R1), YMM(R1)) \
    VMULPD(YMM(0), YMM(R2), YMM(3))  \
    VMULPD(YMM(1), YMM(R2), YMM(R2)) \
    VMULPD(YMM(0), YMM(11), YMM(13)) \
    VMULPD(YMM(1), YMM(11), YMM(11)) \
    VMULPD(YMM(0), YMM(12), YMM(14))  \
    VMULPD(YMM(1), YMM(12), YMM(12)) \
    \
    VPERMILPD(IMM(0x55), YMM(R1), YMM(R1)) \
    VPERMILPD(IMM(0x55), YMM(R2), YMM(R2)) \
    VPERMILPD(IMM(0x55), YMM(11), YMM(11)) \
    VPERMILPD(IMM(0x55), YMM(12), YMM(12)) \
    \
    VADDSUBPD(YMM(R1), YMM(2), YMM(R1)) \
    VADDSUBPD(YMM(R2), YMM(3), YMM(R2)) \
    VADDSUBPD(YMM(11), YMM(13), YMM(11)) \
    VADDSUBPD(YMM(12), YMM(14), YMM(12)) \
    \
    VEXTRACTF128(IMM(0x1), YMM(I3), XMM(R3))  \
    \
    VADDPD(YMM(R1), YMM(I1), YMM(I1))   \
    VADDPD(YMM(R2), YMM(I2), YMM(I2))   \
    VADDPD(YMM(11), YMM(I3), YMM(I3))   \
    VADDPD(YMM(12), YMM(R3), YMM(R3))   \
    \
    VMOVUPD(YMM(I1), MEM(RCX))  \
    VMOVUPD(XMM(I3), MEM(RCX, RSI, 2))  \
    ADD(RDI, RCX) \
    VMOVUPD(YMM(I2), MEM(RCX))  \
    VMOVUPD(XMM(R3), MEM(RCX, RSI, 2))  \

// Macro for beta scaling of a 2x2 micro-tile of C when row-stored
#define BETA_GEN_ROW_2x2(C, R1, I1, R2, I2) \
    VMOVUPD(MEM(C), YMM(R1))  \
    VMOVUPD(MEM(C, RDI, 1), YMM(R2))  \
    \
    VMULPD(YMM(0), YMM(R1), YMM(2))  \
    VMULPD(YMM(1), YMM(R1), YMM(R1)) \
    VMULPD(YMM(0), YMM(R2), YMM(3))  \
    VMULPD(YMM(1), YMM(R2), YMM(R2)) \
    \
    VPERMILPD(IMM(0x55), YMM(R1), YMM(R1)) \
    VPERMILPD(IMM(0x55), YMM(R2), YMM(R2)) \
    \
    VADDSUBPD(YMM(R1), YMM(2), YMM(R1)) \
    VADDSUBPD(YMM(R2), YMM(3), YMM(R2)) \
    \
    VADDPD(YMM(R1), YMM(I1), YMM(I1))   \
    VADDPD(YMM(R2), YMM(I2), YMM(I2))   \
    \
    VMOVUPD(YMM(I1), MEM(C))  \
    VMOVUPD(YMM(I2), MEM(C, RDI, 1))  \

// Macro for beta scaling of a 2x1 micro-tile of C when row-stored
#define BETA_GEN_ROW_2x1(C, R1, I1) \
    VMOVUPD(MEM(C), XMM(14))  \
    VMOVUPD(MEM(C, RDI, 1), XMM(15))  \
    \
    VMULPD(YMM(0), YMM(14), YMM(2))  \
    VMULPD(YMM(1), YMM(14), YMM(14)) \
    VMULPD(YMM(0), YMM(15), YMM(3))  \
    VMULPD(YMM(1), YMM(15), YMM(15)) \
    \
    VPERMILPD(IMM(0x55), YMM(14), YMM(14)) \
    VPERMILPD(IMM(0x55), YMM(15), YMM(15)) \
    \
    VADDSUBPD(YMM(14), YMM(2), YMM(14)) \
    VADDSUBPD(YMM(15), YMM(3), YMM(15)) \
    \
    VADDPD(YMM(14), YMM(R1), YMM(R1))   \
    VADDPD(YMM(15), YMM(I1), YMM(I1))   \
    \
    VMOVUPD(XMM(R1), MEM(C))  \
    VMOVUPD(XMM(I1), MEM(C, RDI, 1))  \

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

#define ZGEMM_12MASKx4                                                                    \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						                                              \
    MICRO_TILE_12x4_MASK                                                                  \
    MICRO_TILE_12x4_MASK                                                                  \
    MICRO_TILE_12x4_MASK                                                                  \
    MICRO_TILE_12x4_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                      \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_8_TO_12)                                                           \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_8_TO_12)                                                          \
    LABEL(.ZKLEFTLOOP_EDGE_8_TO_12)                                                       \
																						                                              \
    MICRO_TILE_12x4_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_8_TO_12) /* Accumulating A*B over 12 registers */              \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8, 10)                                                                     \
    PERMUTE(12, 14, 16)                                                                   \
    PERMUTE(18, 20, 22)                                                                   \
    PERMUTE(24, 26, 28)                                                                   \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                            \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                       \
    ACC_COL(17, 18, 19, 20, 21, 22)                                                       \
    ACC_COL(23, 24, 25, 26, 27, 28)                                                       \
																						                                              \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_8_TO_12)                                                      \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8, 10)                                                             \
    ALPHA_MINUS_ONE(12, 14, 16)                                                           \
    ALPHA_MINUS_ONE(18, 20, 22)                                                           \
    ALPHA_MINUS_ONE(24, 26, 28)                                                           \
    JMP(.BETA_SCALE_EDGE_8_TO_12)                                                         \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_8_TO_12)                                                    \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_8_TO_12)                                                         \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8, 10)                                                               \
    ALPHA_GENERIC(12, 14, 16)                                                             \
    ALPHA_GENERIC(18, 20, 22)                                                             \
    ALPHA_GENERIC(24, 26, 28)                                                             \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_8_TO_12)                                                       \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_8_TO_12)  /* Jumping to row storage handling case */           \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_8_TO_12)                                                               \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_8_TO_12)                                                                 \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_8_TO_12)                                                       \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                           \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18, 19, 20, 21, 22)                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 23, 24, 25, 26, 27, 28)                                      \
    JMP(.CONCLUDE)                                                                        \
    LABEL(.BETA_GENERAL_EDGE_8_TO_12) /* Checking if beta == BLIS_MUL_DEFAULT */          \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8, 9, 10)                                             \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14, 15, 16)                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18, 19, 20, 21, 22)                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 23, 24, 25, 26, 27, 28)                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_8_TO_12)                                                              \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                                 \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18, 19, 20, 21, 22)                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 23, 24, 25, 26, 27, 28)                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_8_TO_12)                                                            \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                                         \
    VMOVUPD(ZMM(10), MEM(RCX, 128) MASK_(k(2)))                                           \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                                \
    VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128) MASK_(k(2)))                                   \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                             \
    VMOVUPD(ZMM(20), MEM(R9, 64))                                                         \
    VMOVUPD(ZMM(22), MEM(R9, 128) MASK_(k(2)))                                            \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9, RSI, 1))                                                     \
    VMOVUPD(ZMM(26), MEM(R9, RSI, 1, 64))                                                 \
    VMOVUPD(ZMM(28), MEM(R9, RSI, 1, 128) MASK_(k(2)))                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_8_TO_12)                                                    \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
    TRANSPOSE_4x4(10, 16, 22, 28)                                                         \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_8_TO_12)                                                           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11)                                                                         \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10)                                                                         \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9)                                                                          \
                                                                                          \
    LABEL(.UPDATE11)                                                                      \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4(R9, 7, 8, 13, 14, 19, 20, 25, 26)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4(RCX, 9, 10)                                                          \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 15, 16)                                                         \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 21, 22)                                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE10)                                                                      \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4(R9, 7, 8, 13, 14, 19, 20, 25, 26)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4(RCX, 9, 10)                                                          \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 15, 16)                                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE9)                                                                       \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4(R9, 7, 8, 13, 14, 19, 20, 25, 26)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4(RCX, 9, 10)                                                          \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_8_TO_12)                                                        \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11R)                                                                        \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10R)                                                                        \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9R)                                                                         \
                                                                                          \
    LABEL(.UPDATE11R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))   /*0*/                                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1)) /*1*/                                              \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2)) /*2*/                                              \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))  /*4*/                                              \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8)) /*8*/                                              \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9))             /*3*/                                           \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))     /*5*/                                           \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4))     /*7*/                                           \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX))        /*6*/                                               \
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4))   /*10*/                                           \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9))         /*9*/                                               \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE10R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))   /*0*/                                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1)) /*1*/                                              \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2)) /*2*/                                              \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))  /*4*/                                              \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8)) /*8*/                                              \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9))             /*3*/                                           \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))     /*5*/                                           \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4))     /*7*/                                           \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX))        /*6*/                                               \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9))         /*9*/                                               \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE9R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))   /*0*/                                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1)) /*1*/                                              \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2)) /*2*/                                              \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))  /*4*/                                              \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8)) /*8*/                                              \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9))             /*3*/                                           \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))     /*5*/                                           \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4))     /*7*/                                           \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX))        /*6*/                                               \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_12MASKx3                                                                    \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
																						                                              \
																						                                              \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						                                              \
    MICRO_TILE_12x3_MASK                                                                  \
    MICRO_TILE_12x3_MASK                                                                  \
    MICRO_TILE_12x3_MASK                                                                  \
    MICRO_TILE_12x3_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                      \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_8_TO_12)                                                           \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_8_TO_12)                                                          \
    LABEL(.ZKLEFTLOOP_EDGE_8_TO_12)                                                       \
																						                                              \
    MICRO_TILE_12x3_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_8_TO_12) /* Accumulating A*B over 12 registers */              \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8, 10)                                                                     \
    PERMUTE(12, 14, 16)                                                                   \
    PERMUTE(18, 20, 22)                                                                   \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                            \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                       \
    ACC_COL(17, 18, 19, 20, 21, 22)                                                       \
																						                                              \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_8_TO_12)                                                      \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8, 10)                                                             \
    ALPHA_MINUS_ONE(12, 14, 16)                                                           \
    ALPHA_MINUS_ONE(18, 20, 22)                                                           \
    JMP(.BETA_SCALE_EDGE_8_TO_12)                                                         \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_8_TO_12)                                                    \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_8_TO_12)                                                         \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8, 10)                                                               \
    ALPHA_GENERIC(12, 14, 16)                                                             \
    ALPHA_GENERIC(18, 20, 22)                                                             \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_8_TO_12)                                                       \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_8_TO_12)  /* Jumping to row storage handling case */           \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_8_TO_12)                                                               \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_8_TO_12)                                                                 \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_8_TO_12)                                                       \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                           \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18, 19, 20, 21, 22)                                      \
    JMP(.CONCLUDE)                                                                        \
    LABEL(.BETA_GENERAL_EDGE_8_TO_12) /* Checking if beta == BLIS_MUL_DEFAULT */          \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8, 9, 10)                                             \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14, 15, 16)                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18, 19, 20, 21, 22)                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_8_TO_12)                                                              \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                                 \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18, 19, 20, 21, 22)                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_8_TO_12)                                                            \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                                         \
    VMOVUPD(ZMM(10), MEM(RCX, 128) MASK_(k(2)))                                           \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                                \
    VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128) MASK_(k(2)))                                   \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                             \
    VMOVUPD(ZMM(20), MEM(R9, 64))                                                         \
    VMOVUPD(ZMM(22), MEM(R9, 128) MASK_(k(2)))                                            \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_8_TO_12)                                                    \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
    TRANSPOSE_4x4(10, 16, 22, 28)                                                         \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_8_TO_12)                                                           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11)                                                                         \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10)                                                                         \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9)                                                                          \
                                                                                          \
    LABEL(.UPDATE11)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 21, 22)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE10)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE9)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_8_TO_12)                                                        \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11R)                                                                        \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10R)                                                                        \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9R)                                                                         \
                                                                                          \
    LABEL(.UPDATE11R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))   /*10*/                               \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE10R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE9R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_12MASKx2                                                                    \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
																						                                              \
																						                                              \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                             \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						                                              \
    MICRO_TILE_12x2_MASK                                                                  \
    MICRO_TILE_12x2_MASK                                                                  \
    MICRO_TILE_12x2_MASK                                                                  \
    MICRO_TILE_12x2_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                      \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_8_TO_12)                                                           \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_8_TO_12)                                                          \
    LABEL(.ZKLEFTLOOP_EDGE_8_TO_12)                                                       \
																						                                              \
    MICRO_TILE_12x2_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_8_TO_12) /* Accumulating A*B over 12 registers */              \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8, 10)                                                                     \
    PERMUTE(12, 14, 16)                                                                   \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                            \
    ACC_COL(11, 12, 13, 14, 15, 16)                                                       \
																						                                              \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_8_TO_12)                                                      \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8, 10)                                                             \
    ALPHA_MINUS_ONE(12, 14, 16)                                                           \
    JMP(.BETA_SCALE_EDGE_8_TO_12)                                                         \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_8_TO_12)                                                    \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_8_TO_12)                                                         \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8, 10)                                                               \
    ALPHA_GENERIC(12, 14, 16)                                                             \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_8_TO_12)                                                       \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_8_TO_12)  /* Jumping to row storage handling case */           \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_8_TO_12)                                                               \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_8_TO_12)                                                                 \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_8_TO_12)                                                       \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                           \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                      \
    JMP(.CONCLUDE)                                                                        \
    LABEL(.BETA_GENERAL_EDGE_8_TO_12) /* Checking if beta == BLIS_MUL_DEFAULT */          \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8, 9, 10)                                             \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14, 15, 16)                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_8_TO_12)                                                              \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                                 \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14, 15, 16)                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_8_TO_12)                                                            \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                                         \
    VMOVUPD(ZMM(10), MEM(RCX, 128) MASK_(k(2)))                                           \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                                \
    VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128) MASK_(k(2)))                                   \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_8_TO_12)                                                    \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
		TRANSPOSE_4x4(10, 16, 22, 28)																				                  \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_8_TO_12)                                                           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11)                                                                         \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10)                                                                         \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9)                                                                          \
                                                                                          \
    LABEL(.UPDATE11)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 21, 22)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE10)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE9)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_8_TO_12)                                                        \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11R)                                                                        \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10R)                                                                        \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9R)                                                                         \
                                                                                          \
    LABEL(.UPDATE11R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))   /*10*/                               \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE10R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE9R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_12MASKx1                                                                    \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)   													                                              \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						                                              \
    MICRO_TILE_12x1_MASK                                                                  \
    MICRO_TILE_12x1_MASK                                                                  \
    MICRO_TILE_12x1_MASK                                                                  \
    MICRO_TILE_12x1_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                      \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_8_TO_12)                                                           \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_8_TO_12)                                                          \
    LABEL(.ZKLEFTLOOP_EDGE_8_TO_12)                                                       \
																						                                              \
    MICRO_TILE_12x1_MASK                                                                  \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_8_TO_12) /* Accumulating A*B over 12 registers */              \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8, 10)                                                                     \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8, 9, 10)                                                            \
																						                                              \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_8_TO_12)                                                      \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8, 10)                                                             \
    JMP(.BETA_SCALE_EDGE_8_TO_12)                                                         \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_8_TO_12)                                                    \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_8_TO_12)                                                         \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8, 10)                                                               \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_8_TO_12)                                                       \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_8_TO_12)  /* Jumping to row storage handling case */           \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_8_TO_12)                                                               \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_8_TO_12)                                                                 \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_8_TO_12)                                                       \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                           \
    JMP(.CONCLUDE)                                                                        \
    LABEL(.BETA_GENERAL_EDGE_8_TO_12) /* Checking if beta == BLIS_MUL_DEFAULT */          \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8, 9, 10)                                             \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_8_TO_12)                                                              \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8, 9, 10)                                                 \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_8_TO_12)                                                            \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                                         \
    VMOVUPD(ZMM(10), MEM(RCX, 128) MASK_(k(2)))                                           \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_8_TO_12)                                                    \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
		TRANSPOSE_4x4(10, 16, 22, 28)																				                  \
                                                                                          \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_8_TO_12)                                                           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11)                                                                         \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10)                                                                         \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9)                                                                          \
                                                                                          \
    LABEL(.UPDATE11)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 21, 22)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE10)                                                                      \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 15, 16)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE9)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
                                                                                          \
    BETA_GEN_ROW_1x4_MASK(RCX, 9, 10)                                                     \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_8_TO_12)                                                        \
    CMP(imm(0xb), R11)                                                                    \
    JZ(.UPDATE11R)                                                                        \
    CMP(imm(0xa), R11)                                                                    \
    JZ(.UPDATE10R)                                                                        \
    CMP(imm(0x9), R11)                                                                    \
    JZ(.UPDATE9R)                                                                         \
                                                                                          \
    LABEL(.UPDATE11R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))   /*10*/                               \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE10R)                                                                     \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))         /*9*/                                   \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE9R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))   /*0*/                                         \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3))) /*1*/                                  \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3))) /*2*/                                  \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))  /*4*/                                  \
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3))) /*8*/                                  \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)        /* RCX = RCX + 6*rs_c  */                           \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))             /*3*/                               \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))     /*5*/                               \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))     /*7*/                               \
																						                                              \
    LEA(MEM(R9, RDI, 4), R9)                                                              \
    LEA(MEM(R9, RDI, 2), R9)          /* R9 = RCX + 9*rs_c */                             \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))        /*6*/                                   \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)


#define ZGEMM_8x4                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)                                                   \
                                                                              \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)                                                   \
                                                                              \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)                                                   \
                                                                              \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)                                                   \
                                                                              \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)                                                   \
                                                                              \
                                                                              \
    MOV(VAR(v), R9)                                                           \
    VBROADCASTSD(MEM(R9), ZMM(29))                                            \
                                                                              \
                                                                              \
    RESET_REGISTERS                                                           \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x4                                                            \
    MICRO_TILE_8x4                                                            \
    MICRO_TILE_8x4                                                            \
    MICRO_TILE_8x4                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINEDGE8)                                                     \
                                                                              \
                                                                              \
    LABEL(.ZKLEFTEDGE8)                                                       \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEEDGE8)                                                      \
    LABEL(.ZKLEFTLOOPEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x4                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPEDGE8)                                                     \
                                                                              \
    LABEL(.ACCUMULATEEDGE8)                                                   \
                                                                              \
    PERMUTE(6, 8)                                                             \
    PERMUTE(12, 14)                                                           \
    PERMUTE(18, 20)                                                           \
    PERMUTE(24, 26)                                                           \
                                                                              \
    ACC_COL(5, 6, 7, 8)                                                       \
    ACC_COL(11, 12, 13, 14)                                                   \
    ACC_COL(17, 18, 19, 20)                                                   \
    ACC_COL(23, 24, 25, 26)                                                   \
                                                                              \
    /* A*B is accumulated over the ZMM registers as follows :*/               \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /*  ZMM8  ZMM14  ZMM20  ZMM26 */                                          \
    /* */                                                                     \
                                                                              \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
                                                                              \
    ALPHA_GENERIC(6, 8)                                                       \
    ALPHA_GENERIC(12, 14)                                                     \
    ALPHA_GENERIC(18, 20)                                                     \
    ALPHA_GENERIC(24, 26)                                                     \
                                                                              \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEEDGE8)                                                   \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CEDGE8)  /* Jumping to row storage handling case */       \
                                                                              \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREEDGE8)                                                           \
                                                                              \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                             \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                         \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 17, 18, 19, 20)                                         \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 23, 24, 25, 26)                                         \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREEDGE8)                                                        \
    LEA(MEM(RCX, RSI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                             \
                                                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                    \
                                                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                 \
    VMOVUPD(ZMM(20), MEM(R9, 64))                                             \
                                                                              \
    VMOVUPD(ZMM(24), MEM(R9, RSI, 1))                                         \
    VMOVUPD(ZMM(26), MEM(R9, RSI, 1, 64))                                     \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CEDGE8)                                                \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    TRANSPOSE_4x4(8, 14, 20, 26)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
    /*  ZMM8 */                                                               \
    /*  ZMM14 */                                                              \
    /*  ZMM20 */                                                              \
    /*  ZMM26 */                                                              \
    /* */                                                                     \
                                                                              \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWEDGE8)                                                       \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    LEA(MEM(R9, RDI, 2), R9)                                                  \
    BETA_GEN_ROW_4x4(R9, 7, 8, 13, 14, 19, 20, 25, 26)                        \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWEDGE8)                                                    \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    LEA(MEM(R9, RDI, 1), R9)                                                  \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))                                        \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))                                         \
                                                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    VMOVUPD(ZMM(24), MEM(R9))                                                 \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))                                         \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4))                                         \
                                                                              \
    VMOVUPD(ZMM(20), MEM(RCX))                                                \
                                                                              \
    JMP(.CONCLUDE)

#define ZGEMM_8x3                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)                                                   \
                                                                              \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)                                                   \
                                                                              \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)                                                   \
                                                                              \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)                                                   \
                                                                              \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)                                                   \
                                                                              \
                                                                              \
    MOV(VAR(v), R9)                                                           \
    VBROADCASTSD(MEM(R9), ZMM(29))                                            \
                                                                              \
                                                                              \
    RESET_REGISTERS                                                           \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x3                                                            \
    MICRO_TILE_8x3                                                            \
    MICRO_TILE_8x3                                                            \
    MICRO_TILE_8x3                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINEDGE8)                                                     \
                                                                              \
                                                                              \
    LABEL(.ZKLEFTEDGE8)                                                       \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEEDGE8)                                                      \
    LABEL(.ZKLEFTLOOPEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x3                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPEDGE8)                                                     \
                                                                              \
    LABEL(.ACCUMULATEEDGE8)                                                   \
                                                                              \
    PERMUTE(6, 8)                                                             \
    PERMUTE(12, 14)                                                           \
    PERMUTE(18, 20)                                                           \
    PERMUTE(24, 26)                                                           \
                                                                              \
    ACC_COL(5, 6, 7, 8)                                                       \
    ACC_COL(11, 12, 13, 14)                                                   \
    ACC_COL(17, 18, 19, 20)                                                   \
                                                                              \
    /* A*B is accumulated over the ZMM registers as follows :*/               \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /*  ZMM8  ZMM14  ZMM20  ZMM26 */                                          \
    /* */                                                                     \
                                                                              \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
                                                                              \
    ALPHA_GENERIC(6, 8)                                                       \
    ALPHA_GENERIC(12, 14)                                                     \
    ALPHA_GENERIC(18, 20)                                                     \
                                                                              \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEEDGE8)                                                   \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CEDGE8)  /* Jumping to row storage handling case */       \
                                                                              \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREEDGE8)                                                           \
                                                                              \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                             \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                         \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 17, 18, 19, 20)                                         \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREEDGE8)                                                        \
    LEA(MEM(RCX, RSI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                             \
                                                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                    \
                                                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                 \
    VMOVUPD(ZMM(20), MEM(R9, 64))                                             \
                                                                              \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CEDGE8)                                                \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    TRANSPOSE_4x4(8, 14, 20, 26)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
    /*  ZMM8 */                                                               \
    /*  ZMM14 */                                                              \
    /*  ZMM20 */                                                              \
    /*  ZMM26 */                                                              \
    /* */                                                                     \
                                                                              \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWEDGE8)                                                       \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    LEA(MEM(R9, RDI, 2), R9)                                                  \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                        \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWEDGE8)                                                    \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    LEA(MEM(R9, RDI, 1), R9)                                                  \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))                             \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                                    \
                                                                              \
    JMP(.CONCLUDE)


#define ZGEMM_8x2                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)                                                   \
                                                                              \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)                                                   \
                                                                              \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)                                                   \
                                                                              \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)                                                   \
                                                                              \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)                                                   \
                                                                              \
                                                                              \
    MOV(VAR(v), R9)                                                           \
    VBROADCASTSD(MEM(R9), ZMM(29))                                            \
                                                                              \
                                                                              \
    RESET_REGISTERS                                                           \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x2                                                            \
    MICRO_TILE_8x2                                                            \
    MICRO_TILE_8x2                                                            \
    MICRO_TILE_8x2                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINEDGE8)                                                     \
                                                                              \
                                                                              \
    LABEL(.ZKLEFTEDGE8)                                                       \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEEDGE8)                                                      \
    LABEL(.ZKLEFTLOOPEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x2                                                            \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPEDGE8)                                                     \
                                                                              \
    LABEL(.ACCUMULATEEDGE8)                                                   \
                                                                              \
    PERMUTE(6, 8)                                                             \
    PERMUTE(12, 14)                                                           \
                                                                              \
    ACC_COL(5, 6, 7, 8)                                                       \
    ACC_COL(11, 12, 13, 14)                                                   \
                                                                              \
    /* A*B is accumulated over the ZMM registers as follows :*/               \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /*  ZMM8  ZMM14  ZMM20  ZMM26 */                                          \
    /* */                                                                     \
                                                                              \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
                                                                              \
    ALPHA_GENERIC(6, 8)                                                       \
    ALPHA_GENERIC(12, 14)                                                     \
                                                                              \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEEDGE8)                                                   \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CEDGE8)  /* Jumping to row storage handling case */       \
                                                                              \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREEDGE8)                                                           \
                                                                              \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                             \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12, 13, 14)                                         \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREEDGE8)                                                        \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                             \
                                                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))                                    \
                                                                              \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CEDGE8)                                                \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    TRANSPOSE_4x4(8, 14, 20, 26)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
    /*  ZMM8 */                                                               \
    /*  ZMM14 */                                                              \
    /*  ZMM20 */                                                              \
    /*  ZMM26 */                                                              \
    /* */                                                                     \
                                                                              \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWEDGE8)                                                       \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    LEA(MEM(R9, RDI, 2), R9)                                                  \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                        \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWEDGE8)                                                    \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    LEA(MEM(R9, RDI, 1), R9)                                                  \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))                             \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                                    \
                                                                              \
    JMP(.CONCLUDE)


#define ZGEMM_8x1                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)                                                   \
                                                                              \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)                                                   \
                                                                              \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)                                                   \
                                                                              \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)                                                   \
                                                                              \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)                                                   \
                                                                              \
                                                                              \
    MOV(VAR(v), R9)                                                           \
    VBROADCASTSD(MEM(R9), ZMM(29))                                            \
                                                                              \
                                                                              \
    RESET_REGISTERS                                                           \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x1_SET1                                                       \
    MICRO_TILE_8x1_SET2                                                       \
    MICRO_TILE_8x1_SET1                                                       \
    MICRO_TILE_8x1_SET2                                                       \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINEDGE8)                                                     \
                                                                              \
                                                                              \
    LABEL(.ZKLEFTEDGE8)                                                       \
    VADDPD(ZMM(5), ZMM(9), ZMM(5))                                            \
    VADDPD(ZMM(6), ZMM(10), ZMM(6))                                           \
    VADDPD(ZMM(7), ZMM(11), ZMM(7))                                           \
    VADDPD(ZMM(8), ZMM(12), ZMM(8))                                           \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEEDGE8)                                                      \
    LABEL(.ZKLEFTLOOPEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x1_SET1                                                       \
                                                                              \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPEDGE8)                                                     \
                                                                              \
    LABEL(.ACCUMULATEEDGE8)                                                   \
                                                                              \
    PERMUTE(6, 8)                                                             \
                                                                              \
    ACC_COL(5, 6, 7, 8)                                                       \
                                                                              \
    /* A*B is accumulated over the ZMM registers as follows :*/               \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /*  ZMM8  ZMM14  ZMM20  ZMM26 */                                          \
    /* */                                                                     \
                                                                              \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
                                                                              \
    ALPHA_GENERIC(6, 8)                                                       \
                                                                              \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEEDGE8)                                                   \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CEDGE8)  /* Jumping to row storage handling case */       \
                                                                              \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREEDGE8)                                                           \
                                                                              \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6, 7, 8)                                             \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREEDGE8)                                                        \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(8), MEM(RCX, 64))                                             \
                                                                              \
                                                                              \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CEDGE8)                                                \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    TRANSPOSE_4x4(8, 14, 20, 26)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
    /*  ZMM8 */                                                               \
    /*  ZMM14 */                                                              \
    /*  ZMM20 */                                                              \
    /*  ZMM26 */                                                              \
    /* */                                                                     \
                                                                              \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWEDGE8)                                                       \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
                                                                              \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                   \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    LEA(MEM(R9, RDI, 2), R9)                                                  \
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)                   \
    JMP(.CONCLUDE)                                                            \
                                                                              \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWEDGE8)                                                    \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    LEA(MEM(R9, RDI, 1), R9)                                                  \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                \
    LEA(MEM(RCX, RDI, 2), RCX)                                                \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))                             \
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))                             \
                                                                              \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                                    \
                                                                              \
    JMP(.CONCLUDE)


#define ZGEMM_8MASKx4                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)  													                                              \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						                                              \
    MICRO_TILE_8x4_MASK                                                                   \
    MICRO_TILE_8x4_MASK                                                                   \
    MICRO_TILE_8x4_MASK                                                                   \
    MICRO_TILE_8x4_MASK                                                                   \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_4_TO_8)                                                            \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_4_TO_8)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_4_TO_8)                                                        \
																						                                              \
    MICRO_TILE_8x4_MASK                                                                   \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_4_TO_8) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8)                                                                         \
    PERMUTE(12, 14)                                                                       \
    PERMUTE(18, 20)                                                                       \
    PERMUTE(24, 26)                                                                       \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8)                                                                   \
    ACC_COL(11, 12, 13, 14)                                                               \
    ACC_COL(17, 18, 19, 20)                                                               \
    ACC_COL(23, 24, 25, 26)                                                               \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_4_TO_8)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8)                                                                 \
    ALPHA_MINUS_ONE(12, 14)                                                               \
    ALPHA_MINUS_ONE(18, 20)                                                               \
    ALPHA_MINUS_ONE(24, 26)                                                               \
    JMP(.BETA_SCALE_EDGE_4_TO_8)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_4_TO_8)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_4_TO_8)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8)                                                                   \
    ALPHA_GENERIC(12, 14)                                                                 \
    ALPHA_GENERIC(18, 20)                                                                 \
    ALPHA_GENERIC(24, 26)                                                                 \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_4_TO_8)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_4_TO_8)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_4_TO_8)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_4_TO_8)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_4_TO_8)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8)                                                  \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14)                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18, 19, 20)                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 23, 24, 25, 26)                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_4_TO_8) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14)                                                \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18, 19, 20)                                                \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 23, 24, 25, 26)                                                \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_4_TO_8)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18, 19, 20)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 23, 24, 25, 26)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_4_TO_8)                                                             \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64) MASK_(k(2)))                                             \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64) MASK_(k(2)))                                    \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                             \
    VMOVUPD(ZMM(20), MEM(R9, 64) MASK_(k(2)))                                             \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9, RSI, 1))                                                     \
    VMOVUPD(ZMM(26), MEM(R9, RSI, 1, 64) MASK_(k(2)))                                     \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_4_TO_8)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_4_TO_8)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7)                                                                          \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6)                                                                          \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5)                                                                          \
                                                                                          \
    LABEL(.UPDATE7)                                                                       \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4(RCX, 7, 8)                                                           \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 13, 14)                                                         \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 19, 20)                                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE6)                                                                       \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4(RCX, 7, 8)                                                           \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 13, 14)                                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE5)                                                                       \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                                    \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4(RCX, 7, 8)                                                           \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_4_TO_8)                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7R)                                                                         \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6R)                                                                         \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5R)                                                                         \
                                                                                          \
    LABEL(.UPDATE7R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))                  /*0*/                                      \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))         /*1*/                                      \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))         /*2*/                                      \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))          /*4*/                                      \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9))                  /*3*/                                      \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))          /*5*/                                      \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    VMOVUPD(ZMM(20), MEM(RCX))                  /*6*/                                     \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE6R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))                  /*0*/                                      \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))         /*1*/                                      \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))         /*2*/                                      \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))          /*4*/                                      \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9))                  /*3*/                                      \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2))          /*5*/                                      \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE5R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX))                  /*0*/                                      \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))         /*1*/                                      \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))         /*2*/                                      \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))          /*4*/                                      \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9))                  /*3*/                                      \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_8MASKx3                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						                                              \
    MICRO_TILE_8x3_MASK                                                                   \
    MICRO_TILE_8x3_MASK                                                                   \
    MICRO_TILE_8x3_MASK                                                                   \
    MICRO_TILE_8x3_MASK                                                                   \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_4_TO_8)                                                            \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_4_TO_8)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_4_TO_8)                                                        \
																						                                              \
    MICRO_TILE_8x3_MASK                                                                   \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_4_TO_8) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8)                                                                         \
    PERMUTE(12, 14)                                                                       \
    PERMUTE(18, 20)                                                                       \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8)                                                                   \
    ACC_COL(11, 12, 13, 14)                                                               \
    ACC_COL(17, 18, 19, 20)                                                               \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_4_TO_8)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8)                                                                 \
    ALPHA_MINUS_ONE(12, 14)                                                               \
    ALPHA_MINUS_ONE(18, 20)                                                               \
    JMP(.BETA_SCALE_EDGE_4_TO_8)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_4_TO_8)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_4_TO_8)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8)                                                                   \
    ALPHA_GENERIC(12, 14)                                                                 \
    ALPHA_GENERIC(18, 20)                                                                 \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_4_TO_8)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_4_TO_8)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_4_TO_8)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_4_TO_8)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_4_TO_8)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8)                                                  \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14)                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18, 19, 20)                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_4_TO_8) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14)                                                \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18, 19, 20)                                                \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_4_TO_8)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18, 19, 20)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_4_TO_8)                                                             \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64) MASK_(k(2)))                                             \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64) MASK_(k(2)))                                    \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9))                                                             \
    VMOVUPD(ZMM(20), MEM(R9, 64) MASK_(k(2)))                                             \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_4_TO_8)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_4_TO_8)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7)                                                                          \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6)                                                                          \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5)                                                                          \
                                                                                          \
    LABEL(.UPDATE7)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 19, 20)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE6)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE5)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_4_TO_8)                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7R)                                                                         \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6R)                                                                         \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5R)                                                                         \
                                                                                          \
    LABEL(.UPDATE7R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                  /*6*/                         \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE6R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE5R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)


#define ZGEMM_8MASKx2                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)  													                                              \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						                                              \
    MICRO_TILE_8x2_MASK_SET1                                                              \
    MICRO_TILE_8x2_MASK_SET2                                                              \
    MICRO_TILE_8x2_MASK_SET1                                                              \
    MICRO_TILE_8x2_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_4_TO_8)                                                            \
    VADDPD(ZMM(5), ZMM(15), ZMM(5))                                                       \
    VADDPD(ZMM(6), ZMM(16), ZMM(6))                                                       \
    VADDPD(ZMM(7), ZMM(17), ZMM(7))                                                       \
    VADDPD(ZMM(8), ZMM(18), ZMM(8))                                                       \
    VADDPD(ZMM(11), ZMM(19), ZMM(11))                                                     \
    VADDPD(ZMM(12), ZMM(20), ZMM(12))                                                     \
    VADDPD(ZMM(13), ZMM(21), ZMM(13))                                                     \
    VADDPD(ZMM(14), ZMM(22), ZMM(14))                                                     \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_4_TO_8)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_4_TO_8)                                                        \
																						                                              \
    MICRO_TILE_8x2_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_4_TO_8) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8)                                                                         \
    PERMUTE(12, 14)                                                                       \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8)                                                                   \
    ACC_COL(11, 12, 13, 14)                                                               \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_4_TO_8)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8)                                                                 \
    ALPHA_MINUS_ONE(12, 14)                                                               \
    JMP(.BETA_SCALE_EDGE_4_TO_8)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_4_TO_8)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_4_TO_8)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8)                                                                   \
    ALPHA_GENERIC(12, 14)                                                                 \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_4_TO_8)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_4_TO_8)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_4_TO_8)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_4_TO_8)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_4_TO_8)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8)                                                  \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12, 13, 14)                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_4_TO_8) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8)                                                    \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12, 13, 14)                                                \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_4_TO_8)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12, 13, 14)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_4_TO_8)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64) MASK_(k(2)))                                             \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                                    \
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64) MASK_(k(2)))                                    \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_4_TO_8)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_4_TO_8)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7)                                                                          \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6)                                                                          \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5)                                                                          \
                                                                                          \
    LABEL(.UPDATE7)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 19, 20)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE6)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE5)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_4_TO_8)                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7R)                                                                         \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6R)                                                                         \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5R)                                                                         \
                                                                                          \
    LABEL(.UPDATE7R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                  /*6*/                         \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE6R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE5R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_8MASKx1                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						                                              \
    MICRO_TILE_8x1_MASK_SET1                                                              \
    MICRO_TILE_8x1_MASK_SET2                                                              \
    MICRO_TILE_8x1_MASK_SET1                                                              \
    MICRO_TILE_8x1_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_4_TO_8)                                                            \
    VADDPD(ZMM(5), ZMM(9), ZMM(5))                                                        \
    VADDPD(ZMM(6), ZMM(10), ZMM(6))                                                       \
    VADDPD(ZMM(7), ZMM(11), ZMM(7))                                                       \
    VADDPD(ZMM(8), ZMM(12), ZMM(8))                                                       \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_4_TO_8)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_4_TO_8)                                                        \
																						                                              \
    MICRO_TILE_8x1_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_4_TO_8) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6, 8)                                                                         \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6, 7, 8)                                                                   \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_4_TO_8)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6, 8)                                                                 \
    JMP(.BETA_SCALE_EDGE_4_TO_8)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_4_TO_8)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_4_TO_8)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6, 8)                                                                   \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_4_TO_8)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_4_TO_8)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_4_TO_8)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_4_TO_8)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_4_TO_8)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6, 7, 8)                                                  \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_4_TO_8) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6, 7, 8)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_4_TO_8)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6, 7, 8)                                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_4_TO_8)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(8), MEM(RCX, 64) MASK_(k(2)))                                             \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_4_TO_8)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
    TRANSPOSE_4x4(8, 14, 20, 26)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_4_TO_8)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7)                                                                          \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6)                                                                          \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5)                                                                          \
                                                                                          \
    LABEL(.UPDATE7)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 19, 20)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE6)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 13, 14)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE5)                                                                       \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                               \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    LEA(MEM(R9, RDI, 2), R9)                                                              \
    BETA_GEN_ROW_1x4_MASK(RCX, 7, 8)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_4_TO_8)                                                         \
    CMP(imm(0x7), R11)                                                                    \
    JZ(.UPDATE7R)                                                                         \
    CMP(imm(0x6), R11)                                                                    \
    JZ(.UPDATE6R)                                                                         \
    CMP(imm(0x5), R11)                                                                    \
    JZ(.UPDATE5R)                                                                         \
                                                                                          \
    LABEL(.UPDATE7R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
    LEA(MEM(RCX, RDI, 4), RCX)                                                            \
    LEA(MEM(RCX, RDI, 2), RCX)                                                            \
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))                  /*6*/                         \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE6R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))          /*5*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE5R)                                                                      \
    LEA(MEM(RCX, RDI, 2), R9)                                                             \
    LEA(MEM(R9, RDI, 1), R9)          /* R9 = RCX + 3*rs_c */                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                  /*0*/                          \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))         /*1*/                          \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))         /*2*/                          \
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))          /*4*/                          \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))                  /*3*/                          \
																						                                              \
																						                                              \
    JMP(.CONCLUDE)


#define ZGEMM_4x4                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */               \
																			                                        \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */               \
																			                                        \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */               \
																			                                        \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */               \
																			                                        \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */               \
																			                                        \
    /* Intermediate register for complex arithmetic */                        \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                       \
    VBROADCASTSD(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */        \
																			                                        \
    /* Resetting all scratch registers */                                     \
    RESET_REGISTERS                                                           \
																			                                        \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x4                                                            \
    MICRO_TILE_4x4                                                            \
    MICRO_TILE_4x4                                                            \
    MICRO_TILE_4x4                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINZGEMM_4)                                                   \
																			                                        \
    /* Remainder loop for k */                                                \
    LABEL(.ZKLEFTZGEMM_4)                                                     \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEZGEMM_4)                                                    \
    LABEL(.ZKLEFTLOOPZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x4                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPZGEMM_4)                                                   \
																			                                        \
    LABEL(.ACCUMULATEZGEMM_4) /* Accumulating A*B over 4 registers */         \
    /* Shuffling the registers FMAed with imaginary components in B. */       \
    PERMUTE(6)                                                                \
    PERMUTE(12)                                                               \
    PERMUTE(18)                                                               \
    PERMUTE(24)                                                               \
																			                                        \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */                \
    ACC_COL(5, 6)                                                             \
    ACC_COL(11, 12)                                                           \
    ACC_COL(17, 18)                                                           \
    ACC_COL(23, 24)                                                           \
																			                                        \
    /* A*B is accumulated over the ZMM registers as follows : */              \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /* */                                                                     \
																			                                        \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
																			                                        \
    ALPHA_GENERIC(6)                                                          \
    ALPHA_GENERIC(12)                                                         \
    ALPHA_GENERIC(18)                                                         \
    ALPHA_GENERIC(24)                                                         \
																			                                        \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEZGEMM_4)                                                 \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CZGEMM_4)  /* Jumping to row storage handling case */     \
																			                                        \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREZGEMM_4)                                                         \
																			                                        \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6)                                                   \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12)                                                 \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 17, 18)                                                 \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 23, 24)                                                 \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREZGEMM_4)                                                      \
    LEA(MEM(RCX, RSI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
																			                                        \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
																			                                        \
    VMOVUPD(ZMM(18), MEM(R9))                                                 \
																			                                        \
    VMOVUPD(ZMM(24), MEM(R9, RSI, 1))                                         \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CZGEMM_4)                                              \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
																			                                        \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWZGEMM_4)                                                     \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWZGEMM_4)                                                  \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))                                        \
    VMOVUPD(ZMM(24), MEM(R9, RDI, 1))                                         \
																			                                        \
    JMP(.CONCLUDE)

#define ZGEMM_4x3                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */               \
																			                                        \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */               \
																			                                        \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */               \
																			                                        \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */               \
																			                                        \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */               \
																			                                        \
    /* Intermediate register for complex arithmetic */                        \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                       \
    VBROADCASTSD(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */        \
																			                                        \
    /* Resetting all scratch registers */                                     \
    RESET_REGISTERS                                                           \
																			                                        \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x3                                                            \
    MICRO_TILE_4x3                                                            \
    MICRO_TILE_4x3                                                            \
    MICRO_TILE_4x3                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINZGEMM_4)                                                   \
																			                                        \
    /* Remainder loop for k */                                                \
    LABEL(.ZKLEFTZGEMM_4)                                                     \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEZGEMM_4)                                                    \
    LABEL(.ZKLEFTLOOPZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x3                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPZGEMM_4)                                                   \
																			                                        \
    LABEL(.ACCUMULATEZGEMM_4) /* Accumulating A*B over 4 registers */         \
    /* Shuffling the registers FMAed with imaginary components in B. */       \
    PERMUTE(6)                                                                \
    PERMUTE(12)                                                               \
    PERMUTE(18)                                                               \
																			                                        \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */                \
    ACC_COL(5, 6)                                                             \
    ACC_COL(11, 12)                                                           \
    ACC_COL(17, 18)                                                           \
																			                                        \
    /* A*B is accumulated over the ZMM registers as follows : */              \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /* */                                                                     \
																			                                        \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
																			                                        \
    ALPHA_GENERIC(6)                                                          \
    ALPHA_GENERIC(12)                                                         \
    ALPHA_GENERIC(18)                                                         \
																			                                        \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEZGEMM_4)                                                 \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CZGEMM_4)  /* Jumping to row storage handling case */     \
																			                                        \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREZGEMM_4)                                                         \
																			                                        \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6)                                                   \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12)                                                 \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 17, 18)                                                 \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREZGEMM_4)                                                      \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
																			                                        \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
																			                                        \
    VMOVUPD(ZMM(18), MEM(RCX,RSI, 2))                                         \
																			                                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CZGEMM_4)                                              \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
																			                                        \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWZGEMM_4)                                                     \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWZGEMM_4)                                                  \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(24), MEM(R9, RDI, 1) MASK_(k(3)))                             \
																			                                        \
    JMP(.CONCLUDE)

#define ZGEMM_4x2                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */               \
																			                                        \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */               \
																			                                        \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */               \
																			                                        \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */               \
																			                                        \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */               \
																			                                        \
    /* Intermediate register for complex arithmetic */                        \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                       \
    VBROADCASTSD(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */        \
																			                                        \
    /* Resetting all scratch registers */                                     \
    RESET_REGISTERS                                                           \
																			                                        \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x2                                                            \
    MICRO_TILE_4x2                                                            \
    MICRO_TILE_4x2                                                            \
    MICRO_TILE_4x2                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINZGEMM_4)                                                   \
																			                                        \
    /* Remainder loop for k */                                                \
    LABEL(.ZKLEFTZGEMM_4)                                                     \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEZGEMM_4)                                                    \
    LABEL(.ZKLEFTLOOPZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x2                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPZGEMM_4)                                                   \
																			                                        \
    LABEL(.ACCUMULATEZGEMM_4) /* Accumulating A*B over 4 registers */         \
    /* Shuffling the registers FMAed with imaginary components in B. */       \
    PERMUTE(6)                                                                \
    PERMUTE(12)                                                               \
																			                                        \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */                \
    ACC_COL(5, 6)                                                             \
    ACC_COL(11, 12)                                                           \
																			                                        \
    /* A*B is accumulated over the ZMM registers as follows : */              \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /* */                                                                     \
																			                                        \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
																			                                        \
    ALPHA_GENERIC(6)                                                          \
    ALPHA_GENERIC(12)                                                         \
																			                                        \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEZGEMM_4)                                                 \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CZGEMM_4)  /* Jumping to row storage handling case */     \
																			                                        \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREZGEMM_4)                                                         \
																			                                        \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6)                                                   \
    ADD(RSI, RCX)                                                             \
    BETA_GENERIC(RCX, 11, 12)                                                 \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREZGEMM_4)                                                      \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
																			                                        \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))                                        \
																			                                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CZGEMM_4)                                              \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
																			                                        \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWZGEMM_4)                                                     \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWZGEMM_4)                                                  \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(24), MEM(R9, RDI, 1) MASK_(k(3)))                             \
																			                                        \
    JMP(.CONCLUDE)

#define ZGEMM_4x1                                                             \
    MOV(VAR(cs_a), R13)                                                       \
    LEA(MEM(, R13, 8), R13)                                                   \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */               \
																			                                        \
    MOV(VAR(rs_b), R14)                                                       \
    LEA(MEM(, R14, 8), R14)                                                   \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */               \
																			                                        \
    MOV(VAR(cs_b), R15)                                                       \
    LEA(MEM(, R15, 8), R15)                                                   \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */               \
																			                                        \
    MOV(VAR(rs_c), RDI)                                                       \
    LEA(MEM(, RDI, 8), RDI)                                                   \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */               \
																			                                        \
    MOV(VAR(cs_c), RSI)                                                       \
    LEA(MEM(, RSI, 8), RSI)                                                   \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */               \
																			                                        \
    /* Intermediate register for complex arithmetic */                        \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                       \
    VBROADCASTSD(MEM(R9), ZMM(29)) /* Broadcasting 1.0 over ZMM(29) */        \
																			                                        \
    /* Resetting all scratch registers */                                     \
    RESET_REGISTERS                                                           \
																			                                        \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x1                                                            \
    MICRO_TILE_4x1                                                            \
    MICRO_TILE_4x1                                                            \
    MICRO_TILE_4x1                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKITERMAINZGEMM_4)                                                   \
																			                                        \
    /* Remainder loop for k */                                                \
    LABEL(.ZKLEFTZGEMM_4)                                                     \
    MOV(VAR(k_left), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ACCUMULATEZGEMM_4)                                                    \
    LABEL(.ZKLEFTLOOPZGEMM_4)                                                 \
																			                                        \
    MICRO_TILE_4x1                                                            \
																			                                        \
    DEC(R8)                                                                   \
    JNZ(.ZKLEFTLOOPZGEMM_4)                                                   \
																			                                        \
    LABEL(.ACCUMULATEZGEMM_4) /* Accumulating A*B over 4 registers */         \
    /* Shuffling the registers FMAed with imaginary components in B. */       \
    PERMUTE(6)                                                                \
																			                                        \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */                \
    ACC_COL(5, 6)                                                             \
																			                                        \
    /* A*B is accumulated over the ZMM registers as follows : */              \
    /* */                                                                     \
    /*  ZMM6  ZMM12  ZMM18  ZMM24 */                                          \
    /* */                                                                     \
																			                                        \
    /* Alpha scaling */                                                       \
    MOV(VAR(alpha), RAX)                                                      \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                         \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                       \
																			                                        \
    ALPHA_GENERIC(6)                                                          \
																			                                        \
    /* Beta scaling */                                                        \
    LABEL(.BETA_SCALEZGEMM_4)                                                 \
    /* Checking for storage scheme of C */                                    \
    CMP(IMM(16), RSI)                                                         \
    JE(.ROW_STORAGE_CZGEMM_4)  /* Jumping to row storage handling case */     \
																			                                        \
    /* Beta scaling when C is column stored */                                \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STOREZGEMM_4)                                                         \
																			                                        \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    /* Scaling C with beta, one column at a time */                           \
    BETA_GENERIC(RCX, 5, 6)                                                   \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STOREZGEMM_4)                                                      \
    VMOVUPD(ZMM(6), MEM(RCX))                                                 \
																			                                        \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Beta scaling when C is row stored */                                   \
    LABEL(.ROW_STORAGE_CZGEMM_4)                                              \
    /* */                                                                     \
    /*  In-register transposition happens over the 12x4 micro-tile */         \
    /*  in blocks of 4x4. */                                                  \
    /* */                                                                     \
    TRANSPOSE_4x4(6, 12, 18, 24)                                              \
    /* */                                                                     \
    /*  The layout post transposition and accumalation is as follows: */      \
    /*  ZMM6 */                                                               \
    /*  ZMM12 */                                                              \
    /*  ZMM18 */                                                              \
    /*  ZMM24 */                                                              \
    /* */                                                                     \
																			                                        \
    /* Loading C(row stored) and beta scaling */                              \
    MOV(RCX, R9)                                                              \
    MOV(VAR(beta_mul_type), AL)                                               \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                            \
    JE(.STORE_ROWZGEMM_4)                                                     \
    MOV(VAR(beta), RBX)                                                       \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                        \
																			                                        \
    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)                   \
    JMP(.CONCLUDE)                                                            \
																			                                        \
    /* Handling when beta == 0 */                                             \
    LABEL(.STORE_ROWZGEMM_4)                                                  \
    LEA(MEM(RCX, RDI, 2), R9)                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                     \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                            \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                            \
    VMOVUPD(ZMM(24), MEM(R9, RDI, 1) MASK_(k(3)))                             \
																			                                        \
    JMP(.CONCLUDE)



#define ZGEMM_4MASKx4                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						                                              \
    MICRO_TILE_4x4_MASK_SET1                                                              \
    MICRO_TILE_4x4_MASK_SET2                                                              \
    MICRO_TILE_4x4_MASK_SET1                                                              \
    MICRO_TILE_4x4_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                       \
																						                                              \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_1_TO_4)                                                            \
    VADDPD(ZMM(5), ZMM(7), ZMM(5))                                                        \
    VADDPD(ZMM(6), ZMM(8), ZMM(6))                                                        \
    VADDPD(ZMM(11), ZMM(13), ZMM(11))                                                     \
    VADDPD(ZMM(12), ZMM(14), ZMM(12))                                                     \
    VADDPD(ZMM(17), ZMM(19), ZMM(17))                                                     \
    VADDPD(ZMM(18), ZMM(20), ZMM(18))                                                     \
    VADDPD(ZMM(23), ZMM(25), ZMM(23))                                                     \
    VADDPD(ZMM(24), ZMM(26), ZMM(24))                                                     \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_1_TO_4)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_1_TO_4)                                                        \
																						                                              \
    MICRO_TILE_4x4_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_1_TO_4) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6)                                                                            \
    PERMUTE(12)                                                                           \
    PERMUTE(18)                                                                           \
    PERMUTE(24)                                                                           \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6)                                                                         \
    ACC_COL(11, 12)                                                                       \
    ACC_COL(17, 18)                                                                       \
    ACC_COL(23, 24)                                                                       \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_1_TO_4)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6)                                                                    \
    ALPHA_MINUS_ONE(12)                                                                   \
    ALPHA_MINUS_ONE(18)                                                                   \
    ALPHA_MINUS_ONE(24)                                                                   \
    JMP(.BETA_SCALE_EDGE_1_TO_4)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_1_TO_4)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_1_TO_4)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6)                                                                      \
    ALPHA_GENERIC(12)                                                                     \
    ALPHA_GENERIC(18)                                                                     \
    ALPHA_GENERIC(24)                                                                     \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_1_TO_4)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_1_TO_4)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_1_TO_4)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_1_TO_4)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_1_TO_4)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12)                                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18)                                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 23, 24)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_1_TO_4) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 23, 24)                                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_1_TO_4)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6)                                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12)                                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18)                                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 23, 24)                                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_1_TO_4)                                                             \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(2)))                                                 \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                        \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9) MASK_(k(2)))                                                 \
																						                                              \
    VMOVUPD(ZMM(24), MEM(R9, RSI, 1) MASK_(k(2)))                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_1_TO_4)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_1_TO_4)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3)                                                                          \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1)                                                                          \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4(RCX, 5, 6)                                                           \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 11, 12)                                                         \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4(RCX, 17, 18)                                                         \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE1)                                                                       \
    BETA_GEN_ROW_1x4(RCX, 5, 6)                                                           \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3R)                                                                         \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1R)                                                                         \
    LABEL(.UPDATE3R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))                                                    \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))                                                    \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX))                                                             \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_4MASKx3                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						                                              \
    MICRO_TILE_4x3_MASK_SET1                                                              \
    MICRO_TILE_4x3_MASK_SET2                                                              \
    MICRO_TILE_4x3_MASK_SET1                                                              \
    MICRO_TILE_4x3_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_1_TO_4)                                                            \
    VADDPD(ZMM(5), ZMM(7), ZMM(5))                                                        \
    VADDPD(ZMM(6), ZMM(8), ZMM(6))                                                        \
    VADDPD(ZMM(11), ZMM(13), ZMM(11))                                                     \
    VADDPD(ZMM(12), ZMM(14), ZMM(12))                                                     \
    VADDPD(ZMM(17), ZMM(19), ZMM(17))                                                     \
    VADDPD(ZMM(18), ZMM(20), ZMM(18))                                                     \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_1_TO_4)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_1_TO_4)                                                        \
																						                                              \
    MICRO_TILE_4x3_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_1_TO_4) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6)                                                                            \
    PERMUTE(12)                                                                           \
    PERMUTE(18)                                                                           \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6)                                                                         \
    ACC_COL(11, 12)                                                                       \
    ACC_COL(17, 18)                                                                       \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_1_TO_4)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6)                                                                    \
    ALPHA_MINUS_ONE(12)                                                                   \
    ALPHA_MINUS_ONE(18)                                                                   \
    JMP(.BETA_SCALE_EDGE_1_TO_4)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_1_TO_4)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_1_TO_4)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6)                                                                      \
    ALPHA_GENERIC(12)                                                                     \
    ALPHA_GENERIC(18)                                                                     \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_1_TO_4)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_1_TO_4)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_1_TO_4)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_1_TO_4)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_1_TO_4)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12)                                                      \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 17, 18)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_1_TO_4) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 17, 18)                                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_1_TO_4)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6)                                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12)                                                            \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 17, 18)                                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_1_TO_4)                                                             \
    LEA(MEM(RCX, RSI, 2), R9)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(2)))                                                 \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                        \
																						                                              \
    VMOVUPD(ZMM(18), MEM(R9) MASK_(k(2)))                                                 \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_1_TO_4)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_1_TO_4)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3)                                                                          \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1)                                                                          \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE1)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3R)                                                                         \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1R)                                                                         \
    LABEL(.UPDATE3R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                                        \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_4MASKx2                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						                                              \
    MICRO_TILE_4x2_MASK_SET1                                                              \
    MICRO_TILE_4x2_MASK_SET2                                                              \
    MICRO_TILE_4x2_MASK_SET1                                                              \
    MICRO_TILE_4x2_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_1_TO_4)                                                            \
    VADDPD(ZMM(5), ZMM(7), ZMM(5))                                                        \
    VADDPD(ZMM(6), ZMM(8), ZMM(6))                                                        \
    VADDPD(ZMM(11), ZMM(13), ZMM(11))                                                     \
    VADDPD(ZMM(12), ZMM(14), ZMM(12))                                                     \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_1_TO_4)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_1_TO_4)                                                        \
																						                                              \
    MICRO_TILE_4x2_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_1_TO_4) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6)                                                                            \
    PERMUTE(12)                                                                           \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6)                                                                         \
    ACC_COL(11, 12)                                                                       \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_1_TO_4)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6)                                                                    \
    ALPHA_MINUS_ONE(12)                                                                   \
    JMP(.BETA_SCALE_EDGE_1_TO_4)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_1_TO_4)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_1_TO_4)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6)                                                                      \
    ALPHA_GENERIC(12)                                                                     \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_1_TO_4)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_1_TO_4)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_1_TO_4)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_1_TO_4)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_1_TO_4)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6)                                                        \
    ADD(RSI, RCX)                                                                         \
    BETA_MINUS_ONE_MASK(RCX, 11, 12)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_1_TO_4) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6)                                                          \
    ADD(RSI, RCX)                                                                         \
    BETA_GENERIC_MASK(RCX, 11, 12)                                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_1_TO_4)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6)                                                              \
    ADD(RSI, RCX)                                                                         \
    BETA_ONE_MASK(RCX, 11, 12)                                                            \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_1_TO_4)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(2)))                                                 \
																						                                              \
    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1) MASK_(k(2)))                                        \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_1_TO_4)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_1_TO_4)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3)                                                                          \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1)                                                                          \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE1)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3R)                                                                         \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1R)                                                                         \
    LABEL(.UPDATE3R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                                        \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
																						                                              \
    JMP(.CONCLUDE)


#define ZGEMM_4MASKx1                                                                     \
    MOV(VAR(cs_a), R13)                                                                   \
    LEA(MEM(, R13, 8), R13)                                                               \
    LEA(MEM(, R13, 2), R13)                                                               \
																						                                              \
    MOV(VAR(rs_b), R14)                                                                   \
    LEA(MEM(, R14, 8), R14)                                                               \
    LEA(MEM(, R14, 2), R14)                                                               \
																						                                              \
    MOV(VAR(cs_b), R15)                                                                   \
    LEA(MEM(, R15, 8), R15)                                                               \
    LEA(MEM(, R15, 2), R15)                                                               \
																						                                              \
    MOV(VAR(rs_c), RDI)                                                                   \
    LEA(MEM(, RDI, 8), RDI)                                                               \
    LEA(MEM(, RDI, 2), RDI)                                                               \
																						                                              \
    MOV(VAR(cs_c), RSI)                                                                   \
    LEA(MEM(, RSI, 8), RSI)                                                               \
    LEA(MEM(, RSI, 2), RSI)                                                               \
																						                                              \
																						                                              \
    MOV(VAR(v), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(29))                                                        \
    RESET_REGISTERS                                                                       \
																						                                              \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														                                              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						                                              \
    MICRO_TILE_4x1_MASK_SET1                                                              \
    MICRO_TILE_4x1_MASK_SET2                                                              \
    MICRO_TILE_4x1_MASK_SET1                                                              \
    MICRO_TILE_4x1_MASK_SET2                                                              \
																						                                              \
    DEC(R8)             /* k_iter -= 1 */                                                 \
    JNZ(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                       \
																						                                              \
    /* Remainder loop for k */                                                            \
    LABEL(.ZKLEFT_EDGE_1_TO_4)                                                            \
    VADDPD(ZMM(5), ZMM(7), ZMM(5))                                                        \
    VADDPD(ZMM(6), ZMM(8), ZMM(6))                                                        \
                                                                                          \
    MOV(VAR(k_left), R8)                                                                  \
    TEST(R8, R8)                                                                          \
    JE(.ACCUMULATE_EDGE_1_TO_4)                                                           \
    LABEL(.ZKLEFTLOOP_EDGE_1_TO_4)                                                        \
																						                                              \
    MICRO_TILE_4x1_MASK_SET1                                                              \
																						                                              \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						                                              \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
		/**/																				                                          \
    /*  ZMM(6), ZMM(8), ... , ZMM(28) contain accumulations due to */                     \
    /*  imaginary components broadcasted from B. */                                       \
    /**/                                                                                  \
																						                                              \
    LABEL(.ACCUMULATE_EDGE_1_TO_4) /* Accumulating A*B over 12 registers */               \
    /* Shuffling the registers FMAed with imaginary components in B. */                   \
    PERMUTE(6)                                                                            \
																						                                              \
    /* Final accumulation for A*B on 12 reg using the 24 reg. */                          \
    ACC_COL(5, 6)                                                                         \
																						                                              \
    /* Alpha scaling */                                                                   \
    MOV(VAR(alpha_mul_type), AL)                                                          \
    CMP(IMM(0xFF), AL) /* Checking if alpha == -1 */                                      \
    JNE(.ALPHA_GENERAL_EDGE_1_TO_4)                                                       \
    /* Handling when alpha == -1 */                                                       \
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) /* Resetting ZMM(2) to 0 */                            \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    ALPHA_MINUS_ONE(6)                                                                    \
    JMP(.BETA_SCALE_EDGE_1_TO_4)                                                          \
																						                                              \
    LABEL(.ALPHA_GENERAL_EDGE_1_TO_4)                                                     \
    CMP(IMM(2), AL) /* Checking if alpha == BLIS_MUL_DEFAULT */                           \
    JNE(.BETA_SCALE_EDGE_1_TO_4)                                                          \
    MOV(VAR(alpha), RAX)                                                                  \
    VBROADCASTSD(MEM(RAX), ZMM(0))  /* Alpha->real */                                     \
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) /* Alpha->imag */                                   \
																						                                              \
    ALPHA_GENERIC(6)                                                                      \
																						                                              \
    /* Beta scaling */                                                                    \
    LABEL(.BETA_SCALE_EDGE_1_TO_4)                                                        \
    /* Checking for storage scheme of C */                                                \
    CMP(IMM(16), RSI)                                                                     \
    JE(.ROW_STORAGE_C_EDGE_1_TO_4)  /* Jumping to row storage handling case */            \
																						                                              \
    /* Beta scaling when C is column stored */                                            \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_EDGE_1_TO_4)                                                                \
    CMP(IMM(0x01), AL) /* Checking if beta == 1 */                                        \
    JE(.ADD_EDGE_1_TO_4)                                                                  \
    CMP(IMM(0xFF), AL) /* Checking if beta == -1 */                                       \
    JNE(.BETA_GENERAL_EDGE_1_TO_4)                                                        \
																						                                              \
    /* Subtracting C from alpha*A*B, one column at a time */                              \
    BETA_MINUS_ONE_MASK(RCX, 5, 6)                                                        \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.BETA_GENERAL_EDGE_1_TO_4) /* Checking if beta == BLIS_MUL_DEFAULT */           \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Scaling C with beta, one column at a time */                                       \
    BETA_GENERIC_MASK(RCX, 5, 6)                                                          \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 1 */                                                         \
    LABEL(.ADD_EDGE_1_TO_4)                                                               \
    /* Adding C to alpha*A*B, one column at a time */                                     \
    BETA_ONE_MASK(RCX, 5, 6)                                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_EDGE_1_TO_4)                                                             \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(2)))                                                 \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Beta scaling when C is row stored */                                               \
    LABEL(.ROW_STORAGE_C_EDGE_1_TO_4)                                                     \
    /**/                                                                                  \
    /*  In-register transposition happens over the 12x4 micro-tile*/                      \
    /*  in blocks of 4x4.*/                                                               \
    /**/                                                                                  \
    TRANSPOSE_4x4(6, 12, 18, 24)                                                          \
																						                                              \
    /* Loading C(row stored) and beta scaling */                                          \
    MOV(RCX, R9)                                                                          \
    MOV(VAR(m_left), R11)                                                                 \
    MOV(VAR(beta_mul_type), AL)                                                           \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                                        \
    JE(.STORE_ROW_EDGE_1_TO_4)                                                            \
    MOV(VAR(beta), RBX)                                                                   \
    VBROADCASTSD(MEM(RBX), ZMM(0))    /* Beta->real */                                    \
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) /* Beta->imag */                                    \
																						                                              \
    /* Handling when beta != 0 */                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3)                                                                          \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1)                                                                          \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    LABEL(.UPDATE1)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						                                              \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3R)                                                                         \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1R)                                                                         \
    LABEL(.UPDATE3R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                                        \
																						                                              \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1R)                                                                      \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
																						                                              \
    JMP(.CONCLUDE)

#define ZGEMM_2x4                                                           \
    MOV(VAR(cs_a), R13)                                                     \
    LEA(MEM(, R13, 8), R13)                                                 \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */             \
																			                                      \
    MOV(VAR(rs_b), R14)                                                     \
    LEA(MEM(, R14, 8), R14)                                                 \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */             \
																			                                      \
    MOV(VAR(cs_b), R15)                                                     \
    LEA(MEM(, R15, 8), R15)                                                 \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */             \
																			                                      \
    MOV(VAR(rs_c), RDI)                                                     \
    LEA(MEM(, RDI, 8), RDI)                                                 \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */             \
																			                                      \
    MOV(VAR(cs_c), RSI)                                                     \
    LEA(MEM(, RSI, 8), RSI)                                                 \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */             \
																			                                      \
    /* Intermediate register for complex arithmetic */                      \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                     \
    VBROADCASTSD(MEM(R9), YMM(2)) /* Broadcasting 1.0 over YMM(2) */        \
																			                                      \
																			                                      \
    /* Resetting all scratch registers */                                   \
    RESET_REGISTERS                                                         \
																			                                      \
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(11))                      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(12))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(1), ZMM(17))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(1), ZMM(18))                           \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(1), ZMM(19))                      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(1), ZMM(20))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(11))                      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(12))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(1), ZMM(17))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(1), ZMM(18))                           \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(1), ZMM(19))                      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(1), ZMM(20))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKITERMAINZGEMM_2)                                                 \
																			                                      \
    /* Remainder loop for k */                                              \
    LABEL(.ZKLEFTZGEMM_2)                                                   \
    VADDPD(ZMM(5), ZMM(13), ZMM(5))                                         \
    VADDPD(ZMM(6), ZMM(14), ZMM(6))                                         \
    VADDPD(ZMM(7), ZMM(15), ZMM(7))                                         \
    VADDPD(ZMM(8), ZMM(16), ZMM(8))                                         \
    VADDPD(ZMM(9), ZMM(17), ZMM(9))                                         \
    VADDPD(ZMM(10), ZMM(18), ZMM(10))                                       \
    VADDPD(ZMM(11), ZMM(19), ZMM(11))                                       \
    VADDPD(ZMM(12), ZMM(20), ZMM(12))                                       \
    MOV(VAR(k_left), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ACCUMULATEZGEMM_2)                                                  \
    LABEL(.ZKLEFTLOOPZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    VFMADD231PD(mem_1to8(R9, R15, 1), ZMM(0), ZMM(11))                      \
    VFMADD231PD(mem_1to8(R9, R15, 1, 8), ZMM(0), ZMM(12))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKLEFTLOOPZGEMM_2)                                                 \
																			                                      \
    LABEL(.ACCUMULATEZGEMM_2) /* Accumulating A*B over 4 registers */       \
    /* Shuffling the registers FMAed with imaginary components in B. */     \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
    VPERMILPD(IMM(0x5), YMM(10), YMM(10))                                   \
    VPERMILPD(IMM(0x5), YMM(12), YMM(12))                                   \
																			                                      \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */              \
    VADDSUBPD(YMM(6), YMM(5), YMM(6))                                       \
    VADDSUBPD(YMM(8), YMM(7), YMM(8))                                       \
    VADDSUBPD(YMM(10), YMM(9), YMM(10))                                     \
    VADDSUBPD(YMM(12), YMM(11), YMM(12))                                    \
																			                                      \
    /* A*B is accumulated over the YMM registers as follows : */            \
    /* */                                                                   \
    /*  YMM6  YMM8  YMM10  YMM12 */                                         \
    /* */                                                                   \
																			                                      \
    /* Alpha scaling */                                                     \
    MOV(VAR(alpha), RAX)                                                    \
    VBROADCASTSD(MEM(RAX), YMM(0))  /* Alpha->real */                       \
    VBROADCASTSD(MEM(RAX, 8), YMM(1)) /* Alpha->imag */                     \
																			                                      \
    VMULPD(YMM(0), YMM(6), YMM(15))                                         \
    VMULPD(YMM(1), YMM(6), YMM(6))                                          \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VADDSUBPD(YMM(6), YMM(15), YMM(6))                                      \
																			                                      \
    VMULPD(YMM(0), YMM(8), YMM(15))                                         \
    VMULPD(YMM(1), YMM(8), YMM(8))                                          \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
    VADDSUBPD(YMM(8), YMM(15), YMM(8))                                      \
																			                                      \
    VMULPD(YMM(0), YMM(10), YMM(15))                                        \
    VMULPD(YMM(1), YMM(10), YMM(10))                                        \
    VPERMILPD(IMM(0x5), YMM(10), YMM(10))                                   \
    VADDSUBPD(YMM(10), YMM(15), YMM(10))                                    \
																			                                      \
    VMULPD(YMM(0), YMM(12), YMM(15))                                        \
    VMULPD(YMM(1), YMM(12), YMM(12))                                        \
    VPERMILPD(IMM(0x5), YMM(12), YMM(12))                                   \
    VADDSUBPD(YMM(12), YMM(15), YMM(12))                                    \
																			                                      \
    /* Beta scaling */                                                      \
    LABEL(.BETA_SCALEZGEMM_2)                                               \
    /* Checking for storage scheme of C */                                  \
    CMP(IMM(16), RSI)                                                       \
    JE(.ROW_STORAGE_CZGEMM_2)  /* Jumping to row storage handling case */   \
																			                                      \
    /* Beta scaling when C is column stored */                              \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STOREZGEMM_2)                                                       \
																			                                      \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))  /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(5))                                               \
    VMULPD(YMM(0), YMM(5), YMM(15))                                         \
    VMULPD(YMM(1), YMM(5), YMM(5))                                          \
    VPERMILPD(IMM(0x5), YMM(5), YMM(5))                                     \
    VADDSUBPD(YMM(5), YMM(15), YMM(5))                                      \
    VADDPD(YMM(5), YMM(6), YMM(6))                                          \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(7))                                               \
    VMULPD(YMM(0), YMM(7), YMM(15))                                         \
    VMULPD(YMM(1), YMM(7), YMM(7))                                          \
    VPERMILPD(IMM(0x5), YMM(7), YMM(7))                                     \
    VADDSUBPD(YMM(7), YMM(15), YMM(7))                                      \
    VADDPD(YMM(7), YMM(8), YMM(8))                                          \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(9))                                               \
    VMULPD(YMM(0), YMM(9), YMM(15))                                         \
    VMULPD(YMM(1), YMM(9), YMM(9))                                          \
    VPERMILPD(IMM(0x5), YMM(9), YMM(9))                                     \
    VADDSUBPD(YMM(9), YMM(15), YMM(9))                                      \
    VADDPD(YMM(9), YMM(10), YMM(10))                                        \
    VMOVUPD(YMM(10), MEM(RCX))                                              \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(11))                                              \
    VMULPD(YMM(0), YMM(11), YMM(15))                                        \
    VMULPD(YMM(1), YMM(11), YMM(11))                                        \
    VPERMILPD(IMM(0x5), YMM(11), YMM(11))                                   \
    VADDSUBPD(YMM(11), YMM(15), YMM(11))                                    \
    VADDPD(YMM(11), YMM(12), YMM(12))                                       \
    VMOVUPD(YMM(12), MEM(RCX))                                              \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    LABEL(.STOREZGEMM_2)                                                    \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(10), MEM(RCX))                                              \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(12), MEM(RCX))                                              \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Beta scaling when C is row stored */                                 \
    LABEL(.ROW_STORAGE_CZGEMM_2)                                            \
    TRANSPOSE_2x2(6, 8)                                                     \
    TRANSPOSE_2x2(10, 12)                                                   \
																			                                      \
    /* Loading C(row stored) and beta scaling */                            \
    MOV(RCX, R9)                                                            \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STORE_ROWZGEMM_2)                                                   \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))    /* Beta->real */                      \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    BETA_GEN_ROW_2x4(R9, 5, 6, 9, 10)                                       \
    ADD(RDI, R9)                                                            \
    BETA_GEN_ROW_2x4(R9, 7, 8, 11, 12)                                      \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Handling when beta == 0 */                                           \
    LABEL(.STORE_ROWZGEMM_2)                                                \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    VMOVUPD(YMM(10), MEM(RCX, RSI, 2))                                      \
    ADD(RDI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    VMOVUPD(YMM(12), MEM(RCX, RSI, 2))                                      \
																			                                      \
    JMP(.CONCLUDE)

#define ZGEMM_2x3                                                           \
    MOV(VAR(cs_a), R13)                                                     \
    LEA(MEM(, R13, 8), R13)                                                 \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */             \
																			                                      \
    MOV(VAR(rs_b), R14)                                                     \
    LEA(MEM(, R14, 8), R14)                                                 \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */             \
																			                                      \
    MOV(VAR(cs_b), R15)                                                     \
    LEA(MEM(, R15, 8), R15)                                                 \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */             \
																			                                      \
    MOV(VAR(rs_c), RDI)                                                     \
    LEA(MEM(, RDI, 8), RDI)                                                 \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */             \
																			                                      \
    MOV(VAR(cs_c), RSI)                                                     \
    LEA(MEM(, RSI, 8), RSI)                                                 \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */             \
																			                                      \
    /* Intermediate register for complex arithmetic */                      \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                     \
    VBROADCASTSD(MEM(R9), YMM(2)) /* Broadcasting 1.0 over YMM(2) */        \
																			                                      \
																			                                      \
    /* Resetting all scratch registers */                                   \
    RESET_REGISTERS                                                         \
																			                                      \
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(1), ZMM(17))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(1), ZMM(18))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(1), ZMM(17))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(1), ZMM(18))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKITERMAINZGEMM_2)                                                 \
																			                                      \
    /* Remainder loop for k */                                              \
    LABEL(.ZKLEFTZGEMM_2)                                                   \
    VADDPD(ZMM(5), ZMM(13), ZMM(5))                                         \
    VADDPD(ZMM(6), ZMM(14), ZMM(6))                                         \
    VADDPD(ZMM(7), ZMM(15), ZMM(7))                                         \
    VADDPD(ZMM(8), ZMM(16), ZMM(8))                                         \
    VADDPD(ZMM(9), ZMM(17), ZMM(9))                                         \
    VADDPD(ZMM(10), ZMM(18), ZMM(10))                                       \
                                                                            \
    MOV(VAR(k_left), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ACCUMULATEZGEMM_2)                                                  \
    LABEL(.ZKLEFTLOOPZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    LEA(MEM(RBX, R15, 2), R9)                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD(mem_1to8(R9), ZMM(0), ZMM(9))                               \
    VFMADD231PD(mem_1to8(R9, 8), ZMM(0), ZMM(10))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKLEFTLOOPZGEMM_2)                                                 \
																			                                      \
    LABEL(.ACCUMULATEZGEMM_2) /* Accumulating A*B over 4 registers */       \
    /* Shuffling the registers FMAed with imaginary components in B. */     \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
    VPERMILPD(IMM(0x5), YMM(10), YMM(10))                                   \
																			                                      \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */              \
    VADDSUBPD(YMM(6), YMM(5), YMM(6))                                       \
    VADDSUBPD(YMM(8), YMM(7), YMM(8))                                       \
    VADDSUBPD(YMM(10), YMM(9), YMM(10))                                     \
																			                                      \
    /* A*B is accumulated over the YMM registers as follows : */            \
    /* */                                                                   \
    /*  YMM6  YMM8  YMM10  YMM12 */                                         \
    /* */                                                                   \
																			                                      \
    /* Alpha scaling */                                                     \
    MOV(VAR(alpha), RAX)                                                    \
    VBROADCASTSD(MEM(RAX), YMM(0))  /* Alpha->real */                       \
    VBROADCASTSD(MEM(RAX, 8), YMM(1)) /* Alpha->imag */                     \
																			                                      \
    VMULPD(YMM(0), YMM(6), YMM(15))                                         \
    VMULPD(YMM(1), YMM(6), YMM(6))                                          \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VADDSUBPD(YMM(6), YMM(15), YMM(6))                                      \
																			                                      \
    VMULPD(YMM(0), YMM(8), YMM(15))                                         \
    VMULPD(YMM(1), YMM(8), YMM(8))                                          \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
    VADDSUBPD(YMM(8), YMM(15), YMM(8))                                      \
																			                                      \
    VMULPD(YMM(0), YMM(10), YMM(15))                                        \
    VMULPD(YMM(1), YMM(10), YMM(10))                                        \
    VPERMILPD(IMM(0x5), YMM(10), YMM(10))                                   \
    VADDSUBPD(YMM(10), YMM(15), YMM(10))                                    \
																			                                      \
																			                                      \
    /* Beta scaling */                                                      \
    LABEL(.BETA_SCALEZGEMM_2)                                               \
    /* Checking for storage scheme of C */                                  \
    CMP(IMM(16), RSI)                                                       \
    JE(.ROW_STORAGE_CZGEMM_2)  /* Jumping to row storage handling case */   \
																			                                      \
    /* Beta scaling when C is column stored */                              \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STOREZGEMM_2)                                                       \
																			                                      \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))  /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(5))                                               \
    VMULPD(YMM(0), YMM(5), YMM(15))                                         \
    VMULPD(YMM(1), YMM(5), YMM(5))                                          \
    VPERMILPD(IMM(0x5), YMM(5), YMM(5))                                     \
    VADDSUBPD(YMM(5), YMM(15), YMM(5))                                      \
    VADDPD(YMM(5), YMM(6), YMM(6))                                          \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(7))                                               \
    VMULPD(YMM(0), YMM(7), YMM(15))                                         \
    VMULPD(YMM(1), YMM(7), YMM(7))                                          \
    VPERMILPD(IMM(0x5), YMM(7), YMM(7))                                     \
    VADDSUBPD(YMM(7), YMM(15), YMM(7))                                      \
    VADDPD(YMM(7), YMM(8), YMM(8))                                          \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(9))                                               \
    VMULPD(YMM(0), YMM(9), YMM(15))                                         \
    VMULPD(YMM(1), YMM(9), YMM(9))                                          \
    VPERMILPD(IMM(0x5), YMM(9), YMM(9))                                     \
    VADDSUBPD(YMM(9), YMM(15), YMM(9))                                      \
    VADDPD(YMM(9), YMM(10), YMM(10))                                        \
    VMOVUPD(YMM(10), MEM(RCX))                                              \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    LABEL(.STOREZGEMM_2)                                                    \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(10), MEM(RCX))                                              \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Beta scaling when C is row stored */                                 \
    LABEL(.ROW_STORAGE_CZGEMM_2)                                            \
    TRANSPOSE_2x2(6, 8)                                                     \
																			                                      \
    /* Loading C(row stored) and beta scaling */                            \
    MOV(RCX, R9)                                                            \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STORE_ROWZGEMM_2)                                                   \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))    /* Beta->real */                      \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    BETA_GEN_ROW_2x3(R9, 5, 6, 7, 8, 9, 10)                                 \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Handling when beta == 0 */                                           \
    LABEL(.STORE_ROWZGEMM_2)                                                \
    VEXTRACTF128(IMM(0x1), YMM(10), XMM(9))                                 \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    VMOVUPD(XMM(10), MEM(RCX, RSI, 2))                                      \
    ADD(RDI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    VMOVUPD(XMM(9), MEM(RCX, RSI, 2))                                       \
																			                                      \
    JMP(.CONCLUDE)


#define ZGEMM_2x2                                                           \
    MOV(VAR(cs_a), R13)                                                     \
    LEA(MEM(, R13, 8), R13)                                                 \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */             \
																			                                      \
    MOV(VAR(rs_b), R14)                                                     \
    LEA(MEM(, R14, 8), R14)                                                 \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */             \
																			                                      \
    MOV(VAR(cs_b), R15)                                                     \
    LEA(MEM(, R15, 8), R15)                                                 \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */             \
																			                                      \
    MOV(VAR(rs_c), RDI)                                                     \
    LEA(MEM(, RDI, 8), RDI)                                                 \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */             \
																			                                      \
    MOV(VAR(cs_c), RSI)                                                     \
    LEA(MEM(, RSI, 8), RSI)                                                 \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */             \
																			                                      \
    /* Intermediate register for complex arithmetic */                      \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                     \
    VBROADCASTSD(MEM(R9), YMM(2)) /* Broadcasting 1.0 over YMM(2) */        \
																			                                      \
																			                                      \
    /* Resetting all scratch registers */                                   \
    RESET_REGISTERS                                                         \
																			                                      \
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKITERMAINZGEMM_2)                                                 \
																			                                      \
    /* Remainder loop for k */                                              \
    LABEL(.ZKLEFTZGEMM_2)                                                   \
    VADDPD(ZMM(5), ZMM(13), ZMM(5))                                         \
    VADDPD(ZMM(6), ZMM(14), ZMM(6))                                         \
    VADDPD(ZMM(7), ZMM(15), ZMM(7))                                         \
    VADDPD(ZMM(8), ZMM(16), ZMM(8))                                         \
                                                                            \
    MOV(VAR(k_left), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ACCUMULATEZGEMM_2)                                                  \
    LABEL(.ZKLEFTLOOPZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKLEFTLOOPZGEMM_2)                                                 \
																			                                      \
    LABEL(.ACCUMULATEZGEMM_2) /* Accumulating A*B over 4 registers */       \
    /* Shuffling the registers FMAed with imaginary components in B. */     \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
																			                                      \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */              \
    VADDSUBPD(YMM(6), YMM(5), YMM(6))                                       \
    VADDSUBPD(YMM(8), YMM(7), YMM(8))                                       \
																			                                      \
    /* A*B is accumulated over the YMM registers as follows : */            \
    /* */                                                                   \
    /*  YMM6  YMM8  YMM10  YMM12 */                                         \
    /* */                                                                   \
																			                                      \
    /* Alpha scaling */                                                     \
    MOV(VAR(alpha), RAX)                                                    \
    VBROADCASTSD(MEM(RAX), YMM(0))  /* Alpha->real */                       \
    VBROADCASTSD(MEM(RAX, 8), YMM(1)) /* Alpha->imag */                     \
																			                                      \
    VMULPD(YMM(0), YMM(6), YMM(15))                                         \
    VMULPD(YMM(1), YMM(6), YMM(6))                                          \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VADDSUBPD(YMM(6), YMM(15), YMM(6))                                      \
																			                                      \
    VMULPD(YMM(0), YMM(8), YMM(15))                                         \
    VMULPD(YMM(1), YMM(8), YMM(8))                                          \
    VPERMILPD(IMM(0x5), YMM(8), YMM(8))                                     \
    VADDSUBPD(YMM(8), YMM(15), YMM(8))                                      \
																			                                      \
																			                                      \
    /* Beta scaling */                                                      \
    LABEL(.BETA_SCALEZGEMM_2)                                               \
    /* Checking for storage scheme of C */                                  \
    CMP(IMM(16), RSI)                                                       \
    JE(.ROW_STORAGE_CZGEMM_2)  /* Jumping to row storage handling case */   \
																			                                      \
    /* Beta scaling when C is column stored */                              \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STOREZGEMM_2)                                                       \
																			                                      \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))  /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(5))                                               \
    VMULPD(YMM(0), YMM(5), YMM(15))                                         \
    VMULPD(YMM(1), YMM(5), YMM(5))                                          \
    VPERMILPD(IMM(0x5), YMM(5), YMM(5))                                     \
    VADDSUBPD(YMM(5), YMM(15), YMM(5))                                      \
    VADDPD(YMM(5), YMM(6), YMM(6))                                          \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(7))                                               \
    VMULPD(YMM(0), YMM(7), YMM(15))                                         \
    VMULPD(YMM(1), YMM(7), YMM(7))                                          \
    VPERMILPD(IMM(0x5), YMM(7), YMM(7))                                     \
    VADDSUBPD(YMM(7), YMM(15), YMM(7))                                      \
    VADDPD(YMM(7), YMM(8), YMM(8))                                          \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
																			                                      \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    LABEL(.STOREZGEMM_2)                                                    \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RSI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Beta scaling when C is row stored */                                 \
    LABEL(.ROW_STORAGE_CZGEMM_2)                                            \
    TRANSPOSE_2x2(6, 8)                                                     \
																			                                      \
    /* Loading C(row stored) and beta scaling */                            \
    MOV(RCX, R9)                                                            \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STORE_ROWZGEMM_2)                                                   \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))    /* Beta->real */                      \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    BETA_GEN_ROW_2x2(R9, 5, 6, 7, 8)                                        \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Handling when beta == 0 */                                           \
    LABEL(.STORE_ROWZGEMM_2)                                                \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    ADD(RDI, RCX)                                                           \
    VMOVUPD(YMM(8), MEM(RCX))                                               \
																			                                      \
    JMP(.CONCLUDE)

#define ZGEMM_2x1                                                           \
    MOV(VAR(cs_a), R13)                                                     \
    LEA(MEM(, R13, 8), R13)                                                 \
    LEA(MEM(, R13, 2), R13)   /* R13 = sizeof(dcomplex)*cs_a */             \
																			                                      \
    MOV(VAR(rs_b), R14)                                                     \
    LEA(MEM(, R14, 8), R14)                                                 \
    LEA(MEM(, R14, 2), R14)   /* R14 = sizeof(dcomplex)*rs_b */             \
																			                                      \
    MOV(VAR(cs_b), R15)                                                     \
    LEA(MEM(, R15, 8), R15)                                                 \
    LEA(MEM(, R15, 2), R15)   /* R15 = sizeof(dcomplex)*cs_b */             \
																			                                      \
    MOV(VAR(rs_c), RDI)                                                     \
    LEA(MEM(, RDI, 8), RDI)                                                 \
    LEA(MEM(, RDI, 2), RDI)   /* RDI = sizeof(dcomplex)*rs_c */             \
																			                                      \
    MOV(VAR(cs_c), RSI)                                                     \
    LEA(MEM(, RSI, 8), RSI)                                                 \
    LEA(MEM(, RSI, 2), RSI)   /* RSI = sizeof(dcomplex)*cs_c */             \
																			                                      \
    /* Intermediate register for complex arithmetic */                      \
    MOV(VAR(v), R9)  /* Used in fmaddsub instruction */                     \
    VBROADCASTSD(MEM(R9), YMM(2)) /* Broadcasting 1.0 over YMM(2) */        \
																			                                      \
																			                                      \
    /* Resetting all scratch registers */                                   \
    RESET_REGISTERS                                                         \
																			                                      \
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKITERMAINZGEMM_2)                                                 \
																			                                      \
    /* Remainder loop for k */                                              \
    LABEL(.ZKLEFTZGEMM_2)                                                   \
    VADDPD(ZMM(5), ZMM(13), ZMM(5))                                         \
    VADDPD(ZMM(6), ZMM(14), ZMM(6))                                         \
                                                                            \
    MOV(VAR(k_left), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ACCUMULATEZGEMM_2)                                                  \
    LABEL(.ZKLEFTLOOPZGEMM_2)                                               \
																			                                      \
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                           \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			                                      \
    DEC(R8)                                                                 \
    JNZ(.ZKLEFTLOOPZGEMM_2)                                                 \
																			                                      \
    LABEL(.ACCUMULATEZGEMM_2) /* Accumulating A*B over 4 registers */       \
    /* Shuffling the registers FMAed with imaginary components in B. */     \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
																			                                      \
    /* Final accumulation for A*B on 4 reg using the 8 reg. */              \
    VADDSUBPD(YMM(6), YMM(5), YMM(6))                                       \
																			                                      \
    /* A*B is accumulated over the YMM registers as follows : */            \
    /* */                                                                   \
    /*  YMM6  YMM8  YMM10  YMM12 */                                         \
    /* */                                                                   \
																			                                      \
    /* Alpha scaling */                                                     \
    MOV(VAR(alpha), RAX)                                                    \
    VBROADCASTSD(MEM(RAX), YMM(0))  /* Alpha->real */                       \
    VBROADCASTSD(MEM(RAX, 8), YMM(1)) /* Alpha->imag */                     \
																			                                      \
    VMULPD(YMM(0), YMM(6), YMM(15))                                         \
    VMULPD(YMM(1), YMM(6), YMM(6))                                          \
    VPERMILPD(IMM(0x5), YMM(6), YMM(6))                                     \
    VADDSUBPD(YMM(6), YMM(15), YMM(6))                                      \
																			                                      \
																			                                      \
																			                                      \
    /* Beta scaling */                                                      \
    LABEL(.BETA_SCALEZGEMM_2)                                               \
    /* Checking for storage scheme of C */                                  \
    CMP(IMM(16), RSI)                                                       \
    JE(.ROW_STORAGE_CZGEMM_2)  /* Jumping to row storage handling case */   \
																			                                      \
    /* Beta scaling when C is column stored */                              \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STOREZGEMM_2)                                                       \
																			                                      \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))  /* Beta->real */                        \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    VMOVUPD(MEM(RCX), YMM(5))                                               \
    VMULPD(YMM(0), YMM(5), YMM(15))                                         \
    VMULPD(YMM(1), YMM(5), YMM(5))                                          \
    VPERMILPD(IMM(0x5), YMM(5), YMM(5))                                     \
    VADDSUBPD(YMM(5), YMM(15), YMM(5))                                      \
    VADDPD(YMM(5), YMM(6), YMM(6))                                          \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
																			                                      \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    LABEL(.STOREZGEMM_2)                                                    \
    VMOVUPD(YMM(6), MEM(RCX))                                               \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Beta scaling when C is row stored */                                 \
    LABEL(.ROW_STORAGE_CZGEMM_2)                                            \
    VEXTRACTF128(IMM(0x1), YMM(6), XMM(5))                                  \
																			                                      \
    /* Loading C(row stored) and beta scaling */                            \
    MOV(RCX, R9)                                                            \
    MOV(VAR(beta_mul_type), AL)                                             \
    CMP(IMM(0), AL)    /* Checking if beta == 0 */                          \
    JE(.STORE_ROWZGEMM_2)                                                   \
    MOV(VAR(beta), RBX)                                                     \
    VBROADCASTSD(MEM(RBX), YMM(0))    /* Beta->real */                      \
    VBROADCASTSD(MEM(RBX, 8), YMM(1)) /* Beta->imag */                      \
																			                                      \
    BETA_GEN_ROW_2x1(R9, 6, 5)                                              \
    JMP(.CONCLUDE)                                                          \
																			                                      \
    /* Handling when beta == 0 */                                           \
    LABEL(.STORE_ROWZGEMM_2)                                                \
    VMOVUPD(XMM(6), MEM(RCX))                                               \
    ADD(RDI, RCX)                                                           \
    VMOVUPD(XMM(5), MEM(RCX))																			          \
    JMP(.CONCLUDE)



void bli_zgemmsup_cv_zen4_asm_12x4m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
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

    // Obtaining the panel stride for A, In case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a16  = ps_a * sizeof( dcomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    uint8_t m_load_mask = ( (uint8_t)1 << ( 2 * (m_left % 4) ) ) - (uint8_t)1;


    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)// (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)// (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    uint64_t n_iter = (uint64_t)n0 / NR;
    dcomplex *bb = b;
    dcomplex *cc = c;
    dcomplex *a_ref = a;
    dim_t iter = 0;
    uint64_t m_iter_ref = (uint64_t)m0 / MR;

    for(; iter < n_iter; iter++)
    {
        b = bb + iter * NR *  cs_b0;
        c = cc + iter * NR *  cs_c0;
        a = a_ref;
        m_iter = m_iter_ref;

        BEGIN_ASM()
        MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
        MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
        MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

        MOV(VAR(cs_a), R13)
        LEA(MEM(, R13, 8), R13)
        LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*cs_a

        MOV(VAR(rs_b), R14)
        LEA(MEM(, R14, 8), R14)
        LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*rs_b

        MOV(VAR(cs_b), R15)
        LEA(MEM(, R15, 8), R15)
        LEA(MEM(, R15, 2), R15)   // R15 = sizeof(dcomplex)*cs_b

        MOV(VAR(rs_c), RDI)
        LEA(MEM(, RDI, 8), RDI)
        LEA(MEM(, RDI, 2), RDI)   // RDI = sizeof(dcomplex)*rs_c

        MOV(VAR(cs_c), RSI)
        LEA(MEM(, RSI, 8), RSI)
        LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

        // Intermediate register for complex arithmetic
        MOV(VAR(v), R9)  // Used in fmaddsub instruction
        VBROADCASTSD(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

        MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
        cmp(imm(0), r11)      /* check i via logical AND */
        je(.ZMLEFT)            /* jump to m_left case */
        LABEL(.ZMLOOP)
        MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
        MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
        MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

        // Resetting all scratch registers for arithmetic and accumulation
        RESET_REGISTERS

        // Setting iterator for k
        MOV(var(k_iter), R8)

        // Main loop for k
        /*
          The implementation facilitates C prefetching(in case of column-storage) onto
          L1 cache before accessing it. The k-loop is dissected into 3 segments, namely
          (B)efore (P)refetch, (D)uring (P)refetch and (A)fter (P)refetch. (D)uring (P)refetch
          segment prefetches C over 4 unrolled units of the 12x4 micro-tile computation in the k-loop.
          (A)fter (P)refetch segment runs over PREFETCH_DIST urolled units of k-loop.
        */
        SUB(IMM(4 + PREFETCH_DIST_C), R8)
        JLE(.ZK_DP)
        // Iterations of k(unroll factor = 4) before prefetching
        LABEL(.ZKITERLOOP_BP)     // K loop (B)efore (P)refetch of C

        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4

        DEC(R8)             // k_iter -= 1
        JNZ(.ZKITERLOOP_BP)

        LABEL(.ZK_DP)       // Prefetching over computation
        ADD(IMM(4), R8)     // Check if iterations available to prefetch over
        JLE(.ZK_AP)         // Jump without prefetching if not available
        MOV(RCX, R9)
        LABEL(.ZKITERLOOP_DP) // K loop (D)uring (P)refetch of C

        PREFETCH(1, MEM(R9))
        PREFETCH(1, MEM(R9, 64))
        PREFETCH(1, MEM(R9, 128))

        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4

        ADD(RSI, R9)

        DEC(R8)             // k_iter -= 1
        JNZ(.ZKITERLOOP_DP)

        LABEL(.ZK_AP)         // Computation after prefetching
        ADD(IMM(0 + PREFETCH_DIST_C), R8) // Check if enough iterations are available
        JLE(.ZKLEFT)          // Jump if not available
        LABEL(.ZKITERLOOP_AP) // K loop (A)fter (P)refetch of C

        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4
        MICRO_TILE_12x4

        DEC(R8)             // k_iter -= 1
        JNZ(.ZKITERLOOP_AP)

        // Remainder loop for k
        LABEL(.ZKLEFT)
        MOV(VAR(k_left), R8)
        TEST(R8, R8)
        JE(.ACCUMULATE)
        LABEL(.ZKLEFTLOOP)

        MICRO_TILE_12x4

        DEC(R8)             // k_left -= 1
        JNZ(.ZKLEFTLOOP)

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
          ZMM6  ZMM12  ZMM18  ZMM24
          ZMM8  ZMM14  ZMM20  ZMM26
          ZMM10 ZMM16  ZMM22  ZMM28
        */

        // Alpha scaling
        MOV(VAR(alpha_mul_type), AL)
        CMP(IMM(0xFF), AL) // Checking if alpha == -1
        JNE(.ALPHA_GENERAL)
        // Handling when alpha == -1
        VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

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
        VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
        VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

        ALPHA_GENERIC(6, 8, 10)
        ALPHA_GENERIC(12, 14, 16)
        ALPHA_GENERIC(18, 20, 22)
        ALPHA_GENERIC(24, 26, 28)

        // Beta scaling
        /*
          The final result of the GEMM operation is obtained in 2 steps:
          1. Loading C and beta scaling over loaded registers.
          2. Adding with registers containing alpha*A*B

          ZMM(5), ZMM(7), ... , ZMM(27) are used for implementing the first step.
          Final result of the GEMM operation is accumalated over ZMM(6), ZMM(8), ... , ZMM(28).
        */
        LABEL(.BETA_SCALE)
        // Checking for storage scheme of C
        CMP(IMM(16), RSI)
        JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

        // Beta scaling when C is column stored
        MOV(VAR(beta_mul_type), AL)
        CMP(IMM(0), AL)    // Checking if beta == 0
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
        VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
        VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

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
        VMOVUPD(ZMM(6), MEM(RCX))
        VMOVUPD(ZMM(8), MEM(RCX, 64))
        VMOVUPD(ZMM(10), MEM(RCX, 128))

        VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))
        VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))
        VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128))

        VMOVUPD(ZMM(18), MEM(R9))
        VMOVUPD(ZMM(20), MEM(R9, 64))
        VMOVUPD(ZMM(22), MEM(R9, 128))

        VMOVUPD(ZMM(24), MEM(R9, RSI, 1))
        VMOVUPD(ZMM(26), MEM(R9, RSI, 1, 64))
        VMOVUPD(ZMM(28), MEM(R9, RSI, 1, 128))
        JMP(.END)

        // Beta scaling when C is row stored
        LABEL(.ROW_STORAGE_C)
        /*
          In-register transposition happens over the 12x4 micro-tile
          in blocks of 4x4.
        */
        TRANSPOSE_4x4(6, 12, 18, 24)
        TRANSPOSE_4x4(8, 14, 20, 26)
        TRANSPOSE_4x4(10, 16, 22, 28)
        /*
          The layout post transposition and accumalation is as follows:
          ZMM6
          ZMM12
          ZMM18
          ZMM24

          ZMM8
          ZMM14
          ZMM20
          ZMM26

          ZMM10
          ZMM16
          ZMM22
          ZMM28
        */
        // Loading C(row stored) and beta scaling
        MOV(RCX, R9)
        MOV(VAR(beta_mul_type), AL)
        CMP(IMM(0), AL)    // Checking if beta == 0
        JE(.STORE_ROW)
        MOV(VAR(beta), RBX)
        VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
        VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

        // Handling when beta != 0
        BETA_GEN_ROW_4x4(R9, 5, 6, 11, 12, 17, 18, 23, 24)
        LEA(MEM(RCX, RDI, 2), RCX)
        LEA(MEM(R9, RDI, 2), R9)
        BETA_GEN_ROW_4x4(R9, 7, 8, 13, 14, 19, 20, 25, 26)
        LEA(MEM(RCX, RDI, 2), RCX)
        LEA(MEM(R9, RDI, 2), R9)
        BETA_GEN_ROW_4x4(R9, 9, 10, 15, 16, 21, 22, 27, 28)
        JMP(.END)

        // Handling when beta == 0
        LABEL(.STORE_ROW)
        LEA(MEM(RCX, RDI, 2), R9)
        LEA(MEM(R9, RDI, 1), R9)          // R9 = RCX + 3*rs_c
        VMOVUPD(ZMM(6), MEM(RCX))
        VMOVUPD(ZMM(12), MEM(RCX, RDI, 1))
        VMOVUPD(ZMM(18), MEM(RCX, RDI, 2))
        VMOVUPD(ZMM(8), MEM(RCX, RDI, 4))
        VMOVUPD(ZMM(10), MEM(RCX, RDI, 8))

        LEA(MEM(RCX, RDI, 4), RCX)
        LEA(MEM(RCX, RDI, 2), RCX)        // RCX = RCX + 6*rs_c
        VMOVUPD(ZMM(24), MEM(R9))
        VMOVUPD(ZMM(14), MEM(R9, RDI, 2))
        VMOVUPD(ZMM(26), MEM(R9, RDI, 4))
        VMOVUPD(ZMM(28), MEM(R9, RDI, 8))

        LEA(MEM(R9, RDI, 4), R9)
        LEA(MEM(R9, RDI, 2), R9)          // R9 = RCX + 9*rs_c
        VMOVUPD(ZMM(20), MEM(RCX))
        VMOVUPD(ZMM(22), MEM(RCX, RDI, 4))

        VMOVUPD(ZMM(16), MEM(R9))

        LABEL(.END)
        /*
          Adjusting the addresses for loading the
          next micro panel from A and the next micro
          tile from C.
        */
        MOV(VAR(ps_a16), RBX)
        ADD(RBX, R10)
        LEA(MEM(R12, RDI, 8), R12)
        LEA(MEM(R12, RDI, 4), R12)

        DEC(R11)
        JNE(.ZMLOOP)

        LABEL(.ZMLEFT)
        mov(var(m_left), r11)
        cmp(imm(0x0), r11)
        JZ(.CONCLUDE)


        MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
        MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
        MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

        // Check for m_left and based on m_left value we jump to repsective case.
        // m_left value handles as follows.
        // if m_left is 9, 10 or 11, it is computed by code block ZGEMM_12MASKx4 using masked load/store
        // if m_left is 8, it is computed by code block ZGEMM_8x4 using two zmm vector registers
        // if m_left is 5, 6 or 7, it is computed by code block ZGEMM_8MASKx4 using masked load/store
        // if m_left is 4, it is computed by code block ZGEMM_4x4 using 1 zmm vector register
        // if m_left is 1 or 3, it is computed by code block ZGEMM_4MASKx4 using masked load/store
        // if m_left 2, it is computed by code block ZGEMM_2x4 using ymm vector register.

        cmp(imm(0x8), r11)
        JZ(.EDGE8XN)
        JG(.EDGE12MASKXN)

        cmp(imm(0x4), r11)
        JZ(.EDGE4XN)
        JG(.EDGE8MASKXN)

        cmp(imm(0x2), r11)
        JZ(.EDGE2XN)

        cmp(imm(0x1), r11)
        jge(.EDGE4MASKXN)

        label(.EDGE12MASKXN)
        MOV(VAR(m_load_mask), EDI)
        KMOVW(EDI, k(2))             // k(2) = m_load_mask
        ZGEMM_12MASKx4               // Handles m_remainder case 9, 10, 11

        label(.EDGE8XN)
        ZGEMM_8x4                    // Handles m_remainder case 8

        label(.EDGE8MASKXN)
        MOV(VAR(m_load_mask), EDI)
        KMOVW(EDI, k(2))            // k(2) = m_load_mask
        ZGEMM_8MASKx4               // Handles m_remainder case 5, 6, 7

        label(.EDGE4XN)
        ZGEMM_4x4                   // Handles m_remainder case 4

        label(.EDGE2XN)
        ZGEMM_2x4                   // Handles m_remainder case 2

        label(.EDGE4MASKXN)
        MOV(VAR(m_load_mask), EDI)
        KMOVW(EDI, k(2))           // k(2) = m_load_mask
        ZGEMM_4MASKx4              // Handles m_remainder case 1, 3

        label(.CONCLUDE)

        END_ASM(
        : // output operands (none)
        : // input operands
          [v]  "m" (v),
          [m_iter]  "m" (m_iter),
          [m_load_mask] "m" (m_load_mask),
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
          [ps_a16]   "m" (ps_a16),
          [cs_a]   "m" (cs_a),
          [rs_b]   "m" (rs_b),
          [cs_b]   "m" (cs_b),
          [rs_c]   "m" (rs_c),
          [cs_c]   "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
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

    uint64_t n_left = n0 % NR;
    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 12x?m kernels, as needed.
    if ( n_left )
    {
      dcomplex*  cij = cc + iter * NR *  cs_c0;;
      dcomplex*  bj  = bb + iter * NR *  cs_b0;;
      dcomplex*  ai  = a_ref;

      if ( 3 == n_left )
      {
        const dim_t nr_cur = 3;
        bli_zgemmsup_cv_zen4_asm_12x3m(conja, conjb, m0, nr_cur, k0,
                                       alpha, ai, rs_a0, cs_a0,
                                       bj, rs_b0, cs_b0, beta,
                                       cij, rs_c0, cs_c0,
                                       data, cntx);
      }

      if ( 2 == n_left )
      {
        const dim_t nr_cur = 2;
        bli_zgemmsup_cv_zen4_asm_12x2m(conja, conjb, m0, nr_cur, k0,
                                       alpha, ai, rs_a0, cs_a0,
                                       bj, rs_b0, cs_b0, beta,
                                       cij, rs_c0, cs_c0,
                                       data, cntx);
      }
      if ( 1 == n_left )
      {
        const dim_t nr_cur = 1;
        bli_zgemmsup_cv_zen4_asm_12x1m(conja, conjb, m0, nr_cur, k0,
                                       alpha, ai, rs_a0, cs_a0,
                                       bj, rs_b0, cs_b0, beta,
                                       cij, rs_c0, cs_c0,
                                       data, cntx);
      }
      return;
    }

}

void bli_zgemmsup_cv_zen4_asm_12x3m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // This kernel is invoked at the beginning of 12x4m
    // In case of n_left == 3
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Obtaining the panel stride for A, In case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a16  = ps_a * sizeof( dcomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR;

    /*
      The mask bits below are set for ensuring ?x3 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 8 double precision elements.
    */
    uint8_t m_load_mask = ( (uint8_t)1 << ( 2 * (m_left % 4) ) ) - (uint8_t)1;
    uint8_t trans_load_mask = 0x3F; // Mask for transposing and loading = 0b 00 11 11 11
                                                                        //  0b 00  11 11 11 11 11
    /*
      This mask ensures that the ZMM registers disregard the last 16 bytes while
      using masked load/stores or FMA operations.
    */

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)// (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)// (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(ps_a16), R11)
    LEA(MEM(, R11, 8), R11)
    LEA(MEM(, R11, 2), R11)   // R11 = sizeof(dcomplex)*ps_a16

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15)
    LEA(MEM(, R15, 2), R15)   // R15 = sizeof(dcomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI)
    LEA(MEM(, RDI, 2), RDI)   // RDI = sizeof(dcomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    cmp(imm(0), r11)      /* check i via logical AND */
    je(.ZMLEFT)            /* jump to m_left case */
    LABEL(.ZMLOOP)
    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.ZKLEFT)
    // Main loop for k
    LABEL(.ZKITERMAIN)

    MICRO_TILE_12x3
    MICRO_TILE_12x3
    MICRO_TILE_12x3
    MICRO_TILE_12x3

    DEC(R8)
    JNZ(.ZKITERMAIN)

    // Remainder loop for k
    LABEL(.ZKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.ZKLEFTLOOP)

    MICRO_TILE_12x3

    DEC(R8)
    JNZ(.ZKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 9 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)
    PERMUTE(12, 14, 16)
    PERMUTE(18, 20, 22)

    // Final accumulation for A*B on 9 reg using the 24 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)
    ACC_COL(11, 12, 13, 14, 15, 16)
    ACC_COL(17, 18, 19, 20, 21, 22)

    // A*B is accumulated over the ZMM registers as follows :
    /*
      ZMM6  ZMM12  ZMM18
      ZMM8  ZMM14  ZMM20
      ZMM10 ZMM16  ZMM22
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    ALPHA_MINUS_ONE(12, 14, 16)
    ALPHA_MINUS_ONE(18, 20, 22)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)
    ALPHA_GENERIC(12, 14, 16)
    ALPHA_GENERIC(18, 20, 22)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(16), RSI)
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
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
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

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
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(8), MEM(RCX, 64))
    VMOVUPD(ZMM(10), MEM(RCX, 128))

    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))
    VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128))

    VMOVUPD(ZMM(18), MEM(RCX, RSI, 2))
    VMOVUPD(ZMM(20), MEM(RCX, RSI, 2, 64))
    VMOVUPD(ZMM(22), MEM(RCX, RSI, 2, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
      In-register transposition happens over the 12x4 micro-tile
      in blocks of 4x4.
    */
    TRANSPOSE_4x4(6, 12, 18, 24)
    TRANSPOSE_4x4(8, 14, 20, 26)
    TRANSPOSE_4x4(10, 16, 22, 28)
    /*
      The layout post transposition and accumalation is as follows:
      ZMM6
      ZMM12
      ZMM18
      ZMM24

      ZMM8
      ZMM14
      ZMM20
      ZMM26

      ZMM10
      ZMM16
      ZMM22
      ZMM28
    */

    // Loading C(row stored) and beta scaling
    MOV(RCX, R9)
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE_ROW)
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 9, 10, 15, 16, 21, 22, 27, 28)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    LEA(MEM(RCX, RDI, 2), R9)
    LEA(MEM(R9, RDI, 1), R9)
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3)))

    LEA(MEM(RCX, RDI, 4), RCX)
    LEA(MEM(RCX, RDI, 2), RCX)
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(28), MEM(R9, RDI, 8) MASK_(k(3)))

    LEA(MEM(R9, RDI, 4), R9)
    LEA(MEM(R9, RDI, 2), R9)
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))

    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a16), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 4), R12)

    DEC(R11)
    JNE(.ZMLOOP)

  LABEL(.ZMLEFT)
    mov(var(m_left), r11)
    cmp(imm(0x0), r11)
    JZ(.CONCLUDE)


    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Check for m_left and based on m_left value we jump to repsective case.
    // m_left value handles as follows.
    // if m_left is 9, 10 or 11, it is computed by code block ZGEMM_12MASKx3 using masked load/store
    // if m_left is 8, it is computed by code block ZGEMM_8x3 using two zmm vector registers
    // if m_left is 5, 6 or 7, it is computed by code block ZGEMM_8MASKx3 using masked load/store
    // if m_left is 4, it is computed by code block ZGEMM_4x3 using 1 zmm vector register
    // if m_left is 1 or 3, it is computed by code block ZGEMM_4MASKx3 using masked load/store
    // if m_left 2, it is computed by code block ZGEMM_2x3 using ymm vector register.

    cmp(imm(0x8), r11)
    JZ(.EDGE8XN)
    JG(.EDGE12MASKXN)

    cmp(imm(0x4), r11)
    JZ(.EDGE4XN)
    JG(.EDGE8MASKXN)

    cmp(imm(0x2), r11)
    JZ(.EDGE2XN)

    cmp(imm(0x1), r11)
    jge(.EDGE4MASKXN)

    label(.EDGE12MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))             // k(2) = m_load_mask
    ZGEMM_12MASKx3               // Handles m_remainder case 9, 10, 11

    label(.EDGE8XN)
    ZGEMM_8x3                    // Handles m_remainder case 8

    label(.EDGE8MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))            // k(2) = m_load_mask
    ZGEMM_8MASKx3               // Handles m_remainder case 5, 6, 7

    label(.EDGE4XN)
    ZGEMM_4x3                   // Handles m_remainder case 4

    label(.EDGE2XN)
    ZGEMM_2x3                   // Handles m_remainder case 2

    label(.EDGE4MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))           // k(2) = m_load_mask
    ZGEMM_4MASKx3              // Handles m_remainder case 1, 3

    label(.CONCLUDE)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_load_mask] "m" (m_load_mask),
      [m_left]  "m" (m_left),
      [k_iter]  "m" (k_iter),
      [k_left]  "m" (k_left),
      [trans_load_mask] "m" (trans_load_mask),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [alpha_mul_type]   "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [ps_a16]   "m" (ps_a16),
      [cs_a]   "m" (cs_a),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
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

void bli_zgemmsup_cv_zen4_asm_12x2m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // This kernel is invoked at the beginning of 12x4m
    // In case of n_left == 2
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Obtaining the panel stride for A, In case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a16  = ps_a * sizeof( dcomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR;

    /*
      The mask bits below are set for ensuring ?x2 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 8 double precision elements.
    */
    uint8_t m_load_mask = ( (uint8_t)1 << ( 2 * (m_left % 4) ) ) - (uint8_t)1;
    uint64_t trans_load_mask = 0x0F; // mask for transposing and loading = 0b 00 00 11 11
    /*
      This mask ensures that the ZMM registers disregard the last 32 bytes while
      using masked load/stores or FMA operations.
    */

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)// (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)// (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(ps_a16), R11)
    LEA(MEM(, R11, 8), R11)
    LEA(MEM(, R11, 2), R11)   // R11 = sizeof(dcomplex)*ps_a16

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15)
    LEA(MEM(, R15, 2), R15)   // R15 = sizeof(dcomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI)
    LEA(MEM(, RDI, 2), RDI)   // RDI = sizeof(dcomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    cmp(imm(0), r11)      /* check i via logical AND */
    je(.ZMLEFT)            /* jump to m_left case */
    LABEL(.ZMLOOP)
    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.ZKLEFT)
    // Main loop for k
    LABEL(.ZKITERMAIN)

    MICRO_TILE_12x2
    MICRO_TILE_12x2
    MICRO_TILE_12x2
    MICRO_TILE_12x2

    DEC(R8)
    JNZ(.ZKITERMAIN)

    // Remainder loop for k
    LABEL(.ZKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.ZKLEFTLOOP)

    MICRO_TILE_12x2

    DEC(R8)
    JNZ(.ZKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 6 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)
    PERMUTE(12, 14, 16)

    // Final accumulation for A*B on 6 reg using the 12 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)
    ACC_COL(11, 12, 13, 14, 15, 16)

    // A*B is accumulated over the ZMM registers as follows :
    /*
      ZMM6  ZMM12
      ZMM8  ZMM14
      ZMM10 ZMM16
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    ALPHA_MINUS_ONE(12, 14, 16)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)
    ALPHA_GENERIC(12, 14, 16)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(16), RSI)
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
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
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

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
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(8), MEM(RCX, 64))
    VMOVUPD(ZMM(10), MEM(RCX, 128))

    VMOVUPD(ZMM(12), MEM(RCX, RSI, 1))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1, 64))
    VMOVUPD(ZMM(16), MEM(RCX, RSI, 1, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
      In-register transposition happens over the 12x4 micro-tile
      in blocks of 4x4.
    */
    TRANSPOSE_4x4(6, 12, 18, 24)
    TRANSPOSE_4x4(8, 14, 20, 26)
    TRANSPOSE_4x4(10, 16, 22, 28)
    /*
      The layout post transposition and accumalation is as follows:
      ZMM6
      ZMM12
      ZMM18
      ZMM24

      ZMM8
      ZMM14
      ZMM20
      ZMM26

      ZMM10
      ZMM16
      ZMM22
      ZMM28
    */

    // Loading C(row stored) and beta scaling
    MOV(RCX, R9)
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE_ROW)
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 9, 10, 15, 16, 21, 22, 27, 28)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    LEA(MEM(RCX, RDI, 2), R9)
    LEA(MEM(R9, RDI, 1), R9)
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3)))

    LEA(MEM(RCX, RDI, 4), RCX)
    LEA(MEM(RCX, RDI, 2), RCX)
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(28), MEM(R9, RDI, 8) MASK_(k(3)))

    LEA(MEM(R9, RDI, 4), R9)
    LEA(MEM(R9, RDI, 2), R9)
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))

    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a16), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 4), R12)

    DEC(R11)
    JNE(.ZMLOOP)

    LABEL(.ZMLEFT)
    mov(var(m_left), r11)
    cmp(imm(0x0), r11)
    JZ(.CONCLUDE)


    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Check for m_left and based on m_left value we jump to repsective case.
    // m_left value handles as follows.
    // if m_left is 9, 10 or 11, it is computed by code block ZGEMM_12MASKx2 using masked load/store
    // if m_left is 8, it is computed by code block ZGEMM_8x2 using two zmm vector registers
    // if m_left is 5, 6 or 7, it is computed by code block ZGEMM_8MASKx2 using masked load/store
    // if m_left is 4, it is computed by code block ZGEMM_4x2 using 1 zmm vector register
    // if m_left is 1 or 3, it is computed by code block ZGEMM_4MASKx2 using masked load/store
    // if m_left 2, it is computed by code block ZGEMM_2x2 using ymm vector register.

    cmp(imm(0x8), r11)
    JZ(.EDGE8XN)
    JG(.EDGE12MASKXN)

    cmp(imm(0x4), r11)
    JZ(.EDGE4XN)
    JG(.EDGE8MASKXN)

    cmp(imm(0x2), r11)
    JZ(.EDGE2XN)

    cmp(imm(0x1), r11)
    jge(.EDGE4MASKXN)

    label(.EDGE12MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))             // k(2) = m_load_mask
    ZGEMM_12MASKx2               // Handles m_remainder case 9, 10, 11

    label(.EDGE8XN)
    ZGEMM_8x2                    // Handles m_remainder case 8

    label(.EDGE8MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))            // k(2) = m_load_mask
    ZGEMM_8MASKx2               // Handles m_remainder case 5, 6, 7

    label(.EDGE4XN)
    ZGEMM_4x2                   // Handles m_remainder case 4

    label(.EDGE2XN)
    ZGEMM_2x2                   // Handles m_remainder case 2

    label(.EDGE4MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))           // k(2) = m_load_mask
    ZGEMM_4MASKx2              // Handles m_remainder case 1, 3

    label(.CONCLUDE)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_load_mask] "m" (m_load_mask),
      [m_left]  "m" (m_left),
      [k_iter]  "m" (k_iter),
      [k_left]  "m" (k_left),
      [trans_load_mask] "m" (trans_load_mask),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [alpha_mul_type]   "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [ps_a16]   "m" (ps_a16),
      [cs_a]   "m" (cs_a),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
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

void bli_zgemmsup_cv_zen4_asm_12x1m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // This kernel is invoked at the beginning of 12x4m
    // In case of n_left == 1
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Obtaining the panel stride for A, In case of packing.
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a16  = ps_a * sizeof( dcomplex );

    uint64_t k_iter = k0 / 4; // Unroll factor of 4
    uint64_t k_left = k0 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    /*
      The mask bits below are set for ensuring ?x1 compatability
      while transposing, and loading/storing C in case of row-storage(k(3) opmask register).
      Mask is of length 8-bits, sinze a ZMM register holds 8 double precision elements.
    */
    uint8_t m_load_mask = ( (uint8_t)1 << ( 2 * (m_left % 4) ) ) - (uint8_t)1;
    uint64_t trans_load_mask = 0x03; // mask for transposing and loading = 0b 00 00 00 11
    /*
      This mask ensures that the ZMM registers disregard the last 48 bytes while
      using masked load/stores or FMA operations.
    */

    // Assigning the type of alpha and beta scaling
    // In order to facilitate handling special cases separately
    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    if(alpha->imag == 0.0)// (alpha is real)
    {
        if(alpha->real == 1.0)          alpha_mul_type = BLIS_MUL_ONE;
        else if(alpha->real == -1.0)    alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if(beta->imag == 0.0)// (beta is real)
    {
        if(beta->real == 1.0)       beta_mul_type = BLIS_MUL_ONE;
        else if(beta->real == -1.0) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if(beta->real == 0.0)  beta_mul_type = BLIS_MUL_ZERO;
    }

    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(ps_a16), R11)
    LEA(MEM(, R11, 8), R11)
    LEA(MEM(, R11, 2), R11)   // R11 = sizeof(dcomplex)*ps_a16

    MOV(VAR(cs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*cs_a

    MOV(VAR(rs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*rs_b

    MOV(VAR(cs_b), R15)
    LEA(MEM(, R15, 8), R15)
    LEA(MEM(, R15, 2), R15)   // R15 = sizeof(dcomplex)*cs_b

    MOV(VAR(rs_c), RDI)
    LEA(MEM(, RDI, 8), RDI)
    LEA(MEM(, RDI, 2), RDI)   // RDI = sizeof(dcomplex)*rs_c

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(VAR(trans_load_mask), EAX)
    KMOVW(EAX, k(3))               // k(3) = trans_load_mask

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R9)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R9), ZMM(29)) // Broadcasting 1.0 over ZMM(29)

    MOV(VAR(m_iter), R11) // Iterating in steps of MR, until MC(m var)
    cmp(imm(0), r11)      /* check i via logical AND */
    je(.ZMLEFT)            /* jump to m_left case */
    LABEL(.ZMLOOP)
    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(var(b), RBX)  // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Resetting all scratch registers
    RESET_REGISTERS

    // Setting iterator for k
    MOV(var(k_iter), R8)
    TEST(R8, R8)
    JE(.ZKLEFT)
    // Main loop for k
    LABEL(.ZKITERMAIN)

    MICRO_TILE_12x1
    MICRO_TILE_12x1
    MICRO_TILE_12x1
    MICRO_TILE_12x1

    DEC(R8)
    JNZ(.ZKITERMAIN)

    // Remainder loop for k
    LABEL(.ZKLEFT)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)
    LABEL(.ZKLEFTLOOP)

    MICRO_TILE_12x1

    DEC(R8)
    JNZ(.ZKLEFTLOOP)

    LABEL(.ACCUMULATE) // Accumulating A*B over 3 registers
    // Shuffling the registers FMAed with imaginary components in B.
    PERMUTE(6, 8, 10)

    // Final accumulation for A*B on 3 reg using the 6 reg.
    ACC_COL(5, 6, 7, 8, 9, 10)

    // A*B is accumulated over the ZMM registers as follows :
    /*
      ZMM6
      ZMM8
      ZMM10
    */

    // Alpha scaling
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0

    // Subtracting C from alpha*A*B, one column at a time
    ALPHA_MINUS_ONE(6, 8, 10)
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    ALPHA_GENERIC(6, 8, 10)

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Checking for storage scheme of C
    CMP(IMM(16), RSI)
    JE(.ROW_STORAGE_C)  // Jumping to row storage handling case

    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
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
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    // Scaling C with beta, one column at a time
    BETA_GENERIC(RCX, 5, 6, 7, 8, 9, 10)
    JMP(.END)

    // Handling when beta == 1
    LABEL(.ADD)
    // Adding C to alpha*A*B, one column at a time
    BETA_ONE(RCX, 5, 6, 7, 8, 9, 10)

    // Handling when beta == 0
    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(8), MEM(RCX, 64))
    VMOVUPD(ZMM(10), MEM(RCX, 128))
    JMP(.END)

    // Beta scaling when C is row stored
    LABEL(.ROW_STORAGE_C)
    /*
      In-register transposition happens over the 12x4 micro-tile
      in blocks of 4x4.
    */
    TRANSPOSE_4x4(6, 12, 18, 24)
    TRANSPOSE_4x4(8, 14, 20, 26)
    TRANSPOSE_4x4(10, 16, 22, 28)
    /*
      The layout post transposition and accumalation is as follows:
      ZMM6
      ZMM12
      ZMM18
      ZMM24

      ZMM8
      ZMM14
      ZMM20
      ZMM26

      ZMM10
      ZMM16
      ZMM22
      ZMM28
    */

    // Loading C(row stored) and beta scaling
    MOV(RCX, R9)
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE_ROW)
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    BETA_GEN_ROW_4x4_MASK(R9, 5, 6, 11, 12, 17, 18, 23, 24)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 7, 8, 13, 14, 19, 20, 25, 26)
    LEA(MEM(RCX, RDI, 2), RCX)
    LEA(MEM(R9, RDI, 2), R9)
    BETA_GEN_ROW_4x4_MASK(R9, 9, 10, 15, 16, 21, 22, 27, 28)
    JMP(.END)

    // Handling when beta == 0
    LABEL(.STORE_ROW)
    LEA(MEM(RCX, RDI, 2), R9)
    LEA(MEM(R9, RDI, 1), R9)
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(8), MEM(RCX, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(10), MEM(RCX, RDI, 8) MASK_(k(3)))

    LEA(MEM(RCX, RDI, 4), RCX)
    LEA(MEM(RCX, RDI, 2), RCX)
    VMOVUPD(ZMM(24), MEM(R9) MASK_(k(3)))
    VMOVUPD(ZMM(14), MEM(R9, RDI, 2) MASK_(k(3)))
    VMOVUPD(ZMM(26), MEM(R9, RDI, 4) MASK_(k(3)))
    VMOVUPD(ZMM(28), MEM(R9, RDI, 8) MASK_(k(3)))

    LEA(MEM(R9, RDI, 4), R9)
    LEA(MEM(R9, RDI, 2), R9)
    VMOVUPD(ZMM(20), MEM(RCX) MASK_(k(3)))
    VMOVUPD(ZMM(22), MEM(RCX, RDI, 4) MASK_(k(3)))

    VMOVUPD(ZMM(16), MEM(R9) MASK_(k(3)))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next micro panel from A and the next micro
      tile from C.
    */
    MOV(VAR(ps_a16), RBX)
    ADD(RBX, R10)
    LEA(MEM(R12, RDI, 8), R12)
    LEA(MEM(R12, RDI, 4), R12)

    DEC(R11)
    JNE(.ZMLOOP)

    LABEL(.ZMLEFT)
    mov(var(m_left), r11)
    cmp(imm(0x0), r11)
    JZ(.CONCLUDE)


    MOV(R10, RAX)     // RAX = addr of A for the MRxKC block
    MOV(RDX, RBX)     // RBX = addr of B for the KCxNR block
    MOV(R12, RCX)     // RCX = addr of C for the MRxNR block

    // Check for m_left and based on m_left value we jump to repsective case.
    // m_left value handles as follows.
    // if m_left is 9, 10 or 11, it is computed by code block ZGEMM_12MASKx1 using masked load/store
    // if m_left is 8, it is computed by code block ZGEMM_8x1 using two zmm vector registers
    // if m_left is 5, 6 or 7, it is computed by code block ZGEMM_8MASKx1 using masked load/store
    // if m_left is 4, it is computed by code block ZGEMM_4x1 using 1 zmm vector register
    // if m_left is 1 or 3, it is computed by code block ZGEMM_4MASKx1 using masked load/store
    // if m_left 2, it is computed by code block ZGEMM_2x1 using ymm vector register.

    cmp(imm(0x8), r11)
    JZ(.EDGE8XN)
    JG(.EDGE12MASKXN)

    cmp(imm(0x4), r11)
    JZ(.EDGE4XN)
    JG(.EDGE8MASKXN)

    cmp(imm(0x2), r11)
    JZ(.EDGE2XN)

    cmp(imm(0x1), r11)
    jge(.EDGE4MASKXN)

    label(.EDGE12MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))             // k(2) = m_load_mask
    ZGEMM_12MASKx1               // Handles m_remainder case 9, 10, 11

    label(.EDGE8XN)
    ZGEMM_8x1                    // Handles m_remainder case 8

    label(.EDGE8MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))            // k(2) = m_load_mask
    ZGEMM_8MASKx1               // Handles m_remainder case 5, 6, 7

    label(.EDGE4XN)
    ZGEMM_4x1                   // Handles m_remainder case 4

    label(.EDGE2XN)
    ZGEMM_2x1                   // Handles m_remainder case 2

    label(.EDGE4MASKXN)
    MOV(VAR(m_load_mask), EDI)
    KMOVW(EDI, k(2))           // k(2) = m_load_mask
    ZGEMM_4MASKx1              // Handles m_remainder case 1, 3

    label(.CONCLUDE)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_load_mask] "m" (m_load_mask),
      [m_left]  "m" (m_left),
      [k_iter]  "m" (k_iter),
      [k_left]  "m" (k_left),
      [trans_load_mask] "m" (trans_load_mask),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [alpha_mul_type]   "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [ps_a16]   "m" (ps_a16),
      [cs_a]   "m" (cs_a),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "al",
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
