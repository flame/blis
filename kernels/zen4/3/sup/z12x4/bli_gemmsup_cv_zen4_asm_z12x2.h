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


#define MICRO_TILE_12x2_MASK_CONJA                  \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    VMULPD(ZMM(30), ZMM(0), ZMM(0))                 \
    VMULPD(ZMM(30), ZMM(1), ZMM(1))                 \
    VMULPD(ZMM(30), ZMM(2), ZMM(2))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))      \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                             \
    FMA(4, 12, 14, 16)                             \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_12x2_MASK_CONJB                  \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                              \
    FMA(4, 12, 14, 16)                              \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_12x2_MASK_CONJA_CONJB            \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2) MASK_KZ(2))       \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                 \
    VMULPD(ZMM(1), ZMM(30), ZMM(1))                 \
    VMULPD(ZMM(2), ZMM(30), ZMM(2))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                              \
    FMA(4, 12, 14, 16)                              \
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


#define MICRO_TILE_8x2_MASK_SET1_CONJA              \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    VMULPD(ZMM(27), ZMM(0), ZMM(0))                 \
    VMULPD(ZMM(27), ZMM(1), ZMM(1))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_8x2_MASK_SET2_CONJA              \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(23))                 \
    VBROADCASTSD(MEM(RBX, 8), ZMM(24))              \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(3))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(4) MASK_KZ(2))        \
    VMULPD(ZMM(27), ZMM(3), ZMM(3))                 \
    VMULPD(ZMM(27), ZMM(4), ZMM(4))                 \
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


#define MICRO_TILE_8x2_MASK_SET1_CONJB              \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(27), ZMM(4), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    VMULPD(ZMM(27), ZMM(31), ZMM(31))               \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(30, 11, 13)                                 \
    FMA(31, 12, 14)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_8x2_MASK_SET2_CONJB              \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(23))                 \
    VBROADCASTSD(MEM(RBX, 8), ZMM(24))              \
    VMULPD(ZMM(27), ZMM(24), ZMM(24))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(3))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(4) MASK_KZ(2))        \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(30))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(31))      \
    VMULPD(ZMM(27), ZMM(31), ZMM(31))               \
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


#define MICRO_TILE_8x2_MASK_SET1_CONJA_CONJB        \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1) MASK_KZ(2))        \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                 \
    VMULPD(ZMM(1), ZMM(30), ZMM(1))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* 4 FMAs over 2 broadcasts */                  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1,  8), ZMM(4))      \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    FMA(3, 11, 13)                                  \
    FMA(4, 12, 14)                                  \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_8x2_MASK_SET2_CONJA_CONJB        \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(23))                 \
    VBROADCASTSD(MEM(RBX, 8), ZMM(24))              \
    VMULPD(ZMM(24), ZMM(31), ZMM(24))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(3))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(4) MASK_KZ(2))        \
    VMULPD(ZMM(3), ZMM(30), ZMM(3))                 \
    VMULPD(ZMM(4), ZMM(30), ZMM(4))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    VFMADD231PD(ZMM(3), ZMM(23), ZMM(15))           \
    VFMADD231PD(ZMM(4), ZMM(23), ZMM(17))           \
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(16))           \
    VFMADD231PD(ZMM(4), ZMM(24), ZMM(18))           \
    /* 4 FMAs over 2 broadcasts */                  \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(23))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(24))      \
    VMULPD(ZMM(24), ZMM(31), ZMM(24))               \
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


#define MICRO_TILE_4x2_MASK_SET1_CONJA                      \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VMULPD(ZMM(30), ZMM(0), ZMM(0))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_4x2_MASK_SET2_CONJA                      \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    VMULPD(ZMM(30), ZMM(1), ZMM(1))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1), ZMM(8))           \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(14))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


#define MICRO_TILE_4x2_MASK_SET1_CONJB                      \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(6))                     \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(12))                    \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_4x2_MASK_SET2_CONJB                      \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(8))                     \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(14))                    \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


#define MICRO_TILE_4x2_MASK_SET1_CONJA_CONJB                \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(2))                    \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(6))                     \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(12))                    \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

#define MICRO_TILE_4x2_MASK_SET2_CONJA_CONJB                \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(1) MASK_KZ(2))                    \
    VMULPD(ZMM(1), ZMM(30), ZMM(1))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(1), ZMM(7))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(8))                     \
    /* 2 FMAs over 2 broadcasts */                          \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(13))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(14))                    \
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


#define MICRO_TILE_12x2_CONJA                       \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    VMULPD(ZMM(30), ZMM(0), ZMM(0))                 \
    VMULPD(ZMM(30), ZMM(1), ZMM(1))                 \
    VMULPD(ZMM(30), ZMM(2), ZMM(2))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                              \
    FMA(4, 12, 14, 16)                              \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_12x2_CONJB                       \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                              \
    FMA(4, 12, 14, 16)                              \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \


#define MICRO_TILE_12x2_CONJA_CONJB                 \
    /* Macro for 12x2 micro-tile evaluation   */    \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(2) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMOVUPD(MEM(RAX, 128), ZMM(2))                  \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                 \
    VMULPD(ZMM(1), ZMM(30), ZMM(1))                 \
    VMULPD(ZMM(2), ZMM(30), ZMM(2))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7, 9)                                 \
    FMA(4, 6, 8, 10)                                \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* 6 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13, 15)                              \
    FMA(4, 12, 14, 16)                              \
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

#define MICRO_TILE_8x2_CONJA                        \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMULPD(ZMM(30), ZMM(0), ZMM(0))                 \
    VMULPD(ZMM(30), ZMM(1), ZMM(1))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))      \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13)                                 \
    FMA(4, 12, 14)                                 \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x2_CONJB                        \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))          \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))       \
    VMULPD(ZMM(30), ZMM(4), ZMM(4))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13)                                  \
    FMA(4, 12, 14)                                  \
    /* Adjusting addresses for next micro tiles */  \
    ADD(R14, RBX)                                   \
    ADD(R13, RAX)                                   \

#define MICRO_TILE_8x2_CONJA_CONJB                  \
    /* Macro for 8x2 micro-tile evaluation   */     \
    /* Prebroadcasting B on ZMM(3) and ZMM(4) */    \
    VBROADCASTSD(MEM(RBX), ZMM(3))                  \
    VBROADCASTSD(MEM(RBX, 8), ZMM(4))               \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* Loading A using ZMM(0) - ZMM(1) */           \
    VMOVUPD(MEM(RAX), ZMM(0))                       \
    VMOVUPD(MEM(RAX, 64), ZMM(1))                   \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                 \
    VMULPD(ZMM(1), ZMM(30), ZMM(1))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 5, 7)                                    \
    FMA(4, 6, 8)                                    \
    /* Prebroadcasting B on ZMM(30) and ZMM(31) */  \
    VBROADCASTSD(MEM(RBX, R15, 1), ZMM(3))         \
    VBROADCASTSD(MEM(RBX, R15, 1, 8), ZMM(4))      \
    VMULPD(ZMM(4), ZMM(31), ZMM(4))                 \
    /* 4 FMAs over 2 broadcasts */                  \
    FMA(3, 11, 13)                                 \
    FMA(4, 12, 14)                                 \
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


#define MICRO_TILE_4x2_CONJA                                \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    VMULPD(ZMM(30), ZMM(0), ZMM(0))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0), ZMM(6))           \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(12))  \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


#define MICRO_TILE_4x2_CONJB                                \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(6))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(12))                    \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \


#define MICRO_TILE_4x2_CONJA_CONJB                          \
    /* Macro for 4x2 micro-tile evaluation   */             \
    /* Loading A using ZMM(0) */                            \
    VMOVUPD(MEM(RAX), ZMM(0))                               \
    VMULPD(ZMM(0), ZMM(30), ZMM(0))                         \
    VFMADD231PD(mem_1to8(RBX), ZMM(0), ZMM(5))              \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4))               \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(6))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(11))     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4))       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(12))                    \
    /* Adjusting addresses for next micro tiles */          \
    ADD(R14, RBX)                                           \
    ADD(R13, RAX)                                           \

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
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
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
	TRANSPOSE_4x4(10, 16, 22, 28)								                          \
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

#define ZGEMM_12MASKx2_CONJA                                                              \
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
    MOV(VAR(conja_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(30))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
																						  \
																						  \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						  \
    MICRO_TILE_12x2_MASK_CONJA                                                            \
    MICRO_TILE_12x2_MASK_CONJA                                                            \
    MICRO_TILE_12x2_MASK_CONJA                                                            \
    MICRO_TILE_12x2_MASK_CONJA                                                            \
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
    MICRO_TILE_12x2_MASK_CONJA                                                            \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
	TRANSPOSE_4x4(10, 16, 22, 28)								                          \
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


#define ZGEMM_12MASKx2_CONJB                                                              \
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
    MOV(VAR(conjb_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(30))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
																						  \
																						  \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						  \
    MICRO_TILE_12x2_MASK_CONJB                                                            \
    MICRO_TILE_12x2_MASK_CONJB                                                            \
    MICRO_TILE_12x2_MASK_CONJB                                                            \
    MICRO_TILE_12x2_MASK_CONJB                                                            \
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
    MICRO_TILE_12x2_MASK_CONJB                                                            \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
	TRANSPOSE_4x4(10, 16, 22, 28)								                          \
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


#define ZGEMM_12MASKx2_CONJA_CONJB                                                        \
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
    MOV(VAR(conja_array), R9)                                                             \
    VBROADCASTSD(MEM(R9), ZMM(30))                                                        \
    MOV(VAR(conjb_array), R9)                                                             \
    VBROADCASTSD(MEM(R9), ZMM(31))                                                        \
																						  \
    MOV(var(k_iter), R8)                                                                  \
																						  \
																						  \
    TEST(R8, R8)                                                                          \
    JE(.ZKLEFT_EDGE_8_TO_12)                                                              \
    LABEL(.ZKITERLOOP_BP_EDGE_8_TO_12)                                                    \
																						  \
    MICRO_TILE_12x2_MASK_CONJA_CONJB                                                      \
    MICRO_TILE_12x2_MASK_CONJA_CONJB                                                      \
    MICRO_TILE_12x2_MASK_CONJA_CONJB                                                      \
    MICRO_TILE_12x2_MASK_CONJA_CONJB                                                      \
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
    MICRO_TILE_12x2_MASK_CONJA_CONJB                                                      \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_8_TO_12)                                                         \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
	TRANSPOSE_4x4(10, 16, 22, 28)								                          \
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


#define ZGEMM_8x2_CONJA                                                       \
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
    MOV(VAR(conja_array), R9)                                                 \
    VMOVUPD(MEM(R9), ZMM(30))                                                 \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x2_CONJA                                                      \
    MICRO_TILE_8x2_CONJA                                                      \
    MICRO_TILE_8x2_CONJA                                                      \
    MICRO_TILE_8x2_CONJA                                                      \
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
    MICRO_TILE_8x2_CONJA                                                      \
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


#define ZGEMM_8x2_CONJB                                                       \
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
    MOV(VAR(conjb_array), R9)                                                 \
    VMOVUPD(MEM(R9), ZMM(30))                                                 \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x2_CONJB                                                      \
    MICRO_TILE_8x2_CONJB                                                      \
    MICRO_TILE_8x2_CONJB                                                      \
    MICRO_TILE_8x2_CONJB                                                      \
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
    MICRO_TILE_8x2_CONJB                                                      \
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


#define ZGEMM_8x2_CONJA_CONJB                                                 \
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
    MOV(VAR(conja_array), R9)                                                 \
    VBROADCASTSD(MEM(R9), ZMM(30))                                            \
    MOV(VAR(conjb_array), R9)                                                 \
    VBROADCASTSD(MEM(R9), ZMM(31))                                            \
                                                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTEDGE8)                                                          \
    LABEL(.ZKITERMAINEDGE8)                                                   \
                                                                              \
    MICRO_TILE_8x2_CONJA_CONJB                                                \
    MICRO_TILE_8x2_CONJA_CONJB                                                \
    MICRO_TILE_8x2_CONJA_CONJB                                                \
    MICRO_TILE_8x2_CONJA_CONJB                                                \
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
    MICRO_TILE_8x2_CONJA_CONJB                                                \
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
	TEST(R8, R8)  													                      \
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


#define ZGEMM_8MASKx2_CONJA                                                               \
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
    MOV(VAR(conja_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(27))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
	TEST(R8, R8)  													                      \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						  \
    MICRO_TILE_8x2_MASK_SET1_CONJA                                                        \
    MICRO_TILE_8x2_MASK_SET2_CONJA                                                        \
    MICRO_TILE_8x2_MASK_SET1_CONJA                                                        \
    MICRO_TILE_8x2_MASK_SET2_CONJA                                                        \
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
    MICRO_TILE_8x2_MASK_SET1_CONJA                                                        \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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


#define ZGEMM_8MASKx2_CONJB                                                               \
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
    MOV(VAR(conjb_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(27))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
	TEST(R8, R8)  													                      \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						  \
    MICRO_TILE_8x2_MASK_SET1_CONJB                                                        \
    MICRO_TILE_8x2_MASK_SET2_CONJB                                                        \
    MICRO_TILE_8x2_MASK_SET1_CONJB                                                        \
    MICRO_TILE_8x2_MASK_SET2_CONJB                                                        \
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
    MICRO_TILE_8x2_MASK_SET1_CONJB                                                        \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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


#define ZGEMM_8MASKx2_CONJA_CONJB                                                         \
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
    MOV(VAR(conja_array), R9)                                                             \
    VBROADCASTSD(MEM(R9), ZMM(30))                                                        \
    MOV(VAR(conjb_array), R9)                                                             \
    VBROADCASTSD(MEM(R9), ZMM(31))                                                        \
																						  \
    MOV(var(k_iter), R8)                                                                  \
	TEST(R8, R8)  													                      \
    JE(.ZKLEFT_EDGE_4_TO_8)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_4_TO_8)                                                     \
																						  \
    MICRO_TILE_8x2_MASK_SET1_CONJA_CONJB                                                  \
    MICRO_TILE_8x2_MASK_SET2_CONJA_CONJB                                                  \
    MICRO_TILE_8x2_MASK_SET1_CONJA_CONJB                                                  \
    MICRO_TILE_8x2_MASK_SET2_CONJA_CONJB                                                  \
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
    MICRO_TILE_8x2_MASK_SET1_CONJA_CONJB                                                  \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_4_TO_8)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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


#define ZGEMM_4x2_CONJA                                                       \
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
    MOV(VAR(conja_array), R9)                                                 \
    VMOVUPD(MEM(R9), ZMM(30))                                                 \
																			  \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			  \
    MICRO_TILE_4x2_CONJA                                                      \
    MICRO_TILE_4x2_CONJA                                                      \
    MICRO_TILE_4x2_CONJA                                                      \
    MICRO_TILE_4x2_CONJA                                                      \
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
    MICRO_TILE_4x2_CONJA                                                      \
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


#define ZGEMM_4x2_CONJB                                                       \
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
    MOV(VAR(conjb_array), R9)                                                 \
    VMOVUPD(MEM(R9), ZMM(30))                                                 \
																			  \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			  \
    MICRO_TILE_4x2_CONJB                                                      \
    MICRO_TILE_4x2_CONJB                                                      \
    MICRO_TILE_4x2_CONJB                                                      \
    MICRO_TILE_4x2_CONJB                                                      \
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
    MICRO_TILE_4x2_CONJB                                                      \
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


#define ZGEMM_4x2_CONJA_CONJB                                                 \
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
    MOV(VAR(conja_array), R9)  /* Used in fmaddsub instruction */             \
    VBROADCASTSD(MEM(R9), ZMM(30)) /* Broadcasting 1.0 over ZMM(29) */        \
    MOV(VAR(conjb_array), R9)  /* Used in fmaddsub instruction */             \
    VBROADCASTSD(MEM(R9), ZMM(31)) /* Broadcasting 1.0 over ZMM(29) */        \
																			  \
    /* Setting iterator for k */                                              \
    MOV(VAR(k_iter), R8)                                                      \
    TEST(R8, R8)                                                              \
    JE(.ZKLEFTZGEMM_4)                                                        \
    LABEL(.ZKITERMAINZGEMM_4)                                                 \
																			  \
    MICRO_TILE_4x2_CONJA_CONJB                                                \
    MICRO_TILE_4x2_CONJA_CONJB                                                \
    MICRO_TILE_4x2_CONJA_CONJB                                                \
    MICRO_TILE_4x2_CONJA_CONJB                                                \
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
    MICRO_TILE_4x2_CONJA_CONJB                                                \
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
		TEST(R8, R8)														              \
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
    JZ(.UPDATE1NN)                                                                        \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						  \
    LABEL(.UPDATE1NN)                                                                     \
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

#define ZGEMM_4MASKx2_CONJA                                                               \
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
    MOV(VAR(conja_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(30))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
    TEST(R8, R8)														                  \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						  \
    MICRO_TILE_4x2_MASK_SET1_CONJA                                                        \
    MICRO_TILE_4x2_MASK_SET2_CONJA                                                        \
    MICRO_TILE_4x2_MASK_SET1_CONJA                                                        \
    MICRO_TILE_4x2_MASK_SET2_CONJA                                                        \
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
    MICRO_TILE_4x2_MASK_SET1_CONJA                                                        \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
    JZ(.UPDATE3CONJA)                                                                     \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1CONJA)                                                                     \
    LABEL(.UPDATE3CONJA)                                                                  \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						  \
    LABEL(.UPDATE1CONJA)                                                                  \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						  \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3RCONJA)                                                                    \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1RCONJA)                                                                    \
    LABEL(.UPDATE3RCONJA)                                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                                        \
																						  \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1RCONJA)                                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
																						  \
    JMP(.CONCLUDE)


#define ZGEMM_4MASKx2_CONJB                                                               \
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
    MOV(VAR(conjb_array), R9)                                                             \
    VMOVUPD(MEM(R9), ZMM(30))                                                             \
																						  \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						  \
    MICRO_TILE_4x2_MASK_SET1_CONJB                                                        \
    MICRO_TILE_4x2_MASK_SET2_CONJB                                                        \
    MICRO_TILE_4x2_MASK_SET1_CONJB                                                        \
    MICRO_TILE_4x2_MASK_SET2_CONJB                                                        \
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
    MICRO_TILE_4x2_MASK_SET1_CONJB                                                        \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
    JZ(.UPDATE3CONJB)                                                                     \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1CONJB)                                                                     \
    LABEL(.UPDATE3CONJB)                                                                  \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						  \
    LABEL(.UPDATE1CONJB)                                                                  \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    JMP(.CONCLUDE)                                                                        \
																						  \
    /* Handling when beta == 0 */                                                         \
    LABEL(.STORE_ROW_EDGE_1_TO_4)                                                         \
    CMP(imm(0x3), R11)                                                                    \
    JZ(.UPDATE3RCONJB)                                                                    \
    CMP(imm(0x1), R11)                                                                    \
    JZ(.UPDATE1RCONJB)                                                                    \
    LABEL(.UPDATE3RCONJB)                                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
    VMOVUPD(ZMM(12), MEM(RCX, RDI, 1) MASK_(k(3)))                                        \
    VMOVUPD(ZMM(18), MEM(RCX, RDI, 2) MASK_(k(3)))                                        \
																						  \
    JMP(.CONCLUDE)                                                                        \
                                                                                          \
    LABEL(.UPDATE1RCONJB)                                                                 \
    VMOVUPD(ZMM(6), MEM(RCX) MASK_(k(3)))                                                 \
																						  \
    JMP(.CONCLUDE)


#define ZGEMM_4MASKx2_CONJA_CONJB                                                         \
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
    MOV(VAR(conja_array), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(30))                                                        \
    MOV(VAR(conjb_array), R9)                                                                       \
    VBROADCASTSD(MEM(R9), ZMM(31))                                                        \
																						  \
    MOV(var(k_iter), R8)                                                                  \
		TEST(R8, R8)														              \
    JE(.ZKLEFT_EDGE_1_TO_4)                                                               \
    LABEL(.ZKITERLOOP_BP_EDGE_1_TO_4)                                                     \
																						  \
    MICRO_TILE_4x2_MASK_SET1_CONJA_CONJB                                                  \
    MICRO_TILE_4x2_MASK_SET2_CONJA_CONJB                                                  \
    MICRO_TILE_4x2_MASK_SET1_CONJA_CONJB                                                  \
    MICRO_TILE_4x2_MASK_SET2_CONJA_CONJB                                                  \
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
    MICRO_TILE_4x2_MASK_SET1_CONJA_CONJB                                                  \
																						  \
    DEC(R8)             /* k_left -= 1 */                                                 \
    JNZ(.ZKLEFTLOOP_EDGE_1_TO_4)                                                          \
																						  \
    /**/                                                                                  \
    /*  ZMM(5), ZMM(7), ... , ZMM(27) contain accumulations due to */                     \
    /*  real components broadcasted from B. */                                            \
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
    JZ(.UPDATE1CONJACONJB)                                                                \
    LABEL(.UPDATE3)                                                                       \
    BETA_GEN_ROW_1x4_MASK(RCX, 5, 6)                                                      \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 11, 12)                                                    \
    ADD(RDI, RCX)                                                                         \
    BETA_GEN_ROW_1x4_MASK(RCX, 17, 18)                                                    \
    JMP(.CONCLUDE)                                                                        \
																						  \
    LABEL(.UPDATE1CONJACONJB)                                                             \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                  \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                  \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
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


#define ZGEMM_2x2_CONJA                                                     \
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
    MOV(VAR(conja_array), R9)                                               \
    VMOVUPD(MEM(R9), ZMM(30))                                               \
																			\
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    VMULPD(YMM(30), YMM(0), YMM(0))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    VMULPD(YMM(30), YMM(1), YMM(1))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                  \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    VMULPD(YMM(30), YMM(0), YMM(0))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    VMULPD(YMM(30), YMM(1), YMM(1))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VFMADD231PD(mem_1to8(RBX, R15, 1, 8), ZMM(1), ZMM(16))                  \
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
    VMULPD(YMM(30), YMM(0), YMM(0))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VFMADD231PD(mem_1to8(RBX, 8), ZMM(0),  ZMM(6))                          \
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


#define ZGEMM_2x2_CONJB                                                     \
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
    MOV(VAR(conjb_array), R9)                                               \
    VMOVUPD(MEM(R9), ZMM(30))                                               \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                                    \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                                     \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(1),  ZMM(14))                                   \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(16))                                    \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                                    \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                                     \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(1),  ZMM(14))                                   \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(16))                                    \
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
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(30), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                                    \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(30), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                                     \
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


#define ZGEMM_2x2_CONJA_CONJB                                               \
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
    MOV(VAR(conja_array), R9) \
    VMOVUPD(MEM(R9), ZMM(30)) \
    MOV(VAR(conjb_array), R9) \
    VMOVUPD(MEM(R9), ZMM(31)) \
																			\
    /* Setting iterator for k */                                            \
    MOV(VAR(k_iter), R8)                                                    \
    TEST(R8, R8)                                                            \
    JE(.ZKLEFTZGEMM_2)                                                      \
    LABEL(.ZKITERMAINZGEMM_2)                                               \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    VMULPD(YMM(0), YMM(30), YMM(0)) \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                          \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    VMULPD(YMM(1), YMM(30), YMM(1)) \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(16))                  \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(0))                                               \
    VMULPD(YMM(0), YMM(30), YMM(0)) \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                          \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                   \
    /* Adjusting addresses for next micro tiles */                          \
    ADD(R14, RBX)                                                           \
    ADD(R13, RAX)                                                           \
																			\
    /* Macro for 2x4 micro-tile evaluation   */                             \
    VMOVUPD(MEM(RAX), YMM(1))                                               \
    VMULPD(YMM(1), YMM(30), YMM(1)) \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(1), ZMM(13))                            \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(1),  ZMM(14))                         \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(1), ZMM(15))                     \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4)) \
    VFMADD231PD(ZMM(4), ZMM(1), ZMM(16))                  \
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
    VMULPD(YMM(0), YMM(30), YMM(0))                                         \
    /* Prebroadcasting B on YMM(13) and YMM(14) */                          \
    VFMADD231PD( mem_1to8(RBX), ZMM(0), ZMM(5))                             \
    VMULPD(mem_1to8(RBX, 8), ZMM(31), ZMM(4))                               \
    VFMADD231PD(ZMM(4), ZMM(0),  ZMM(6))                                    \
    /* Prebroadcasting B on YMM(3) and YMM(4) */                            \
    VFMADD231PD(mem_1to8(RBX, R15, 1), ZMM(0), ZMM(7))                      \
    VMULPD(mem_1to8(RBX, R15, 1, 8), ZMM(31), ZMM(4))                       \
    VFMADD231PD(ZMM(4), ZMM(0), ZMM(8))                                     \
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
