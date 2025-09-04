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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/* NOTE : This file contains the macros used to implement CGEMM native
          kernels of the following (MR, NR) pairs :
          Column-major : (24, 4)
          Row-major    : (4, 24)
   The macro for micro-tile computation(SUB_ITER_24x4) accepts the load
   and broadcast addresses as parameters. Thus, we could use this macro
   for in both 24x4 col major kernel and 4x24 row major kernel, by passing
   the appropriate load and broadcast address registers.
*/

/* Macro to set all the registers to zero */
#define SET_ZERO() \
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
  VXORPS(ZMM(29), ZMM(29), ZMM(29)) \
  VXORPS(ZMM(30), ZMM(30), ZMM(30)) \
  VXORPS(ZMM(31), ZMM(31), ZMM(31)) \

/* Macro to perform the rank-1 update using loads and broadcasts from the matrices(24x4) */
/* RL represents the register that has load address, RB represents broadcast address */
/* For the sake of comments, let's assume RL = A, and RB = B(column-major kernel) */
#define SUB_ITER_24x4(n, RL, RB) \
  VMOVAPS(MEM(RL, (24 * n + 0) * 8), ZMM(0))      /* ZMM(0) = A[0:7][n] */ \
  VMOVAPS(MEM(RL, (24 * n + 8) * 8), ZMM(1))      /* ZMM(1) = A[8:15][n] */ \
  VMOVAPS(MEM(RL, (24 * n + 16) * 8), ZMM(2))     /* ZMM(2) = A[16:23][n] */ \
\
  VBROADCASTSS(MEM(RB, (8 * n + 0) * 4), ZMM(28)) /* ZMM(28)  = Real(B[n][0]) */ \
  VBROADCASTSS(MEM(RB, (8 * n + 1) * 4), ZMM(29)) /* ZMM(29)  = Imag(B[n][0]) */ \
  VFMADD231PS(ZMM(0), ZMM(28), ZMM(3))            /* ZMM(3)  = A[0:7][n] * Real(B[n][0]) */ \
  VFMADD231PS(ZMM(1), ZMM(28), ZMM(4))            /* ZMM(4)  = A[8:15][n] * Real(B[n][0]) */ \
  VFMADD231PS(ZMM(2), ZMM(28), ZMM(5))            /* ZMM(5)  = A[16:23][n] * Real(B[n][0]) */ \
  VFMADD231PS(ZMM(0), ZMM(29), ZMM(15))           /* ZMM(15) = A[0:7][n] * Imag(B[n][0]) */ \
  VFMADD231PS(ZMM(1), ZMM(29), ZMM(16))           /* ZMM(16) = A[8:15][n] * Imag(B[n][0]) */ \
  VFMADD231PS(ZMM(2), ZMM(29), ZMM(17))           /* ZMM(17) = A[16:23][n] * Imag(B[n][0]) */ \
\
  VBROADCASTSS(MEM(RB, (8 * n + 2) * 4), ZMM(30)) /* ZMM(30)  = Real(B[n][1]) */ \
  VBROADCASTSS(MEM(RB, (8 * n + 3) * 4), ZMM(31)) /* ZMM(31)  = Imag(B[n][1]) */ \
  VFMADD231PS(ZMM(0), ZMM(30), ZMM(6))            /* ZMM(6)  = A[0:7][n] * Real(B[n][1]) */ \
  VFMADD231PS(ZMM(1), ZMM(30), ZMM(7))            /* ZMM(7)  = A[8:15][n] * Real(B[n][1]) */ \
  VFMADD231PS(ZMM(2), ZMM(30), ZMM(8))            /* ZMM(8)  = A[16:23][n] * Real(B[n][1]) */ \
  VFMADD231PS(ZMM(0), ZMM(31), ZMM(18))           /* ZMM(18) = A[0:7][n] * Imag(B[n][1]) */ \
  VFMADD231PS(ZMM(1), ZMM(31), ZMM(19))           /* ZMM(19) = A[8:15][n] * Imag(B[n][1]) */ \
  VFMADD231PS(ZMM(2), ZMM(31), ZMM(20))           /* ZMM(20) = A[16:23][n] * Imag(B[n][1]) */ \
\
  VBROADCASTSS(MEM(RB, (8 * n + 4) * 4), ZMM(28)) /* ZMM(28) = Real(B[n][2]) */ \
  VBROADCASTSS(MEM(RB, (8 * n + 5) * 4), ZMM(29)) /* ZMM(29) = Imag(B[n][2]) */ \
  VFMADD231PS(ZMM(0), ZMM(28), ZMM(9))            /* ZMM(9)  = A[0:7][n] * Real(B[n][2]) */ \
  VFMADD231PS(ZMM(1), ZMM(28), ZMM(10))           /* ZMM(10) = A[8:15][n] * Real(B[n][2]) */ \
  VFMADD231PS(ZMM(2), ZMM(28), ZMM(11))           /* ZMM(11) = A[16:23][n] * Real(B[n][2]) */ \
  VFMADD231PS(ZMM(0), ZMM(29), ZMM(21))           /* ZMM(21) = A[0:7][n] * Imag(B[n][2]) */ \
  VFMADD231PS(ZMM(1), ZMM(29), ZMM(22))           /* ZMM(22) = A[8:15][n] * Imag(B[n][2]) */ \
  VFMADD231PS(ZMM(2), ZMM(29), ZMM(23))           /* ZMM(23) = A[16:23][n] * Imag(B[n][2]) */ \
\
  VBROADCASTSS(MEM(RB, (8 * n + 6) * 4), ZMM(30)) /* ZMM(30) = Real(B[n][3]) */ \
  VBROADCASTSS(MEM(RB, (8 * n + 7) * 4), ZMM(31)) /* ZMM(31) = Imag(B[n][3]) */ \
  VFMADD231PS(ZMM(0), ZMM(30), ZMM(12))           /* ZMM(12) = A[0:7][n] * Real(B[n][3]) */ \
  VFMADD231PS(ZMM(1), ZMM(30), ZMM(13))           /* ZMM(13) = A[8:15][n] * Real(B[n][3]) */ \
  VFMADD231PS(ZMM(2), ZMM(30), ZMM(14))           /* ZMM(14) = A[16:23][n] * Real(B[n][3]) */ \
  VFMADD231PS(ZMM(0), ZMM(31), ZMM(24))           /* ZMM(24) = A[0:7][n] * Imag(B[n][3]) */ \
  VFMADD231PS(ZMM(1), ZMM(31), ZMM(25))           /* ZMM(25) = A[8:15][n] * Imag(B[n][3]) */ \
  VFMADD231PS(ZMM(2), ZMM(31), ZMM(26))           /* ZMM(26) = A[16:23][n] * Imag(B[n][3]) */ \

/* Macro to scale the registers */
/* 'B' represents the broadcasted register, to be used for scaling */
#define SCALE(B, R1, O1, R2, O2, R3, O3) \
  VMULPS(ZMM(B), ZMM(R1), ZMM(O1)) /* ZMM(O1) = ZMM(B) * ZMM(R1) */ \
  VMULPS(ZMM(B), ZMM(R2), ZMM(O2)) /* ZMM(O2) = ZMM(B) * ZMM(R2) */ \
  VMULPS(ZMM(B), ZMM(R3), ZMM(O3)) /* ZMM(O3) = ZMM(B) * ZMM(R3) */ \

/* Macro to shuffle even and odd indexed elements in a ZMM register */
#define PERMUTE(I1, I2, I3) \
  /* For col major kernel: ZMM(I1) = { Ai0.Bi0, Ar0.Bi0, Ai1.Bi1, Ar1.Bi1, ... }
     For row major kernel: ZMM(I1) = { Bi0.Ai0, Br0.Ai0, Bi1.Ai1, Br1.Ai1, ... }.
     Similarly done for the other registers */ \
  VPERMILPS(IMM(0xB1), ZMM(I1), ZMM(I1)) \
  VPERMILPS(IMM(0xB1), ZMM(I2), ZMM(I2)) \
  VPERMILPS(IMM(0xB1), ZMM(I3), ZMM(I3)) \

/* Macro to reduce a real and imag accumulator pair, as per complex arithmetic */
/* Macro assumes that ZMM(0) has 1.0f broadcasted in it */
#define FMADDSUB(R1, I1, R2, I2, R3, I3) \
  /* ZMM(R1) = ZMM(R1) - 1.0f * ZMM(I1)
     For col major kernel: ZMM(R1) = { Ar0.Br0 - Ai0.Bi0, Ai0.Br0 + Ar0.Bi0 }
     For row major kernel: ZMM(R1) = { Br0.Ar0 - Bi0.Ai0, Bi0.Ar0 + Br0.Ai0 }
     Similarly done for the other pairs */ \
  VFMADDSUB132PS(ZMM(0), ZMM(I1), ZMM(R1)) \
  VFMADDSUB132PS(ZMM(0), ZMM(I2), ZMM(R2)) \
  VFMADDSUB132PS(ZMM(0), ZMM(I3), ZMM(R3)) \

/* Macro to handle alpha scaling when alpha is -1.0f */
/* Macro assumes that ZMM(1) and ZMM(2) have real and imag components
   of alpha already broadcasted */
/* ZMM(S1) = ZMM(S2) = ZMM(S3) = 0.0f, for alpha-scaling */
#define ALPHA_MINUS_ONE(R1, S1, R2, S2, R3, S3) \
  VSUBPS(ZMM(R1), ZMM(S1), ZMM(R1)) /* ZMM(R1) = 0.0f - ZMM(R1) */ \
  VSUBPS(ZMM(R2), ZMM(S2), ZMM(R2)) /* ZMM(R2) = 0.0f - ZMM(R2) */ \
  VSUBPS(ZMM(R3), ZMM(S3), ZMM(R3)) /* ZMM(R3) = 0.0f - ZMM(R3) */ \

/* Macro to hadnle alpha scaling in generic case */
/* Macro assumes that ZMM(1) and ZMM(2) have real and imag components
   of alpha already broadcasted */
#define ALPHA_DEFAULT(R1, I1, R2, I2, R3, I3) \
  /* Scale with real and imag components of beta */ \
  /* Assume ZMM(R1) = { Ar0, Ai0, ... } */ \
  /* ZMM(I1) = { Ar0.alphai, Ai0.alphai, ... }
     Similarly done for other registers */ \
  SCALE(2, R1, I1, R2, I2, R3, I3) \
  /* ZMM(R1) = { Ar0.alphar, Ai0.alphar, ... }
     Similarly done for other registers */ \
  SCALE(1, R1, R1, R2, R2, R3, R3) \
\
  /* Shuffle the imag accumulators for reduction */ \
  /* ZMM(I1) = { Ai0.alphai, Ar0.alphai, ... }
     Similarly done for other registers */ \
  PERMUTE(I1, I2, I3) \
\
  /* Reduce using fmaddsub instruction */ \
  /* ZMM(R1) = { Ar0.alphar - Ai0.alphai, Ai0.alphar + Ar0.alphai, ... }
     Similarly done for other registers */ \
  FMADDSUB(R1, I1, R2, I2, R3, I3) \

/* Macro to handle beta scaling when beta is 1.0f, with primary storage */
#define BETA_ONE_PRIMARY(R1, C1, R2, C2, R3, C3) \
  /* Load C onto the registers*/ \
  VMOVUPS(MEM(RCX), ZMM(C1)) \
  VMOVUPS(MEM(RCX, 64), ZMM(C2)) \
  VMOVUPS(MEM(RCX, 128), ZMM(C3)) \
\
  /* Add C to the result of A*B */ \
  VADDPS(ZMM(C1), ZMM(R1), ZMM(C1)) /* ZMM(C1) = ZMM(R1) + ZMM(C1) */ \
  VADDPS(ZMM(C2), ZMM(R2), ZMM(C2)) /* ZMM(C2) = ZMM(R2) + ZMM(C2) */ \
  VADDPS(ZMM(C3), ZMM(R3), ZMM(C3)) /* ZMM(C3) = ZMM(R2) + ZMM(C3) */ \
\
  /* Store the results onto C */ \
  VMOVUPS(ZMM(C1), MEM(RCX)) \
  VMOVUPS(ZMM(C2), MEM(RCX, 64)) \
  VMOVUPS(ZMM(C3), MEM(RCX, 128)) \

/* Macro to handle beta scaling when beta is -1.0f, with primary storage */
#define BETA_MINUS_ONE_PRIMARY(R1, C1, R2, C2, R3, C3) \
  /* Load C onto the registers*/ \
  VMOVUPS(MEM(RCX), ZMM(C1)) \
  VMOVUPS(MEM(RCX, 64), ZMM(C2)) \
  VMOVUPS(MEM(RCX, 128), ZMM(C3)) \
\
  /* Subtract C from the result of alpha*A*B(use pre-existing macro) */ \
  /* ZMM(C1) = ZMM(R1) - ZMM(C1)
     Similarly done for other registers */ \
  ALPHA_MINUS_ONE(C1, R1, C2, R2, C3, R3) \
\
  /* Store the results onto C */ \
  VMOVUPS(ZMM(C1), MEM(RCX)) \
  VMOVUPS(ZMM(C2), MEM(RCX, 64)) \
  VMOVUPS(ZMM(C3), MEM(RCX, 128)) \

/* Macro to scale a set of 3 registers with beta, with primary storage */
/* Macro assumes that ZMM(1) and ZMM(2) have real and imag components
   of beta already broadcasted */
/* Macro uses C1...C3 to load C, and R1...R3 contains the result of alpha*A*B */
/* Macro uses ZMM(28) - ZMM(30) for beta*C computation */
#define BETA_DEFAULT_PRIMARY(R1, C1, R2, C2, R3, C3) \
  /* Load C onto the registers*/ \
  VMOVUPS(MEM(RCX), ZMM(C1)) \
  VMOVUPS(MEM(RCX, 64), ZMM(C2)) \
  VMOVUPS(MEM(RCX, 128), ZMM(C3)) \
\
  /* Scale C by beta(using pre-existing macro) */ \
  /* Assume ZMM(C1) = { Cr0, Ci0, ... } */ \
  /* ZMM(C1) = { Cr0.betar - Ci0.betai, Ci0.betar + Cr0.betai, ... }
     Similarly done for other registers */ \
  ALPHA_DEFAULT(C1, 28, C2, 29, C3, 30) \
\
  /* Add beta*C to the result of alpha*A*B */ \
  VADDPS(ZMM(C1), ZMM(R1), ZMM(C1)) /* ZMM(C1) = ZMM(R1) + ZMM(C1) */ \
  VADDPS(ZMM(C2), ZMM(R2), ZMM(C2)) /* ZMM(C2) = ZMM(R2) + ZMM(C2) */ \
  VADDPS(ZMM(C3), ZMM(R3), ZMM(C3)) /* ZMM(C3) = ZMM(R2) + ZMM(C3) */ \
\
  /* Store the results onto C */ \
  VMOVUPS(ZMM(C1), MEM(RCX)) \
  VMOVUPS(ZMM(C2), MEM(RCX, 64)) \
  VMOVUPS(ZMM(C3), MEM(RCX, 128)) \

/* Macro to perform 8x8 transpose of 64-bit elements */
/* Transpose is in-place(R0-R7), T0-T7 are temporary registers */
#define TRANSPOSE_8X8(R0, R1, R2, R3, R4, R5, R6, R7, \
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

/* Macro to scale a register with beta, with secondary storage */
/* Macro uses C1 to load C, and R1 contains the result of alpha*A*B */
/* Macro assumes that ZMM(1) and ZMM(2) have real and imag components
   of beta already broadcasted */
#define BETA_DEFAULT_SECONDARY(R1, C1, T1) \
  /* Load C onto the register*/ \
  VMOVUPS(MEM(RCX), YMM(C1)) \
\
  /* Scale C by beta */ \
  /* Assume ZMM(C1) = { Cr0, Ci0, ... } */ \
  /* YMM(C1) = { Cr0.betar - Ci0.betai, Ci0.betar + Cr0.betai, ... } */ \
  VMULPS(YMM(2), YMM(C1), YMM(T1)) \
  VMULPS(YMM(1), YMM(C1), YMM(C1)) \
  VPERMILPS(IMM(0xB1), YMM(T1), YMM(T1)) \
  VFMADDSUB132PS(YMM(0), YMM(T1), YMM(C1)) \
\
  /* Add beta*C to the result of alpha*A*B */ \
  VADDPS(YMM(C1), YMM(R1), YMM(C1)) /* ZMM(C1) = ZMM(R1) + ZMM(C1) */ \
\
  /* Store the result onto C */ \
  VMOVUPS(YMM(C1), MEM(RCX)) \

/* Macro to scale a register with beta, with general strides */
/* Macro gets alpha*A*B in R1, R2, R3 for a column */
/* Macro uses ZMM(15) - ZMM(17) to gather C */
/* Macro uses ZMM(18) - ZMM(20) as temmporary registers */
/* Macro assumes that ZMM(28) - ZMM(30) have the addresses of C for
   gather/scatter(one column at a time) */
/* Macro assumes that ZMM(1) and ZMM(2) have real and imag components
   of beta already broadcasted */
#define BETA_DEFAULT_GENERAL(R1, R2, R3) \
    /* Set the masks to 1 */ \
    KXNORW(K(0), K(0), K(1)) \
    KXNORW(K(0), K(0), K(2)) \
    KXNORW(K(0), K(0), K(3)) \
\
    /* Gather elements from C, one column at a time */ \
    VGATHERQPD(MEM(RCX, ZMM(28), 1), ZMM(15) MASK_K(1)) \
    VGATHERQPD(MEM(RCX, ZMM(29), 1), ZMM(16) MASK_K(2)) \
    VGATHERQPD(MEM(RCX, ZMM(30), 1), ZMM(17) MASK_K(3)) \
\
    /* Scale C by beta */ \
    /* Assume ZMM(15) = { Cr0, Ci0, ... } */ \
    /* ZMM(15) = { Cr0.betar - Ci0.betai, Ci0.betar + Cr0.betai, ... } */ \
    VMULPS(ZMM(2), ZMM(15), ZMM(18)) \
    VMULPS(ZMM(1), ZMM(15), ZMM(15)) \
    VPERMILPS(IMM(0xB1), ZMM(18), ZMM(18)) \
    VFMADDSUB132PS(ZMM(0), ZMM(18), ZMM(15)) \
\
    /* Scale C by beta */ \
    /* Assume ZMM(16) = { Cr1, Ci1, ... } */ \
    /* ZMM(16) = { Cr1.betar - Ci1.betai, Ci1.betar + Cr1.betai, ... } */ \
    VMULPS(ZMM(2), ZMM(16), ZMM(19)) \
    VMULPS(ZMM(1), ZMM(16), ZMM(16)) \
    VPERMILPS(IMM(0xB1), ZMM(19), ZMM(19)) \
    VFMADDSUB132PS(ZMM(0), ZMM(19), ZMM(16)) \
\
    /* Scale C by beta */ \
    /* Assume ZMM(17) = { Cr2, Ci2, ... } */ \
    /* ZMM(17) = { Cr2.betar - Ci2.betai, Ci2.betar + Cr2.betai, ... } */ \
    VMULPS(ZMM(2), ZMM(17), ZMM(20)) \
    VMULPS(ZMM(1), ZMM(17), ZMM(17)) \
    VPERMILPS(IMM(0xB1), ZMM(20), ZMM(20)) \
    VFMADDSUB132PS(ZMM(0), ZMM(20), ZMM(17)) \
\
    /* Add beta*C to the result of alpha*A*B */ \
    VADDPS(ZMM(15), ZMM(R1), ZMM(15)) /* ZMM(15) = ZMM(R1) + ZMM(15) */ \
    VADDPS(ZMM(16), ZMM(R2), ZMM(16)) /* ZMM(16) = ZMM(R2) + ZMM(16) */ \
    VADDPS(ZMM(17), ZMM(R3), ZMM(17)) /* ZMM(17) = ZMM(R3) + ZMM(17) */ \
\
    /* Reset the mask to 1 */ \
    KXNORW(K(0), K(0), K(1)) \
    KXNORW(K(0), K(0), K(2)) \
    KXNORW(K(0), K(0), K(3)) \
\
    /* Scatter the result to C, one column at a time */ \
    VSCATTERQPD(ZMM(15), MEM(RCX, ZMM(28), 1) MASK_K(1)) \
    VSCATTERQPD(ZMM(16), MEM(RCX, ZMM(29), 1) MASK_K(2)) \
    VSCATTERQPD(ZMM(17), MEM(RCX, ZMM(30), 1) MASK_K(3)) \

/* Macro to store alpha*A*B onto C, with general strides */
/* Macro gets alpha*A*B in R1, R2, R3 for a column */
/* Macro assumes that ZMM(28) - ZMM(30) have the addresses of C for
   scatter(one column at a time) */
#define BETA_ZERO_GENERAL(R1, R2, R3) \
    /* Set the masks to 1 */ \
    KXNORW(K(0), K(0), K(1)) \
    KXNORW(K(0), K(0), K(2)) \
    KXNORW(K(0), K(0), K(3)) \
\
    /* Scatter the result to C, one column at a time */ \
    VSCATTERQPD(ZMM(R1), MEM(RCX, ZMM(28), 1) MASK_K(1)) \
    VSCATTERQPD(ZMM(R2), MEM(RCX, ZMM(29), 1) MASK_K(2)) \
    VSCATTERQPD(ZMM(R3), MEM(RCX, ZMM(30), 1) MASK_K(3))

