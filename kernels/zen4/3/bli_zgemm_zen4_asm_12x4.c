/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define A_L1_PREFETCH_DIST 4 // in units of k iterations
#define B_L1_PREFETCH_DIST 4
#define TAIL_NITER 6

#define PREFETCH_A_L1(n, k) \
  PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST * 24 * 8 + (2 * n + k) * 24 * 4))
#define PREFETCH_B_L1(n, k) \
  PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST * 8 * 8 + (2 * n + k) * 8 * 4))

#define LOOP_ALIGN ALIGN32

/******************************/
/* Scale R1 register by alpha */
/* Inputs:                    */
/* R1 = A * B                 */
/* ZMM0 = Alpha real          */
/* ZMM1 = Alpha imag          */
/* Output:                    */
/* R1 = (Alpha) * R1          */
/******************************/
#define SCALE_BY_ALPHA(R1) \
  VPERMILPD(IMM(0x55), ZMM(R1), ZMM(9)) \
  VMULPD(ZMM0, ZMM(R1), ZMM(R1)) \
  VMULPD(ZMM1, ZMM(9), ZMM(9)) \
  VFMADDSUB132PD(ZMM(4), ZMM(9), ZMM(R1))

/* Scale R1, R2, R3 register by alpha */
#define SCALE3R_BY_ALPHA(R1, R2, R3) \
  SCALE_BY_ALPHA(R1) \
  SCALE_BY_ALPHA(R2) \
  SCALE_BY_ALPHA(R3)

/* Set R1, R2, R3 to 0 */
#define SET_REG_TO_ZERO(R1, R2, R3) \
  VXORPD(ZMM(R1), ZMM(R1), ZMM(R1)) \
  VXORPD(ZMM(R2), ZMM(R2), ZMM(R2)) \
  VXORPD(ZMM(R3), ZMM(R3), ZMM(R3))

/*****************************/
/* Scale R1 register by beta */
/* Inputs:                   */
/* R1 = A * B                */
/* ZMM2 = Beta real          */
/* ZMM3 = Beta imag          */
/* Output:                   */
/* R1 = (Beta) * R1          */
/*****************************/
#define SCALE_BY_BETA(R1) \
  VPERMILPD(IMM(0x55), ZMM(R1), ZMM(9)) \
  VMULPD(ZMM2, ZMM(R1), ZMM(R1)) \
  VMULPD(ZMM3, ZMM(9), ZMM(9)) \
  VFMADDSUB132PD(ZMM(4), ZMM(9), ZMM(R1))

/***************************************/
/* Scale R1/R2/R3 register by beta and */
/* store the scaled value to C buffer  */
/* Inputs:                             */
/* R1/R2/R3 = Alpha * A * B            */
/* RBX = beta                          */
/* ZMM(0) = C                          */
/* Output:                             */
/* C = RBX * ZMM(0) + R1/R2/R3         */
/***************************************/
#define UPDATE_C_BETASCALE(R1, R2, R3) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  VMOVUPD(MEM(RCX, RDI, 4), ZMM(8)) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX, RDI, 4)) \
  VMOVUPD(MEM(RCX, RDI, 8), ZMM(8)) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX, RDI, 8)) \
  ADD(RSI, RCX)

/**************************************/
/* Add C buffer value to R1/R2/R3 reg */
/* and store the output to C buffer   */
/* Inputs:                            */
/* R1/R2/R3 = Alpha * A * B           */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = ZMM(0) + R1/R2/R3              */
/**************************************/
#define UPDATE_C_BETA1(R1, R2, R3) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  VMOVUPD(MEM(RCX, RDI, 4), ZMM(8)) \
  VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX, RDI, 4)) \
  VMOVUPD(MEM(RCX, RDI, 8), ZMM(8)) \
  VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX, RDI, 8)) \
  ADD(RSI, RCX)

/****************************************/
/* Sub C buffer value with R1/R2/R3 reg */
/* and store the output to C buffer     */
/* Inputs:                              */
/* R1/R2/R3 = Alpha * A * B             */
/* ZMM(0) = C                           */
/* Output:                              */
/* C = -ZMM(0) + R1/R2/R3               */
/****************************************/
#define UPDATE_C_BETAMINUS1(R1, R2, R3) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  VSUBPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  VMOVUPD(MEM(RCX, RDI, 4), ZMM(8)) \
  VSUBPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX, RDI, 4)) \
  VMOVUPD(MEM(RCX, RDI, 8), ZMM(8)) \
  VSUBPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX, RDI, 8)) \
  ADD(RSI, RCX)

/***************************************/
/* Store R1/R2/R3 reg to C buffer      */
/* Input:                              */
/* R1/R2/R3 = Beta * C + Alpha * A * B */
/* Output:                             */
/* C = R1/R2/R3                        */
/***************************************/
#define STORE_C(R1, R2, R3) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  VMOVUPD(ZMM(R2), MEM(RCX, RDI, 4)) \
  VMOVUPD(ZMM(R3), MEM(RCX, RDI, 8)) \
  ADD(RSI, RCX)

/**************************************/
/* Scale R(1-4) register by beta and  */
/* store the scaled value to C buffer */
/* Inputs:                            */
/* R(1-4) = Alpha * A * B             */
/* RBX = beta                         */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = RBX * ZMM(0) + R(1-4)          */
/**************************************/
#define UPDATE_C_BETASCALE_ROW(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX)) \
  ADD(RSI, RCX) \
  EXTRACT_C_ROW(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R4), ZMM(R4)) \
  VMOVUPD(ZMM(R4), MEM(RCX)) \
  ADD(RSI, RCX)

/************************************/
/* Add C buffer value to R(1-4) reg */
/* and store the output to C buffer */
/* Inputs:                          */
/* R(1-4)  = Alpha * A * B          */
/* ZMM(0) = C                       */
/* Output:                          */
/* C = ZMM(0) + R(1-4)              */
/************************************/
#define UPDATE_C_BETA1_ROW(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX)) \
  ADD(RSI, RCX) \
  EXTRACT_C_ROW(8) \
  VADDPD(ZMM(8), ZMM(R4), ZMM(R4)) \
  VMOVUPD(ZMM(R4), MEM(RCX)) \
  ADD(RSI, RCX)

/**************************************/
/* Sub C buffer value with R(1-4) reg */
/* and store the output to C buffer   */
/* Inputs:                            */
/* R(1-4) = Alpha * A * B             */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = -ZMM(0) + R(1-4)               */
/**************************************/
#define UPDATE_C_BETAMINUS1_ROW(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VSUBPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VSUBPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  VMOVUPD(ZMM(R2), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C_ROW(8) \
  VSUBPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  VMOVUPD(ZMM(R3), MEM(RCX)) \
  ADD(RSI, RCX) \
  EXTRACT_C_ROW(8) \
  VSUBPD(ZMM(8), ZMM(R4), ZMM(R4)) \
  VMOVUPD(ZMM(R4), MEM(RCX)) \
  ADD(RSI, RCX)

/*************************************/
/* Store R(1-4) reg to C buffer      */
/* Input:                            */
/* R(1-4) = Beta * C + Alpha * A * B */
/* Output:                           */
/* C = R(1-4)                        */
/*************************************/
#define STORE_C_ROW(R1, R2, R3, R4) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(ZMM(R2), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(ZMM(R3), MEM(RCX)) \
  ADD(RSI, RCX) \
  VMOVUPD(ZMM(R4), MEM(RCX)) \
  ADD(RSI, RCX)

/************************************/
/* Extract 4 elements from C buffer */
/* As the kernel is col major,      */
/* elements are in col major order  */
/* Input:                           */
/* RCX = C                          */
/* Output:                          */
/* R1 = C                           */
/************************************/
#define EXTRACT_C_ROW(R1) \
  VMOVUPD(MEM(RCX), XMM(R1)) \
  VMOVUPD(MEM(RCX, RDI, 1), XMM9) \
  VINSERTF128(IMM(1), XMM9, YMM(R1), YMM(R1)) \
  VMOVUPD(MEM(RCX, RDI, 2), XMM9) \
  VMOVUPD(MEM(RCX, R12, 1), XMM10) \
  VINSERTF128(IMM(1), XMM10, YMM9, YMM9) \
  VINSERTF64X4(IMM(1), YMM9, ZMM(R1), ZMM(R1))

/**************************************/
/* Scale R1 register by alpha and     */
/* scale C buffer with beta and store */
/* the output to C buffer             */
/* Inputs:                            */
/* R1 =  A X B                        */
/* RAX = alpha                        */
/* RBX = beta                         */
/* ZMM8 = C                           */
/* Output:                            */
/* C = RBX * ZMM8 + RAX * R1          */
/**************************************/
#define UPDATE_C_ROW(R1) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RDI, RCX)

/**************************************/
/* Scale R(1-4) register by beta and  */
/* store the scaled value to C buffer */
/* Inputs:                            */
/* R(1-4) = Alpha * A * B             */
/* RBX = beta                         */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = RBX * ZMM(0) + R(1-4)          */
/**************************************/
#define UPDATE_C_BETASCALE_GEN(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  STORE_C_GEN(R1) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  STORE_C_GEN(R2) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  STORE_C_GEN(R3) \
  ADD(RSI, RCX) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R4), ZMM(R4)) \
  STORE_C_GEN(R4) \
  ADD(RSI, RCX)

/************************************/
/* Add C buffer value to R(1-4) reg */
/* and store the output to C buffer */
/* Inputs:                          */
/* R(1-4)  = Alpha * A * B          */
/* ZMM(0) = C                       */
/* Output:                          */
/* C = ZMM(0) + R(1-4)              */
/************************************/
#define UPDATE_C_BETA1_GEN(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
   EXTRACT_C(8) \
   VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
   STORE_C_GEN(R1) \
   ADD(RSI, RCX) \
   VMOVUPD(MEM(RCX), ZMM(8)) \
   EXTRACT_C(8) \
   VADDPD(ZMM(8), ZMM(R2), ZMM(R2)) \
   STORE_C_GEN(R2) \
   ADD(RSI, RCX) \
   VMOVUPD(MEM(RCX), ZMM(8)) \
   EXTRACT_C(8) \
   VADDPD(ZMM(8), ZMM(R3), ZMM(R3)) \
   STORE_C_GEN(R3) \
   ADD(RSI, RCX) \
   EXTRACT_C(8) \
   VADDPD(ZMM(8), ZMM(R4), ZMM(R4)) \
   STORE_C_GEN(R4) \
   ADD(RSI, RCX)

/**************************************/
/* Sub C buffer value with R(1-4) reg */
/* and store the output to C buffer   */
/* Inputs:                            */
/* R(1-4) = Alpha * A * B             */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = -ZMM(0) + R(1-4)               */
/**************************************/
#define UPDATE_C_BETAMINUS1_GEN(R1, R2, R3, R4) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  VSUBPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  STORE_C_GEN(R1) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  VSUBPD(ZMM(8), ZMM(R2), ZMM(R2)) \
  STORE_C_GEN(R2) \
  ADD(RSI, RCX) \
  VMOVUPD(MEM(RCX), ZMM(8)) \
  EXTRACT_C(8) \
  VSUBPD(ZMM(8), ZMM(R3), ZMM(R3)) \
  STORE_C_GEN(R3) \
  ADD(RSI, RCX) \
  EXTRACT_C(8) \
  VSUBPD(ZMM(8), ZMM(R4), ZMM(R4)) \
  STORE_C_GEN(R4) \
  ADD(RSI, RCX)

/*************************************/
/* Store R(1-4) reg to C buffer      */
/* Input:                            */
/* R(1-4) = Beta * C + Alpha * A * B */
/* Output:                           */
/* C = R(1-4)                        */
/*************************************/
#define EXTRACT_STORE_C_GEN(R1, R2, R3, R4) \
  STORE_C_GEN(R1) \
  ADD(RSI, RCX) \
  STORE_C_GEN(R2) \
  ADD(RSI, RCX) \
  STORE_C_GEN(R3) \
  ADD(RSI, RCX) \
  STORE_C_GEN(R4) \
  ADD(RSI, RCX)

/**********************************/
/* Store 4 elements from C buffer */
/* for general stride storage     */
/* Input:                         */
/* RCX = C                        */
/* Output:                        */
/* R1 = C                         */
/**********************************/
#define STORE_C_GEN(R1) \
  VEXTRACTF64X2(IMM(0), ZMM(R1), XMM9) \
  VMOVUPD(XMM9, MEM(RCX)) \
  VEXTRACTF64X2(IMM(1), ZMM(R1), XMM9) \
  VMOVUPD(XMM9, MEM(RCX, RDI, 1)) \
  VEXTRACTF64X2(IMM(2), ZMM(R1), XMM9) \
  VMOVUPD(XMM9, MEM(RCX, RDI, 2)) \
  VEXTRACTF64X2(IMM(3), ZMM(R1), XMM9) \
  VMOVUPD(XMM9, MEM(RCX, R12, 1))

/************************************/
/* Extract 4 elements from C buffer */
/* As the kernel is col major,      */
/* elements are in col major order  */
/* Input:                           */
/* RCX = C                          */
/* Output:                          */
/* R1 = C                           */
/************************************/
#define EXTRACT_C(R1) \
  VMOVUPD(MEM(RCX), XMM(R1)) \
  VMOVUPD(MEM(RCX, RDI, 1), XMM9) \
  VINSERTF128(IMM(1), XMM9, YMM(R1), YMM(R1)) \
  VMOVUPD(MEM(RCX, RDI, 2), XMM9) \
  VMOVUPD(MEM(RCX, R12, 1), XMM10) \
  VINSERTF128(IMM(1), XMM10, YMM9, YMM9) \
  VINSERTF64X4(IMM(1), YMM9, ZMM(R1), ZMM(R1))

/**********************************/
/* Scale R1 register by alpha and */
/* scale C buffer with beta and   */
/* the output to C buffer         */
/* Inputs:                        */
/* R1 =  A X B                    */
/* RAX = alpha                    */
/* RBX = beta                     */
/* ZMM8 = C                       */
/* Output:                        */
/* C = RBX * ZMM8 + RAX * R1      */
/**********************************/
#define UPDATE_C_GEN(R1) \
  EXTRACT_C(8) \
  SCALE_BY_BETA(8) \
  VADDPD(ZMM(8), ZMM(R1), ZMM(R1)) \
  VMOVUPD(ZMM(R1), MEM(RCX)) \
  ADD(RSI, RCX)

/**************************************/
/* Scale R1 register by alpha and     */
/* scale C buffer with beta and store */
/* the output to C buffer             */
/* Inputs:                            */
/* R1 =  A * B                        */
/* RAX = alpha                        */
/* RBX = beta                         */
/* ZMM(0) = C                         */
/* Output:                            */
/* C = RBX * ZMM(0) + RAX * R1        */
/* we operate 12x4 block at a time    */
/**************************************/
#define SUBITER(n) \
  /*PREFETCH_A_L1(n, 0)  */ \
  VBROADCASTSD(MEM(RBX, (8 * n + 2) * 8), ZMM(3)) \
  VFMADD231PD(ZMM(0), ZMM(29), ZMM(5)) \
  VFMADD231PD(ZMM(1), ZMM(29), ZMM(6)) \
  VFMADD231PD(ZMM(2), ZMM(29), ZMM(7)) \
  VBROADCASTSD(MEM(RBX, (8 * n + 3) * 8), ZMM(4)) \
  VFMADD231PD(ZMM(0), ZMM(30), ZMM(8)) \
  VFMADD231PD(ZMM(1), ZMM(30), ZMM(9)) \
  VFMADD231PD(ZMM(2), ZMM(30), ZMM(10)) \
\
  /*PREFETCH_B_L1(n, 0)    */ \
  VBROADCASTSD(MEM(RBX, (8 * n + 4) * 8), ZMM(29)) \
  VFMADD231PD(ZMM(0), ZMM(3), ZMM(11)) \
  VFMADD231PD(ZMM(1), ZMM(3), ZMM(12)) \
  VFMADD231PD(ZMM(2), ZMM(3), ZMM(13)) \
  VBROADCASTSD(MEM(RBX, (8 * n + 5) * 8), ZMM(30)) \
  VFMADD231PD(ZMM(0), ZMM(4), ZMM(14)) \
  VFMADD231PD(ZMM(1), ZMM(4), ZMM(15)) \
  VFMADD231PD(ZMM(2), ZMM(4), ZMM(16)) \
\
  /*PREFETCH_A_L1(n, 1)  */ \
  VBROADCASTSD(MEM(RBX, (8 * n + 6) * 8), ZMM(3)) \
  VFMADD231PD(ZMM(0), ZMM(29), ZMM(17)) \
  VFMADD231PD(ZMM(1), ZMM(29), ZMM(18)) \
  VFMADD231PD(ZMM(2), ZMM(29), ZMM(19)) \
  VBROADCASTSD(MEM(RBX, (8 * n + 7) * 8), ZMM(4)) \
  VFMADD231PD(ZMM(0), ZMM(30), ZMM(20)) \
  VFMADD231PD(ZMM(1), ZMM(30), ZMM(21)) \
  VFMADD231PD(ZMM(2), ZMM(30), ZMM(22)) \
\
  /*PREFETCH_B_L1(n, 1) */ \
  VBROADCASTSD(MEM(RBX, (8 * n + 8) * 8), ZMM(29)) \
  VFMADD231PD(ZMM(0), ZMM(3), ZMM(23)) \
  VFMADD231PD(ZMM(1), ZMM(3), ZMM(24)) \
  VFMADD231PD(ZMM(2), ZMM(3), ZMM(25)) \
  VBROADCASTSD(MEM(RBX, (8 * n + 9) * 8), ZMM(30)) \
  VFMADD231PD(ZMM(0), ZMM(4), ZMM(26)) \
  VMOVAPD(MEM(RAX, (12 * n + 0) * 16), ZMM(0)) \
  VFMADD231PD(ZMM(1), ZMM(4), ZMM(27)) \
  VMOVAPD(MEM(RAX, (12 * n + 4) * 16), ZMM(1)) \
  VFMADD231PD(ZMM(2), ZMM(4), ZMM(28)) \
  VMOVAPD(MEM(RAX, (12 * n + 8) * 16), ZMM(2))

  /*********************************************/
  /* Transpose contents of R0, R1 , R2, R3 and */
  /* store the result to same register         */
  /*********************************************/
#define TRANSPOSE(R0, R1, R2, R3) \
  VSHUFF64X2(IMM(0x88), ZMM(R1), ZMM(R0), ZMM(26)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R1), ZMM(R0), ZMM(27)) \
  VSHUFF64X2(IMM(0x88), ZMM(R3), ZMM(R2), ZMM(28)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R3), ZMM(R2), ZMM(29)) \
  VSHUFF64X2(IMM(0x88), ZMM(28), ZMM(26), ZMM(R0)) \
  VSHUFF64X2(IMM(0xDD), ZMM(28), ZMM(26), ZMM(R2)) \
  VSHUFF64X2(IMM(0x88), ZMM(29), ZMM(27), ZMM(R1)) \
  VSHUFF64X2(IMM(0xDD), ZMM(29), ZMM(27), ZMM(R3))


// This array is used to support ADDSUB instruction.
static double offsets[8] __attribute__((aligned(64)))
                                 = {1, 1, 1, 1, 1, 1, 1, 1};

/**********************************************************/
/* Kernel : bli_zgemm_zen4_asm_12x4                       */
/* It performs  C = C * beta + alpha * A * B              */
/* It is col preferred kernel, A and B are packed         */
/* C could be Row/Col/Gen Stored Matrix                   */
/* Registers are allocated as below                       */
/* Load A :  ZMM(0-2)                                     */
/* Pre Broadcast B :  ZMM(29,30)                          */
/* Broadcast B :  ZMM(3,4)                                */
/* Accumulation of A(real,imag)*Breal :                   */
/*       ZMM(5-7,11-13,17-19,23-25)                       */
/* Accumulation of A(real,imag)*Bimag :                   */
/*       ZMM(8-10,14-16,20-22,26-28)                      */
/* Computation of A(real,imag)*B(real,imag):              */
/*       ZMM(5-7,11-13,17-19,23-25)                       */
/* Registers used for load and brodcast could be          */
/* used for alpha, beta scaling                           */
/* alphar : ZMM0, alphai : ZMM1                           */
/* betar  : ZMM2, betai  : ZMM3                           */
/* Techinques used in kernel                              */
/* 1. k loop is sub divided in to 4 loops                 */
/*    a. iter = k/4-TAIL_NITER-4,  ZMM = A*B              */
/*    b. iter = 4, ZMM = A*B,                             */
/*       Prefetch C mem in anticipation of a write.       */
/*    c. iter = TAIL_NITER-4, ZMM = A*B                   */
/*    All above loops is unrolled 4times                  */
/*    d. iter = k%4, ZMM = A*B, k remainder is executed   */
/* 2. If alpha/beta imag = 0 and alpha/beta real = 0/1/-1 */
/*    Scale with real value(Should not be 0/1/-1)         */
/**********************************************************/
void bli_zgemm_zen4_asm_12x4(
    dim_t k0,
    dcomplex *restrict alpha,
    dcomplex *restrict a,
    dcomplex *restrict b,
    dcomplex *restrict beta,
    dcomplex *restrict c, inc_t rs_c0, inc_t cs_c0,
    auxinfo_t *data,
    cntx_t *restrict cntx)
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
  const uint64_t k = k0;
  /*rowstride * size of one dcomplex element*/
  uint64_t rs_c = rs_c0 * 16;
  /*colstride * size of one dcomplex element*/
  uint64_t cs_c = cs_c0 * 16;
  const double *offsetPtr = &offsets[0];

  uint64_t alpha_mul_type = BLIS_MUL_DEFAULT;
  uint64_t beta_mul_type = BLIS_MUL_DEFAULT;

  if (alpha->imag == 0.0)
  {
    if (alpha->real == 1.0)
      alpha_mul_type = BLIS_MUL_ONE;
    else if (alpha->real == -1.0)
      alpha_mul_type = BLIS_MUL_MINUS_ONE;
    else if (alpha->real == 0.0)
      alpha_mul_type = BLIS_MUL_ZERO;
  }

  if (beta->imag == 0.0)
  {
    if (beta->real == 1.0)
      beta_mul_type = BLIS_MUL_ONE ;
    else if (beta->real == -1.0)
      beta_mul_type = BLIS_MUL_MINUS_ONE;
    else if (beta->real == 0.0)
      beta_mul_type = BLIS_MUL_ZERO;
  }

  BEGIN_ASM()

  // Initialise accumulation registers to zero
  VXORPD(ZMM(5), ZMM(5), ZMM(5))
  VXORPD(ZMM(6), ZMM(6), ZMM(6))
  VXORPD(ZMM(7), ZMM(7), ZMM(7))
  VXORPD(ZMM(8), ZMM(8), ZMM(8))
  VXORPD(ZMM(9), ZMM(9), ZMM(9))
  VXORPD(ZMM(10), ZMM(10), ZMM(10))
  VXORPD(ZMM(11), ZMM(11), ZMM(11))
  VXORPD(ZMM(12), ZMM(12), ZMM(12))
  VXORPD(ZMM(13), ZMM(13), ZMM(13))
  VXORPD(ZMM(14), ZMM(14), ZMM(14))
  VXORPD(ZMM(15), ZMM(15), ZMM(15))
  VXORPD(ZMM(16), ZMM(16), ZMM(16))
  VXORPD(ZMM(17), ZMM(17), ZMM(17))
  VXORPD(ZMM(18), ZMM(18), ZMM(18))
  VXORPD(ZMM(19), ZMM(19), ZMM(19))
  VXORPD(ZMM(20), ZMM(20), ZMM(20))
  VXORPD(ZMM(21), ZMM(21), ZMM(21))
  VXORPD(ZMM(22), ZMM(22), ZMM(22))
  VXORPD(ZMM(23), ZMM(23), ZMM(23))
  VXORPD(ZMM(24), ZMM(24), ZMM(24))
  VXORPD(ZMM(25), ZMM(25), ZMM(25))
  VXORPD(ZMM(26), ZMM(26), ZMM(26))
  VXORPD(ZMM(27), ZMM(27), ZMM(27))
  VXORPD(ZMM(28), ZMM(28), ZMM(28))

  MOV(VAR(k), RSI)

  // load address of buff to reg
  MOV(VAR(a), RAX)
  MOV(VAR(b), RBX)
  MOV(VAR(c), RCX)

  // load R9 with address of C buff to be used during prefetch
  MOV(RCX, R9)
  ADD(IMM(63), R9)

  // pre-load first 12 elements of a to ZMM(0-2)
  VMOVAPD(MEM(RAX, 0 * 16), ZMM(0))
  VMOVAPD(MEM(RAX, 4 * 16), ZMM(1))
  VMOVAPD(MEM(RAX, 8 * 16), ZMM(2))
  // broadcast breal to ZMM29 and bimag to ZMM30
  VBROADCASTSD(MEM(RBX, 0), ZMM(29))
  VBROADCASTSD(MEM(RBX, 8), ZMM(30))
  LEA(MEM(RAX, 12 * 16), RAX) // adjust a after pre-load

  MOV(VAR(cs_c), R10)

  MOV(RSI, RDI)
  AND(IMM(3), RSI)
  SAR(IMM(2), RDI)

  /******************************************************************/
  /* Operation:                                                     */
  /* SUBITER = (Ar, Ai)*(Br, Bi) = (Ar, Ai)*Br , (Ar, Ai)*Bi        */
  /* ZMMR1 = (Ar*Br, Ai*Br), ZMMR2 = (Ar*Bi, Ai*Bi)                 */
  /* ITER_K_LOOP: Loop count depends on k and TAIL_NITER            */
  /*              iter = k/4 - 4 - TAIL_NITER                       */
  /* ITER_4: Fixed loop executed 4 times hence iter = 4             */
  /* TAILNITER: Fixed loop executed TAIL_NITER times hence          */
  /*            iter = TAIL_NITER                                   */
  /* Tail: Leftover k values are executed here, iter = k%4          */
  /* k loop is divided in above way to have a fixed distance to     */
  /* prefetch C.                                                    */
  /******************************************************************/
  SUB(IMM(4 + TAIL_NITER), RDI)
  JLE(K_REMAINDER)

  LOOP_ALIGN
  /*******************************************************/
  /* ITER_K_LOOP: iter = k/4 - 4 - TAIL_NITER            */
  /* (Ar, Ai)*(Br, Bi) is executed                       */
  /* Loop is unrolled 4 times                            */
  /*******************************************************/
  LABEL(ITER_K_LOOP)

  SUBITER(0)
  SUBITER(1)
  SUB(IMM(1), RDI)
  SUBITER(2)
  SUBITER(3)

  LEA(MEM(RAX, 4 * 12 * 16), RAX)
  LEA(MEM(RBX, 4 * 4 * 16), RBX)

  JNZ(ITER_K_LOOP)

  LABEL(K_REMAINDER)

  ADD(IMM(4), RDI)
  JLE(TAILNITER)

  LOOP_ALIGN
  /*******************************************************/
  /* ITER_4: iter = 4                                    */
  /* (Ar, Ai)*(Br, Bi) is executed                       */
  /* C is prefetched to L1/L2 cache line with            */
  /* anticipation of write                               */
  /* Loop is unrolled 4 times                            */
  /*******************************************************/
  LABEL(ITER_4)

  PREFETCHW0(MEM(R9))
  SUBITER(0)

  SUBITER(1)
  PREFETCHW0(MEM(R9, 64))

  SUB(IMM(1), RDI)
  SUBITER(2)
  PREFETCHW0(MEM(R9, 128))
  SUBITER(3)

  LEA(MEM(RAX, 4 * 12 * 16), RAX)
  LEA(MEM(RBX, 4 * 4 * 16), RBX)
  LEA(MEM(R9, R10, 1), R9)

  JNZ(ITER_4)

  /*******************************************************/
  /* TAILNITER: iter = TAILNITER                         */
  /* (Ar, Ai)*(Br, Bi) is executed                       */
  /* Loop is unrolled 4 times                            */
  /*******************************************************/
  LABEL(TAILNITER)

  ADD(IMM(0 + TAIL_NITER), RDI)
  JLE(TAIL)

  LOOP_ALIGN
  LABEL(TAILNITER_LOOP)

  SUBITER(0)
  SUBITER(1)
  SUB(IMM(1), RDI)
  SUBITER(2)
  SUBITER(3)

  LEA(MEM(RAX, 4 * 12 * 16), RAX)
  LEA(MEM(RBX, 4 * 4 * 16), RBX)

  JNZ(TAILNITER_LOOP)

  LABEL(TAIL)

  TEST(RSI, RSI)
  JZ(POSTACCUM)

  LOOP_ALIGN
  /*******************************************************/
  /* TAILNITER: iter = k%4                               */
  /* (Ar, Ai)*(Br, Bi) is executed                       */
  /*******************************************************/
  LABEL(TAIL_LOOP)

  SUB(IMM(1), RSI)
  SUBITER(0)
  LEA(MEM(RAX, 12 * 16), RAX)
  LEA(MEM(RBX, 4 * 16), RBX)

  JNZ(TAIL_LOOP)

  LABEL(POSTACCUM)

  /**************************************************/
  /* Permute imag component register. Shuffle even  */
  /* and odd components                             */
  /* SRC: ZMM8 =(Ar0*Bi0, Ai0*Bi0, Ar1*Bi0, Ai1*Bi0)*/
  /* DST: ZMM8 =(Ai0*Bi0, Ar0*Bi0, Ai1*Bi0, Ar1*Bi0)*/
  /**************************************************/
  VPERMILPD(IMM(0x55), ZMM8, ZMM8)
  VPERMILPD(IMM(0x55), ZMM9, ZMM9)
  VPERMILPD(IMM(0x55), ZMM10, ZMM10)
  VPERMILPD(IMM(0x55), ZMM14, ZMM14)
  VPERMILPD(IMM(0x55), ZMM15, ZMM15)
  VPERMILPD(IMM(0x55), ZMM16, ZMM16)
  VPERMILPD(IMM(0x55), ZMM20, ZMM20)
  VPERMILPD(IMM(0x55), ZMM21, ZMM21)
  VPERMILPD(IMM(0x55), ZMM22, ZMM22)
  VPERMILPD(IMM(0x55), ZMM26, ZMM26)
  VPERMILPD(IMM(0x55), ZMM27, ZMM27)
  VPERMILPD(IMM(0x55), ZMM28, ZMM28)

  MOV(VAR(offsetPtr), R14)
  VMOVAPD(MEM(R14), ZMM(0))
  /***************************************************/
  /* SRC: ZMM5 = (Ar0*Br0, Ai0*Br0, Ar1*Br0, Ai1*Br0)*/
  /* SRC: ZMM8 = (Ai0*Bi0, Ar0*Bi0, Ai1*Bi0, Ar1*Bi0)*/
  /* DST: ZMM5 =(Ar0*Br0-Ai0*Bi0, Ai0*Br0+Ar0*Bi0,   */
  /*             Ar1*Br0-Ai1*Bi0, Ai1*Br0+Ar1*Bi0)   */
  /***************************************************/
  VFMADDSUB132PD(ZMM(0), ZMM(8), ZMM(5))
  VFMADDSUB132PD(ZMM(0), ZMM(9), ZMM(6))
  VFMADDSUB132PD(ZMM(0), ZMM(10), ZMM(7))
  VFMADDSUB132PD(ZMM(0), ZMM(14), ZMM(11))
  VFMADDSUB132PD(ZMM(0), ZMM(15), ZMM(12))
  VFMADDSUB132PD(ZMM(0), ZMM(16), ZMM(13))
  VFMADDSUB132PD(ZMM(0), ZMM(20), ZMM(17))
  VFMADDSUB132PD(ZMM(0), ZMM(21), ZMM(18))
  VFMADDSUB132PD(ZMM(0), ZMM(22), ZMM(19))
  VFMADDSUB132PD(ZMM(0), ZMM(26), ZMM(23))
  VFMADDSUB132PD(ZMM(0), ZMM(27), ZMM(24))
  VFMADDSUB132PD(ZMM(0), ZMM(28), ZMM(25))

  LABEL(STORE)
  MOV(VAR(offsetPtr), RDI)
  VMOVAPD(MEM(RDI), ZMM(4))
  /*Load alpha and beta values*/
  MOV(VAR(alpha), RAX)
  VBROADCASTSD(MEM(RAX, 0), ZMM(0))
  VBROADCASTSD(MEM(RAX, 8), ZMM(1))
  MOV(VAR(beta), RBX)
  VBROADCASTSD(MEM(RBX, 0), ZMM(2))
  VBROADCASTSD(MEM(RBX, 8), ZMM(3))
   /************************************************/
  /* C = (betaR, betaI)*(C)+(alphaR, alphaI)*(A*B) */
  /* ALPHA_SCALE: C = CInter1 + CInter2            */
  /* When alphaI=0                                 */
  /* ALPHA_ZERO:     alphaR=0  => CInter2 = 0      */
  /* ALPHA_REAL_ONE: alphaR=1  => CInter2 = A*B    */
  /* ALPHA_MINUS_ONE:alphaR=-1 => CInter2 = -A*B   */
  /*************************************************/
  MOV(VAR(alpha_mul_type), R14)

  CMP(IMM(1), R14) // Check if alpha = 1.0
  JE(ALPHA_SCALE_DONE)

  CMP(IMM(0), R14) // Check if alpha = 0.0
  JE(ALPHA_ZERO)

  LABEL(ALPHA_SCALE)
  CMP(IMM(2), R14) // Check for BLIS_MUL_DEFAULT

  JNE(ALPHA_MINUS_ONE)
  SCALE3R_BY_ALPHA(5, 6, 7)
  SCALE3R_BY_ALPHA(11, 12, 13)
  SCALE3R_BY_ALPHA(17, 18, 19)
  SCALE3R_BY_ALPHA(23, 24, 25)
  JMP(ALPHA_SCALE_DONE)

  LABEL(ALPHA_MINUS_ONE)
  VXORPD(ZMM8, ZMM8, ZMM8)
  VSUBPD(ZMM(5), ZMM(8), ZMM(5))
  VSUBPD(ZMM(6), ZMM(8), ZMM(6))
  VSUBPD(ZMM(7), ZMM(8), ZMM(7))
  VSUBPD(ZMM(11), ZMM(8), ZMM(11))
  VSUBPD(ZMM(12), ZMM(8), ZMM(12))
  VSUBPD(ZMM(13), ZMM(8), ZMM(13))
  VSUBPD(ZMM(17), ZMM(8), ZMM(17))
  VSUBPD(ZMM(18), ZMM(8), ZMM(18))
  VSUBPD(ZMM(19), ZMM(8), ZMM(19))
  VSUBPD(ZMM(23), ZMM(8), ZMM(23))
  VSUBPD(ZMM(24), ZMM(8), ZMM(24))
  VSUBPD(ZMM(25), ZMM(8), ZMM(25))
  JMP(ALPHA_SCALE_DONE)

  LABEL(ALPHA_ZERO)
  SET_REG_TO_ZERO(5, 6, 7)
  SET_REG_TO_ZERO(11, 12, 13)
  SET_REG_TO_ZERO(17, 18, 19)
  SET_REG_TO_ZERO(23, 24, 25)

  LABEL(ALPHA_SCALE_DONE)
  MOV(VAR(rs_c), RDI)
  LEA(MEM(RDI, RDI, 2), R12)
  MOV(VAR(cs_c), RSI)

  CMP(IMM(16), RDI) // Check if C is column stored

  JNZ(ROWSTORED) // Jump to row stored
  /************************************************/
  /* C = (betaR, betaI)*(C)+(alphaR, alphaI)*(A*B)*/
  /* BETA_SCALE : C = CInter1 + CInter2           */
  /* When betaI = 0                               */
  /* BETAZERO:    betaR=0  => CInter1 = 0         */
  /* BETA_ONE:    betaR=1  => CInter1 = C         */
  /* BETA_MINUS1: betaR=-1 => CInter1 = -C        */
  /************************************************/
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if betaR = 0.0
  JE(BETAZERO)

  CMP(IMM(1), R14)
  JE(BETA_ONE) // Check if betaR = 1.0

  CMP(IMM(2), R14) // Check for betaR = AnyValue(It should not be 0,1,-1)
  JE(BETA_SCALE)

  LABEL(BETA_MINUS1)
  UPDATE_C_BETAMINUS1(5, 6, 7)
  UPDATE_C_BETAMINUS1(11, 12, 13)
  UPDATE_C_BETAMINUS1(17, 18, 19)
  UPDATE_C_BETAMINUS1(23, 24, 25)
  JMP(END)

  LABEL(BETA_ONE)
  UPDATE_C_BETA1(5, 6, 7)
  UPDATE_C_BETA1(11, 12, 13)
  UPDATE_C_BETA1(17, 18, 19)
  UPDATE_C_BETA1(23, 24, 25)
  JMP(END)

  LABEL(BETA_SCALE)
  UPDATE_C_BETASCALE(5, 6, 7)
  UPDATE_C_BETASCALE(11, 12, 13)
  UPDATE_C_BETASCALE(17, 18, 19)
  UPDATE_C_BETASCALE(23, 24, 25)
  JMP(END)

  LABEL(BETAZERO)
  STORE_C(5, 6, 7)
  STORE_C(11, 12, 13)
  STORE_C(17, 18, 19)
  STORE_C(23, 24, 25)
  JMP(END)

  LABEL(ROWSTORED)
  CMP(IMM(16), RSI) // Check if C is row stored
  JNZ(GENSTORED) // Jump to gen stored
  MOV(VAR(cs_c), RDI)
  MOV(VAR(rs_c), RSI)
  LEA(MEM(RDI, RDI, 2), R12)    // r12 =  3*rs_c;

  TRANSPOSE(5, 11, 17, 23)
  TRANSPOSE(6, 12, 18, 24)
  TRANSPOSE(7, 13, 19, 25)

  /************************************************/
  /* C = (betaR, betaI)*(C)+(alphaR, alphaI)*(A*B)*/
  /* BETA_SCALE : C = CInter1 + CInter2           */
  /* When betaI = 0                               */
  /* BETAZERO:    betaR=0  => CInter1 = 0         */
  /* BETA_ONE:    betaR=1  => CInter1 = C         */
  /* BETA_MINUS1: betaR=-1 => CInter1 = -C        */
  /************************************************/
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if betaR = 0.0
  JE(BETAZERO_ROW)

  CMP(IMM(1), R14)
  JE(BETA_ONE_ROW) // Check if betaR = 1.0

  CMP(IMM(2), R14) // Check for betaR = AnyValue(It should not be 0,1,-1)
  JE(BETA_SCALE_ROW)

  LABEL(BETA_MINUS1_ROW)
  UPDATE_C_BETAMINUS1_ROW(5, 11, 17, 23)
  UPDATE_C_BETAMINUS1_ROW(6, 12, 18, 24)
  UPDATE_C_BETAMINUS1_ROW(7, 13, 19, 25)
  JMP(END)

  LABEL(BETA_ONE_ROW)
  UPDATE_C_BETA1_ROW(5, 11, 17, 23)
  UPDATE_C_BETA1_ROW(6, 12, 18, 24)
  UPDATE_C_BETA1_ROW(7, 13, 19, 25)
  JMP(END)

  LABEL(BETA_SCALE_ROW)
  UPDATE_C_BETASCALE_ROW(5, 11, 17, 23)
  UPDATE_C_BETASCALE_ROW(6, 12, 18, 24)
  UPDATE_C_BETASCALE_ROW(7, 13, 19, 25)
  JMP(END)

  LABEL(BETAZERO_ROW)
  STORE_C_ROW(5, 11, 17, 23)
  STORE_C_ROW(6, 12, 18, 24)
  STORE_C_ROW(7, 13, 19, 25)
  JMP(END)

  LABEL(GENSTORED)
  MOV(VAR(rs_c), RSI)
  MOV(VAR(cs_c), RDI)
  LEA(MEM(RDI, RDI, 2), R12)

  TRANSPOSE(5, 11, 17, 23)
  TRANSPOSE(6, 12, 18, 24)
  TRANSPOSE(7, 13, 19, 25)

  /************************************************/
  /* C = (betaR, betaI)*(C)+(alphaR, alphaI)*(A*B)*/
  /* BETA_SCALE : C = CInter1 + CInter2           */
  /* When betaI = 0                               */
  /* BETAZERO:    betaR=0  => CInter1 = 0         */
  /* BETA_ONE:    betaR=1  => CInter1 = C         */
  /* BETA_MINUS1: betaR=-1 => CInter1 = -C        */
  /************************************************/
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if betaR = 0.0
  JE(BETAZERO_GEN)
  CMP(IMM(2), R14) // Check for betaR = AnyValue(It should not be 0,1,-1)
  JE(BETA_SCALE_GEN)
  CMP(IMM(1), R14)
  JE(BETA_ONE_GEN) // Check if betaR = 1.0

  LABEL(BETA_MINUS1_GEN)
  UPDATE_C_BETAMINUS1_GEN(5, 11, 17, 23)
  UPDATE_C_BETAMINUS1_GEN(6, 12, 18, 24)
  UPDATE_C_BETAMINUS1_GEN(7, 13, 19, 25)
  JMP(END)

  LABEL(BETA_ONE_GEN)
  UPDATE_C_BETA1_GEN(5, 11, 17, 23)
  UPDATE_C_BETA1_GEN(6, 12, 18, 24)
  UPDATE_C_BETA1_GEN(7, 13, 19, 25)
  JMP(END)

  LABEL(BETA_SCALE_GEN)
  UPDATE_C_BETASCALE_GEN(5, 11, 17, 23)
  UPDATE_C_BETASCALE_GEN(6, 12, 18, 24)
  UPDATE_C_BETASCALE_GEN(7, 13, 19, 25)
  JMP(END)

  LABEL(BETAZERO_GEN)
  EXTRACT_STORE_C_GEN(5, 11, 17, 23)
  EXTRACT_STORE_C_GEN(6, 12, 18, 24)
  EXTRACT_STORE_C_GEN(7, 13, 19, 25)
  JMP(END)

  LABEL(END)
  VZEROUPPER()

  end_asm(
    :                 // output operands (none)
    :                 // input operands
    [a] "m"(a),       // 1
    [k] "m"(k),       // 2
    [b] "m"(b),       // 3
    [c] "m"(c),       // 8
    [rs_c] "m"(rs_c), // 9
    [cs_c] "m"(cs_c), // 10,
    [alpha] "m"(alpha),
    [beta] "m"(beta),
    [offsetPtr] "m"(offsetPtr),
    [alpha_mul_type] "m"(alpha_mul_type),
    [beta_mul_type] "m"(beta_mul_type)
    : // register clobber list
    "rax", "rbx", "rcx", "rdi", "rsi", "r9", "r10", "r12", "r14",
    "xmm8", "xmm9", "xmm10",
    "ymm8", "ymm9",
    "zmm0", "zmm1", "zmm2",
    "zmm3", "zmm4", "zmm5", "zmm6",  "zmm7", "zmm8",
    "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14",
    "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20",
    "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
    "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory"
  )

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
