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
#include "bli_cgemm_zen4_asm_macros.h"

#define LOOP_ALIGN ALIGN32

/* Minimum number of iterations required for prefetching C */
#define TAIL_ITER 6

// This array is used to support ADDSUB instruction.
static float fma_vec[16] __attribute__((aligned(64)))
                                 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// This is an array used for the scatter/gather instructions.
static int64_t offsets[24] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

/*
    Register usage :
    ZMM(0)  - ZMM(2)   : Load A
    ZMM(28) - ZMM(31)  : Bdcst B
    ZMM(3)  - ZMM(14)  : Accumulate real_scaling
    ZMM(15) - ZMM(26)  : Accumulate imag_scaling / Load C

    Total registers used : 31
*/
void bli_cgemm_zen4_asm_24x4(
    dim_t k0,
    scomplex *restrict alpha,
    scomplex *restrict a,
    scomplex *restrict b,
    scomplex *restrict beta,
    scomplex *restrict c, inc_t rs_c0, inc_t cs_c0,
    auxinfo_t *data,
    cntx_t *restrict cntx
  )
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

  /* Casting all the integers to the same type */
  const uint64_t k = k0;
  uint64_t rs_c = rs_c0 * 8; // rs_c0 = rs_c * 8(size of scomplex datatype)
  uint64_t cs_c = cs_c0 * 8; // cs_c0 = cs_c * 8(size of scomplex datatype)

  /* Storing the address of the fma_vec array, to be used in the computation */
  const float *fmaPtr = &fma_vec[0];
  /* Storing the address of offsets, to be used for general-stride computation */
  const int64_t *offsetPtr = &offsets[0];

  /* Determining the alpha and beta multiplication types */
  /* This is done as part of optimizing the alpha and beta scaling */
  uint64_t alpha_mul_type = BLIS_MUL_DEFAULT;
  uint64_t beta_mul_type = BLIS_MUL_DEFAULT;

  /* Setting alpha_mul_type and bet_mul_type, based on special cases
     of alpha and beta. */
  if ( alpha->imag == 0.0 )
  {
    if ( alpha->real == 1.0 )
      alpha_mul_type = BLIS_MUL_ONE;
    else if ( alpha->real == -1.0 )
      alpha_mul_type = BLIS_MUL_MINUS_ONE;
  }

  if ( beta->imag == 0.0 )
  {
    if ( beta->real == 1.0 )
      beta_mul_type = BLIS_MUL_ONE ;
    else if ( beta->real == -1.0 )
      beta_mul_type = BLIS_MUL_MINUS_ONE;
    else if ( beta->real == 0.0 )
      beta_mul_type = BLIS_MUL_ZERO;
  }

  /* Start of the assembly code-section */
  BEGIN_ASM()

  /* Setting the registers to zero */
  SET_ZERO()

  /* Loading the value of k in RSI */
  MOV(VAR(k), RSI)

  /* Loading the addresses of A, B and C */
  MOV(VAR(a), RAX) // RAX = addr of A
  MOV(VAR(b), RBX) // RBX = addr of B
  MOV(VAR(c), RCX) // RCX = addr of C

  /* Load R9 with address of C to be used during prefetch */
  MOV(RCX, R9)

  /* Loading column-stride of C, since this is a column major kernel */
  /* NOTE : cs_c has already been scaled by the size of the datatype */
  MOV(VAR(cs_c), R10) // R10 = cs_c = 8 * cs_c0

  /* Unrolling by a factor of 4, along k-dimension */
  MOV(RSI, RDI)
  AND(IMM(3), RSI) // RSI = k % 4(k_fringe)
  SAR(IMM(2), RDI) // RDI = k / 4(k_iter)

  /* The k-loop is divided into 4 parts, to have a fixed distance for prefetching C */
  /* The k-loop is divided as follows :
      1. .CK_BEFORE_PREFETCH : k/4 - 4 - TAIL_ITER(before C prefetch)
      2. .CK_PREFETCH        : 4(prefetches C)
      3. .CK_AFTER_PREFETCH  : TAIL_ITER(after C prefetch(prefetch distance))
      4. .CK_FRINGE
  */

  LABEL(.CK_BEFORE_PREFETCH)
  /* Check for entering the k-loop(before prefetch) */
  SUB(IMM(4 + TAIL_ITER), RDI)
  /* Jump to k-loop(prefetch) if k/4 <= 4 + TAIL_ITER */
  JLE(.CK_PREFETCH)

  /* K-loop(unrolled) to perform A*B */
  LOOP_ALIGN
  LABEL(.CK_ITER_BP) // Unrolled iteration (B)efore (P)refetching

  /* Performing rank-1 update 4 times(based on unroll) */
  SUB_ITER_24x4(0, RAX, RBX)
  SUB_ITER_24x4(1, RAX, RBX)
  SUB(IMM(1), RDI) // RDI(iterator) -= 1
  SUB_ITER_24x4(2, RAX, RBX)
  SUB_ITER_24x4(3, RAX, RBX)

  /* Adjusting the addresses of A and B for the next set */
  LEA(MEM(RAX, 4 * 24 * 8), RAX) // RAX = RAX + 4 * MR * 8(loads)
  LEA(MEM(RBX, 4 * 4 * 8), RBX) // RBX = RBX + 4 * NR * 8(broascasts)

  /* Jump if RDI != 0 */
  JNZ(.CK_ITER_BP)

  LABEL(.CK_PREFETCH)
  /* Check for entering the k-loop(for C prefetch) */
  ADD(IMM(4), RDI)
  /* Jump to k-loop(after prefetch) if k/4 <= TAIL_ITER */
  JLE(.CK_AFTER_PREFETCH)

  /* K-loop(unrolled) to perform A*B */
  LOOP_ALIGN
  LABEL(.CK_ITER_P) // Unrolled iteration with (P)refetching

  /* Performing rank-1 update 4 times(based on unroll) */
  /* Also prefetch C */
  PREFETCHW0(MEM(R9))
  SUB_ITER_24x4(0, RAX, RBX)
  PREFETCHW0(MEM(R9, 64))
  SUB_ITER_24x4(1, RAX, RBX)
  PREFETCHW0(MEM(R9, 128))
  SUB(IMM(1), RDI) // RDI(iterator) -= 1
  SUB_ITER_24x4(2, RAX, RBX)
  SUB_ITER_24x4(3, RAX, RBX)

  /* Adjusting the addresses of A and B for the next set */
  LEA(MEM(RAX, 4 * 24 * 8), RAX) // RAX = RAX + 4 * MR * 8(loads)
  LEA(MEM(RBX, 4 * 4 * 8), RBX) // RBX = RBX + 4 * NR * 8(broascasts)
  LEA(MEM(R9, R10, 1), R9) // RCX = RCX + cs_c

  /* Jump if RDI != 0 */
  JNZ(.CK_ITER_P)

  LABEL(.CK_AFTER_PREFETCH)
  /* Check for entering the k-loop(for C prefetch) */
  ADD(IMM(0 + TAIL_ITER), RDI)
  /* Jump to k-loop(after prefetch) if k/4 <= 0 */
  JLE(.CK_FRINGE)

  /* K-loop(unrolled) to perform A*B */
  LOOP_ALIGN
  LABEL(.CK_ITER_AP) // Unrolled iteration (A)fter (P)refetching

  /* Performing rank-1 update 4 times(based on unroll) */
  /* Also prefetch C */
  SUB_ITER_24x4(0, RAX, RBX)
  SUB_ITER_24x4(1, RAX, RBX)
  SUB(IMM(1), RDI) // RDI(iterator) -= 1
  SUB_ITER_24x4(2, RAX, RBX)
  SUB_ITER_24x4(3, RAX, RBX)

  /* Adjusting the addresses of A and B for the next set */
  LEA(MEM(RAX, 4 * 24 * 8), RAX) // RAX = RAX + 4 * MR * 8(loads)
  LEA(MEM(RBX, 4 * 4 * 8), RBX) // RBX = RBX + 4 * NR * 8(broascasts)

  /* Jump if RDI != 0 */
  JNZ(.CK_ITER_AP)

  LABEL(.CK_FRINGE)
  /* Check for entering the k-loop(fringe) */
  TEST(RSI, RSI)
  JE(.POSTACCUM)

  /* K-loop(unrolled) to perform A*B */
  LOOP_ALIGN
  LABEL(.CK_ITER_FRINGE)

  /* Performing rank-1 update */
  SUB_ITER_24x4(0, RAX, RBX)
  SUB(IMM(1), RSI) // RSI(iterator) -= 1

  /* Adjusting the addresses of A and B for the next set */
  /* new_addr = old_addr  + ( unroll_factor * {MR or NR} * size_of_type ) */
  LEA(MEM(RAX, 24 * 8), RAX) // RAX = RAX + MR * 8(loads)
  LEA(MEM(RBX, 4 * 8), RBX) // RBX = RBX + NR * 8(broascasts)

  /* Jump until RSI becomes 0 */
  JNZ(.CK_ITER_FRINGE)

  LABEL(.POSTACCUM)

  /* The registers from ZMM(15) to ZMM(26) contain the FMA ops
     using imaginary components from elements in B matrix.
     We should shuffle them( even and odd indices )
     SRC: ZMM(15) = ( Ar0*Bi0, Ai0*Bi0, Ar1*Bi0, Ai1*Bi0, ... )
     DST: ZMM(15) = ( Ai0*Bi0, Ar0*Bi0, Ai1*Bi0, Ar1*Bi0, ... )
     Similary for the other registers
  */
  PERMUTE(15, 16, 17)
  PERMUTE(18, 19, 20)
  PERMUTE(21, 22, 23)
  PERMUTE(24, 25, 26)

  /* Loading ZMM(0) with 1.0f, for reduction */
  MOV(VAR(fmaPtr), R14)
  VMOVAPS(MEM(R14), ZMM(0))

  /* Reducing the result using real/imag accumulators, for complex arithmetic
     SRC: ZMM(3)  = ( Ar0*Br0, Ai0*Br0, Ar1*Br0, Ai1*Br0, ... )
          ZMM(15) = ( Ai0*Bi0, Ar0*Bi0, Ai1*Bi0, Ar1*Bi0, ... )
     DST: ZMM(3)  = ( Ar0*Br0 - Ai0*Bi0, Ai0*Br0 + Ar0*Bi0, ...  )
     Similarly done for the other registers
  */
  FMADDSUB(3, 15, 4, 16, 5, 17)
  FMADDSUB(6, 18, 7, 19, 8, 20)
  FMADDSUB(9, 21, 10, 22, 11, 23)
  FMADDSUB(12, 24, 13, 25, 14, 26)

  /*
    The result of A*B(micro-tile) is a 24x4 matrix(column-major), as follows :

                Column-1  Column-2  Column-3   Column-4
    Rows(1-8)    ZMM(3)    ZMM(6)    ZMM(9)     ZMM(12)
    Rows(9-16)   ZMM(4)    ZMM(7)    ZMM(10)    ZMM(13)
    Rows(17-24)  ZMM(5)    ZMM(8)    ZMM(11)    ZMM(14)

  */

  LABEL(.ALPHA_SCALING)
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
  MOV(VAR(alpha_mul_type), R14)
  CMP(IMM(1), R14) // Check if alpha = 1.0
  /* Skip alpha scaling and jump to beta scaling */
  JE(.BETA_SCALING)

  CMP(IMM(2), R14) // Check if alpha != -1.0
  /* Jump to the general case of alpha scaling */
  JE(.ALPHA_GENERAL)

  /* Alpha scaling when alpha == -1.0 */
  LABEL(.ALPHA_MINUS_ONE)
  /* Set ZMM(1) to 0.0f, and subtract the registers from ZMM(1) */
  VXORPS(ZMM(1), ZMM(1), ZMM(1))
  /* ZMM(3) = ZMM(1) - ZMM(3) = 0.0f - A*B
     Similarly done for other registers */
  ALPHA_MINUS_ONE(3, 1, 4, 1, 5, 1)
  ALPHA_MINUS_ONE(6, 1, 7, 1, 8, 1)
  ALPHA_MINUS_ONE(9, 1, 10, 1, 11, 1)
  ALPHA_MINUS_ONE(12, 1, 13, 1, 14, 1)

  /* Jump to beta scaling */
  JMP(.BETA_SCALING)

  /* Alpha scaling when alpha != 1.0 and alpha != -1.0 */
  LABEL(.ALPHA_GENERAL)
  /* Load alpha onto a ZMM register */
  MOV(VAR(alpha), RAX)
  /* Broadcast the real and imag components of alpha onto the registers */
  VBROADCASTSS(MEM(RAX, 0), ZMM(1))
  VBROADCASTSS(MEM(RAX, 4), ZMM(2))

  /* Scale the result of A*B with alpha */
  /* ZMM(15) = alphai * ZMM(3)
     ZMM(3)  = alphar * ZMM(3)
     ZMM(3)  = fmaddsub(ZMM(3), permute(ZMM(15)))

     Similarly done for other pairs of registers */
  ALPHA_DEFAULT(3, 15, 4, 16, 5, 17)
  ALPHA_DEFAULT(6, 18, 7, 19, 8, 20)
  ALPHA_DEFAULT(9, 21, 10, 22, 11, 23)
  ALPHA_DEFAULT(12, 24, 13, 25, 14, 26)

  /* Perform beta scaling */
  LABEL(.BETA_SCALING)
  /* Load the row and column strides of C */
  MOV(VAR(rs_c), RDI)
  MOV(VAR(cs_c), RSI)

  CMP(IMM(8), RDI) // Check if C is column stored

  JNE(.ROWSTORED) // Jump to row stored

  LABEL(.COLSTORED)
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
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if beta = 0.0
  /* Skip beta scaling and jump to store */
  JE(.BETA_ZERO_COL)

  CMP(IMM(1), R14) // Check if beta = 1.0
  /* Jump to beta = 1.0 case */
  JE(.BETA_ONE_COL)

  CMP(IMM(2), R14) // Check if alpha != -1.0
  /* Jump to the general case of alpha scaling */
  JE(.BETA_DEFAULT_COL)

  /* Beta scaling when beta == -1.0 */
  LABEL(.BETA_MINUS_ONE_COL)

  /* Perform C = alpha*A*B - C */
  /* ZMM(15) = load(C)
     ZMM(15)  = ZMM(3) - ZMM(15) = alpha*A*B - C
     store(ZMM(15))

     Similarly done for other registers */
  BETA_MINUS_ONE_PRIMARY(3, 15, 4, 16, 5, 17)
  LEA((RCX, R10, 1), RCX)
  BETA_MINUS_ONE_PRIMARY(6, 18, 7, 19, 8, 20)
  LEA((RCX, R10, 1), RCX)
  BETA_MINUS_ONE_PRIMARY(9, 21, 10, 22, 11, 23)
  LEA((RCX, R10, 1), RCX)
  BETA_MINUS_ONE_PRIMARY(12, 24, 13, 25, 14, 26)

  JMP(.END)

  /* Beta scaling when beta == -1.0 */
  LABEL(.BETA_ONE_COL)

  /* Perform C = C + alpha*A*B */
  /* ZMM(15) = load(C)
     ZMM(15)  = ZMM(3) + ZMM(15) = alpha*A*B + C
     store(ZMM(15))

     Similarly done for other registers */
  BETA_ONE_PRIMARY(3, 15, 4, 16, 5, 17)
  LEA((RCX, R10, 1), RCX)
  BETA_ONE_PRIMARY(6, 18, 7, 19, 8, 20)
  LEA((RCX, R10, 1), RCX)
  BETA_ONE_PRIMARY(9, 21, 10, 22, 11, 23)
  LEA((RCX, R10, 1), RCX)
  BETA_ONE_PRIMARY(12, 24, 13, 25, 14, 26)

  JMP(.END)

  /* Beta scaling for generic case */
  LABEL(.BETA_DEFAULT_COL)
  /* Load beta onto a ZMM register */
  MOV(VAR(beta), RBX)
  /* Broadcast the real and imag components of beta onto the registers */
  VBROADCASTSS(MEM(RBX, 0), ZMM(1))
  VBROADCASTSS(MEM(RBX, 4), ZMM(2))

  /* Perform C = beta*C + alpha*A*B */
  /* ZMM(15) = load(C)
     Perform beta scaling of ZMM(15)(similar to alpha scaling)
     ZMM(15) = ZMM(3) + ZMM(15) = alpha*A*B + beta*C
     store(ZMM(15))

     Similarly done for other pairs of registers */
  BETA_DEFAULT_PRIMARY(3, 15, 4, 16, 5, 17)
  LEA((RCX, R10, 1), RCX)
  BETA_DEFAULT_PRIMARY(6, 18, 7, 19, 8, 20)
  LEA((RCX, R10, 1), RCX)
  BETA_DEFAULT_PRIMARY(9, 21, 10, 22, 11, 23)
  LEA((RCX, R10, 1), RCX)
  BETA_DEFAULT_PRIMARY(12, 24, 13, 25, 14, 26)

  JMP(.END)

  LABEL(.BETA_ZERO_COL)
  /* This code-section is taken if we want to skip scaling */
  VMOVUPS(ZMM(3), MEM(RCX))
  VMOVUPS(ZMM(4), MEM(RCX, 64))
  VMOVUPS(ZMM(5), MEM(RCX, 128))
  LEA((RCX, R10, 1), RCX)

  VMOVUPS(ZMM(6), MEM(RCX))
  VMOVUPS(ZMM(7), MEM(RCX, 64))
  VMOVUPS(ZMM(8), MEM(RCX, 128))
  LEA((RCX, R10, 1), RCX)

  VMOVUPS(ZMM(9), MEM(RCX))
  VMOVUPS(ZMM(10), MEM(RCX, 64))
  VMOVUPS(ZMM(11), MEM(RCX, 128))
  LEA((RCX, R10, 1), RCX)

  VMOVUPS(ZMM(12), MEM(RCX))
  VMOVUPS(ZMM(13), MEM(RCX, 64))
  VMOVUPS(ZMM(14), MEM(RCX, 128))

  JMP(.END)

  LABEL(.ROWSTORED)
  /* Check for general stride of C */
  CMP(IMM(8), RSI) // Check if C is row stored
  JNE(.GENERALSTRIDE) // Jump to general stride

  /* This code-section is taken if C is row-stored */
  /*
    Check for beta_mul_type, to jump to the required code-section
     Intermediate C = beta*C + IR, where IR = alpha*A*B
     If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
      C = IR, skip beta-scaling
     else => BLIS_MUL_DEFAULT
      C = beta*C + IR, using complex multiplication
  */
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if beta = 0.0
  /* Skip beta scaling and jump to store */
  JE(.BETA_ZERO_ROW)

  LABEL(.BETA_DEFAULT_ROW)
  /* Load beta onto a ZMM register */
  MOV(VAR(beta), RBX)
  /* Broadcast the real and imag components of beta onto the registers */
  VBROADCASTSS(MEM(RBX, 0), ZMM(1))
  VBROADCASTSS(MEM(RBX, 4), ZMM(2))

  /* We need to transpose the 24x4 block of alpha*A*B,
     in steps of 8x4.
     We use an 8x8 transpose routine with additional
     registers.

     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(3)   ZMM(6)   ZMM(9)   ZMM(12)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(3, 6, 9, 12, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Scale C by beta and compute the result */
  /* This is done one row at a time */
  BETA_DEFAULT_SECONDARY(3, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(6, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(9, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(12, 21, 22)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(28, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(29, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(30, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(31, 21, 22)
  LEA((RCX, RDI, 1), RCX)

  /*
     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-15)    ZMM(4)   ZMM(7)  ZMM(10)  ZMM(13)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(4, 7, 10, 13, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Scale C by beta and compute the result */
  /* This is done one row at a time */
  BETA_DEFAULT_SECONDARY(4, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(7, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(10, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(13, 21, 22)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(28, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(29, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(30, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(31, 21, 22)
  LEA((RCX, RDI, 1), RCX)

  /*
     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(16-23)  ZMM(5)   ZMM(8)   ZMM(11)  ZMM(14)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(5, 8, 11, 14, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Scale C by beta and compute the result */
  /* This is done one row at a time */
  BETA_DEFAULT_SECONDARY(5, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(8, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(11, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(14, 21, 22)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(28, 15, 16)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(29, 17, 18)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(30, 19, 20)
  LEA((RCX, RDI, 1), RCX)
  BETA_DEFAULT_SECONDARY(31, 21, 22)

  JMP(.END)

  LABEL(.BETA_ZERO_ROW)
  /* We need to transpose the 24x4 block of alpha*A*B,
     in steps of 8x4.
     We use an 8x8 transpose routine with additional
     registers.

     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(1-8)    ZMM(3)   ZMM(6)   ZMM(9)   ZMM(12)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */
  /* Set 4 additional registers to 0.0 */
  VXORPS(ZMM(28), ZMM(28), ZMM(28))
  VXORPS(ZMM(29), ZMM(29), ZMM(29))
  VXORPS(ZMM(30), ZMM(30), ZMM(30))
  VXORPS(ZMM(31), ZMM(31), ZMM(31))

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(3, 6, 9, 12, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Store the result back to C */
  /* We need to store only the first 256-bit lane of the
     registers post transpose */
  VMOVUPS(YMM(3), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(6), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(9), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(12), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(28), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(29), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(30), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(31), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)

  /*
     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(9-15)    ZMM(4)   ZMM(7)  ZMM(10)  ZMM(13)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(4, 7, 10, 13, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Store the result back to C */
  /* We need to store only the first 256-bit lane of the
     registers post transpose */
  VMOVUPS(YMM(4), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(7), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(10), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(13), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(28), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(29), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(30), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(31), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)

  /*
     Input for transpose:
                  Column-1 Column-2 Column-3 Column-4 Column-5 Column-6 Column-7 Column-8
      Rows(16-23)  ZMM(5)   ZMM(8)   ZMM(11)  ZMM(14)  ZMM(28)  ZMM(29)  ZMM(30)  ZMM(31)
  */

  /* Transpose the 8x8 block of alpha*A*B */
  /* ZMM(15) to ZMM(22) are used as temporary registers
     for transpose operation */
  TRANSPOSE_8X8(5, 8, 11, 14, 28, 29, 30, 31,
                15, 16, 17, 18, 19, 20, 21, 22)

  /* Store the result back to C */
  /* We need to store only the first 256-bit lane of the
     registers post transpose */
  VMOVUPS(YMM(5), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(8), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(11), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(14), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(28), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(29), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(30), MEM(RCX))
  LEA((RCX, RDI, 1), RCX)
  VMOVUPS(YMM(31), MEM(RCX))

  JMP(.END)

  LABEL(.GENERALSTRIDE)
  /* This code-section is taken if C has general stride */
  /*
     In case of general strides for C, we need to load/store C
     using gather/scatter instructions.
     Visualizing C(8x4):
      ---------------------------------------------
      | C00       | C10       |  ...  | C30       |
      |  | (rs_c) |  | (rs_c) |  ...  |  | (rs_c) |
      | C01       | C11       |  ...  | C31       |
      |  | (rs_c) |  | (rs_c) |  ...  |  | (rs_c) |
      | C02       | C12       |  ...  | C32       |
      |  .        |  .        |  ...  |  .        |
      |  .        |  .        |  ...  |  .        |
      |  .        |  .        |  ...  |  .        |
      | C07       | C17       |  ...  | C37       |
      ---------------------------------------------

      Loading C :
      Gather all elements of C column-wise onto ZMM registers

      Compute with C(based on beta):
      Similar to column-stored case, perform beta scaling and add to
      alpha*A*B

      Storing C :
      Scatter the result one column at a time, using ZMM registers
  */
  MOV(VAR(offsetPtr), R9) // Load address of offsets
  VPBROADCASTQ(RDI, ZMM(31)) // Broadcast rs_c onto a register
  VPMULLQ(MEM(R9), ZMM(31), ZMM(28)) // ZMM28 = { 0*rs_c, 1*rs_c, 2*rs_c, 3*rs_c, ... }
  VPMULLQ(MEM(R9, 64), ZMM(31), ZMM(29)) // ZMM29 = { 8*rs_c, 9*rs_c, 10*rs_c, 11*rs_c, ... }
  VPMULLQ(MEM(R9, 128), ZMM(31), ZMM(30)) // ZMM30 = { 16*rs_c, 17*rs_c, 18*rs_c, 19*rs_c, ... }
  /*
    Check for beta_mul_type, to jump to the required code-section
     Intermediate C = beta*C + IR, where IR = alpha*A*B
     If beta == ( 0.0, 0.0 ) => BLIS_MUL_ZERO
      C = IR, skip beta-scaling
     else => BLIS_MUL_DEFAULT
      C = beta*C + IR, using complex multiplication
  */
  MOV(VAR(beta_mul_type), R14)
  CMP(IMM(0), R14) // Check if beta = 0.0
  /* Skip beta scaling and jump to store */
  JE(.BETA_ZERO_GENERIC)

  LABEL(.BETA_DEFAULT_GENERIC)
  /* Load beta onto a ZMM register */
  MOV(VAR(beta), RBX)
  /* Broadcast the real and imag components of beta onto the registers */
  VBROADCASTSS(MEM(RBX, 0), ZMM(1))
  VBROADCASTSS(MEM(RBX, 4), ZMM(2))

  /* Compute C = beta*C + alpha*A*B, and store to C */
  BETA_DEFAULT_GENERAL(3, 4, 5)
  LEA((RCX, RSI, 1), RCX)
  BETA_DEFAULT_GENERAL(6, 7, 8)
  LEA((RCX, RSI, 1), RCX)
  BETA_DEFAULT_GENERAL(9, 10, 11)
  LEA((RCX, RSI, 1), RCX)
  BETA_DEFAULT_GENERAL(12, 13, 14)

  JMP(.END)

  LABEL(.BETA_ZERO_GENERIC)
  /* Store the result onto C, one column at a time */
  BETA_ZERO_GENERAL(3, 4, 5)
  LEA((RCX, RSI, 1), RCX)
  BETA_ZERO_GENERAL(6, 7, 8)
  LEA((RCX, RSI, 1), RCX)
  BETA_ZERO_GENERAL(9, 10, 11)
  LEA((RCX, RSI, 1), RCX)
  BETA_ZERO_GENERAL(12, 13, 14)

  LABEL(.END)
  VZEROUPPER()

  end_asm(
    :                 // output operands (none)
    :                 // input operands
    [k] "m"(k),
    [a] "m"(a),
    [b] "m"(b),
    [c] "m"(c),
    [rs_c] "m"(rs_c),
    [cs_c] "m"(cs_c),
    [fmaPtr] "m"(fmaPtr),
    [offsetPtr] "m"(offsetPtr),
    [alpha_mul_type] "m"(alpha_mul_type),
    [beta_mul_type] "m"(beta_mul_type),
    [alpha] "m"(alpha),
    [beta] "m"(beta)
    : // register clobber list
    "rax", "rbx", "rcx", "rdi", "rsi", "r9", "r10", "r12", "r14",
    "k0", "k1", "k2", "k3", "k4",
    "ymm0", "ymm1", "ymm2", "ymm3",
    "ymm4", "ymm5", "ymm6", "ymm7",
    "ymm8", "ymm9", "ymm10", "ymm11",
    "ymm12", "ymm13", "ymm14", "ymm15",
    "ymm16", "ymm17", "ymm18", "ymm19",
    "ymm20", "ymm21", "ymm22", "ymm28",
    "ymm29", "ymm30", "ymm31",
    "zmm0", "zmm1", "zmm2",
    "zmm3", "zmm4", "zmm5", "zmm6",  "zmm7", "zmm8",
    "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14",
    "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20",
    "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
    "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory"
  )

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
