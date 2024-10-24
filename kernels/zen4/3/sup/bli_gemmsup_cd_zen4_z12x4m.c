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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

// Defining the micro-kernel dimenisions
#define MR 12
#define NR 4

// Macro for resetting the registers for accumulation
#define RESET_REGISTERS \
  VXORPD(ZMM(0), ZMM(0), ZMM(0))      \
  VXORPD(ZMM(1), ZMM(1), ZMM(1))      \
  VXORPD(ZMM(2), ZMM(2), ZMM(2))      \
  VXORPD(ZMM(3), ZMM(3), ZMM(3))      \
  VXORPD(ZMM(4), ZMM(4), ZMM(4))      \
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

// Macro for performing a 1x2 tile FMA based accumulation
#define FMA_1x2(A0, B0, B1) \
  /* Partial sums without permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(B0), ZMM(6)) \
  VPERMILPD(IMM(0x55), ZMM(B0), ZMM(30)) \
  VFMADD231PD(ZMM(A0), ZMM(B1), ZMM(14)) \
  VPERMILPD(IMM(0x55), ZMM(B1), ZMM(31)) \
  /* Partial sums after permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(30), ZMM(10)) \
  VFMADD231PD(ZMM(A0), ZMM(31), ZMM(18)) \

// Macro for performing a 2x2 tile FMA based accumulation
#define FMA_2x2(A0, A1, B0, B1) \
  /* Partial sums without permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(B0), ZMM(6)) \
  VFMADD231PD(ZMM(A1), ZMM(B0), ZMM(7)) \
  VPERMILPD(IMM(0x55), ZMM(B0), ZMM(30)) \
  VFMADD231PD(ZMM(A0), ZMM(B1), ZMM(14)) \
  VFMADD231PD(ZMM(A1), ZMM(B1), ZMM(15)) \
  VPERMILPD(IMM(0x55), ZMM(B1), ZMM(31)) \
  /* Partial sums after permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(30), ZMM(10)) \
  VFMADD231PD(ZMM(A1), ZMM(30), ZMM(11)) \
  VFMADD231PD(ZMM(A0), ZMM(31), ZMM(18)) \
  VFMADD231PD(ZMM(A1), ZMM(31), ZMM(19)) \

// Macro for performing a 4x2 tile FMA based accumulation
#define FMA_4x2(A0, A1, A2, A3, B0, B1) \
  /* Partial sums without permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(B0), ZMM(6)) \
  VFMADD231PD(ZMM(A1), ZMM(B0), ZMM(7)) \
  VFMADD231PD(ZMM(A2), ZMM(B0), ZMM(8)) \
  VFMADD231PD(ZMM(A3), ZMM(B0), ZMM(9)) \
  VPERMILPD(IMM(0x55), ZMM(B0), ZMM(30)) \
  VFMADD231PD(ZMM(A0), ZMM(B1), ZMM(14)) \
  VFMADD231PD(ZMM(A1), ZMM(B1), ZMM(15)) \
  VFMADD231PD(ZMM(A2), ZMM(B1), ZMM(16)) \
  VFMADD231PD(ZMM(A3), ZMM(B1), ZMM(17)) \
  VPERMILPD(IMM(0x55), ZMM(B1), ZMM(31)) \
  /* Partial sums after permuting B */ \
  VFMADD231PD(ZMM(A0), ZMM(30), ZMM(10)) \
  VFMADD231PD(ZMM(A1), ZMM(30), ZMM(11)) \
  VFMADD231PD(ZMM(A2), ZMM(30), ZMM(12)) \
  VFMADD231PD(ZMM(A3), ZMM(30), ZMM(13)) \
  VFMADD231PD(ZMM(A0), ZMM(31), ZMM(18)) \
  VFMADD231PD(ZMM(A1), ZMM(31), ZMM(19)) \
  VFMADD231PD(ZMM(A2), ZMM(31), ZMM(20)) \
  VFMADD231PD(ZMM(A3), ZMM(31), ZMM(21)) \

// A 4x2 micro tile computation using A(4 x k_fringe) and B(k_fringe x 2)
// The loads are masked since k is not a multiple of SIMD width(4)
#define MICRO_TILE_4x2_FRINGE(mask_reg_num) \
  /* Macro for 4x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5) MASK_KZ(mask_reg_num)) \
  /* Storing an additional offset for A */ \
  LEA(MEM(RAX, R13, 2), RDI) \
  /* Load 4 rows from A */ \
  VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RAX, R13, 1), ZMM(1) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RDI), ZMM(2) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RDI, R13, 1), ZMM(3) MASK_KZ(mask_reg_num)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_4x2(0, 1, 2, 3, 4, 5) \

// A 4x2 micro tile computation using A(4 x 4) and B(4 x 2)
// The loads are unmasked since k is a multiple of SIMD width(4)
#define MICRO_TILE_4x2_MAIN() \
  /* Macro for 4x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5)) \
  /* Storing an additional offset for A */ \
  LEA(MEM(RAX, R13, 2), RDI) \
  /* Load 4 rows from A */ \
  VMOVUPD(MEM(RAX), ZMM(0)) \
  VMOVUPD(MEM(RAX, R13, 1), ZMM(1)) \
  VMOVUPD(MEM(RDI), ZMM(2)) \
  VMOVUPD(MEM(RDI, R13, 1), ZMM(3)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_4x2(0, 1, 2, 3, 4, 5) \
\
  /* Adjusting addresses for next micro tiles */ \
  ADD(IMM(64), RAX) \
  ADD(IMM(64), RBX) \

// A 2x2 micro tile computation using A(2 x k_fringe) and B(k_fringe x 2)
// The loads are masked since k is not a multiple of SIMD width(4)
#define MICRO_TILE_2x2_FRINGE(mask_reg_num) \
  /* Macro for 2x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5) MASK_KZ(mask_reg_num)) \
  /* Load 2 rows from A */ \
  VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RAX, R13, 1), ZMM(1) MASK_KZ(mask_reg_num)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_2x2(0, 1, 4, 5) \

// A 2x2 micro tile computation using A(4 x k_fringe) and B(k_fringe x 2)
// The loads are unmasked since k is a multiple of SIMD width(4)
#define MICRO_TILE_2x2_MAIN() \
  /* Macro for 2x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5)) \
  /* Load 2 rows from A */ \
  VMOVUPD(MEM(RAX), ZMM(0)) \
  VMOVUPD(MEM(RAX, R13, 1), ZMM(1)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_2x2(0, 1, 4, 5) \
\
  /* Adjusting addresses for next micro tiles */ \
  ADD(IMM(64), RAX) \
  ADD(IMM(64), RBX) \

// A 1x2 micro tile computation using A(1 x k_fringe) and B(k_fringe x 2)
// The loads are masked since k is not a multiple of SIMD width(4)
#define MICRO_TILE_1x2_FRINGE(mask_reg_num) \
  /* Macro for 1x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4) MASK_KZ(mask_reg_num)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5) MASK_KZ(mask_reg_num)) \
  /* Load 1 row from A */ \
  VMOVUPD(MEM(RAX), ZMM(0) MASK_KZ(mask_reg_num)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_1x2(0, 4, 5) \

// A 1x2 micro tile computation using A(1 x k_fringe) and B(k_fringe x 2)
// The loads are masked since k is not a multiple of SIMD width(4)
#define MICRO_TILE_1x2_MAIN() \
  /* Macro for 1x2 micro-tile evaluation   */ \
  /* Loading 2 columns from B */ \
  VMOVUPD(MEM(RBX), ZMM(4)) \
  VMOVUPD(MEM(RBX, R14, 1), ZMM(5)) \
  /* Load 1 row from A */ \
  VMOVUPD(MEM(RAX), ZMM(0)) \
\
  /* Perform a set of FMAs for partial dot-products */ \
  FMA_1x2(0, 4, 5) \
\
  /* Adjusting addresses for next micro tiles */ \
  ADD(IMM(64), RAX) \
  ADD(IMM(64), RBX) \

// Macros to perform 128-bit lane shuffling
// R? represent the set of vectors having A(.)B
// I? represent the set of vectors having A(.)(perm(B))
// SHUFFLE_1R is when 1 row of A was loaded for computation
#define SHUFFLE_1R( \
                    R0, \
                    I0, \
                    T0 \
                  ) \
  /*
    If  R0 = AR1.BR1  AI1.BI1  AR2.BR2  AI2.BI2 ...
    and I0 = AR1.BI1  AI1.BR1  AR2.BI2  AI2.BI2 ...
    then post shuffle we have :
      T0 = AR1.BR1  AR1.BI1  AR2.BI2  AR2.BI2 ...
      I0 = AI1.BI1  AI1.BR1  AI2.BI2  AI2.BR2 ...
  */ \
  VSHUFPD(IMM(0x00), ZMM(I0), ZMM(R0), ZMM(T0)) \
\
  VSHUFPD(IMM(0xFF), ZMM(I0), ZMM(R0), ZMM(I0)) \

// SHUFFLE_2R is when 2 rows of A were loaded for computation
#define SHUFFLE_2R( \
                    R0, R1, \
                    I0, I1, \
                    T0, T1 \
                  ) \
  SHUFFLE_1R( \
              R0, \
              I0, \
              T0 \
             ) \
  VSHUFPD(IMM(0x00), ZMM(I1), ZMM(R1), ZMM(T1)) \
\
  VSHUFPD(IMM(0xFF), ZMM(I1), ZMM(R1), ZMM(I1)) \

// SHUFFLE_3R is when 3 rows of A were loaded for computation
// NOTE : This macro is currently not used
#define SHUFFLE_3R( \
                    R0, R1, R2, \
                    I0, I1, I2, \
                    T0, T1, T2 \
                  ) \
  SHUFFLE_2R( \
              R0, R1, \
              I0, I1, \
              T0, T1 \
             ) \
  VSHUFPD(IMM(0x00), ZMM(I2), ZMM(R2), ZMM(T2)) \
\
  VSHUFPD(IMM(0xFF), ZMM(I2), ZMM(R2), ZMM(I2)) \

// SHUFFLE_4R is when 4 rows of A were loaded for computation
#define SHUFFLE_4R( \
                    R0, R1, R2, R3, \
                    I0, I1, I2, I3, \
                    T0, T1, T2, T3 \
                  ) \
  SHUFFLE_3R( \
              R0, R1, R2, \
              I0, I1, I2, \
              T0, T1, T2 \
             ) \
  VSHUFPD(IMM(0x00), ZMM(I3), ZMM(R3), ZMM(T3)) \
\
  VSHUFPD(IMM(0xFF), ZMM(I3), ZMM(R3), ZMM(I3)) \

// Macros to perform the final accumulation of T? and I?
// T? contains A scaled with real components from B
// I? contains A scaled with imag components from B
// ACCUMULATE_1R is when 1 row of A was loaded for computation
#define ACCUMULATE_1R( \
                    I0, \
                    T0 \
                 ) \
  /*
    If  T0 = AR1.BR1  AR1.BI1  AR2.BI2  AR2.BI2 ...
    and I0 = AI1.BI1  AI1.BR1  AI2.BI2  AI2.BR2 ...
    then post shuffle we have :
      I0 = ( AR1.BR1 - AI1.BI1 )  ( AR1.BI1 + AI1.BR1 ) ...
  */ \
  VFMADDSUB231PD(ZMM(T0), ZMM(30), ZMM(I0)) \

// ACCUMULATE_2R is when 2 rows of A were loaded for computation
#define ACCUMULATE_2R( \
                    I0, I1, \
                    T0, T1 \
                 ) \
  ACCUMULATE_1R( \
              I0, \
              T0 \
             ) \
  VFMADDSUB231PD(ZMM(T1), ZMM(30), ZMM(I1)) \

// ACCUMULATE_3R is when 3 rows of A were loaded for computation
#define ACCUMULATE_3R( \
                    I0, I1, I2, \
                    T0, T1, T2 \
                 ) \
  ACCUMULATE_2R( \
              I0, I1, \
              T0, T1 \
             ) \
  VFMADDSUB231PD(ZMM(T2), ZMM(30), ZMM(I2)) \

// ACCUMULATE_4R is when 4 rows of A were loaded for computation
#define ACCUMULATE_4R( \
                    I0, I1, I2, I3, \
                    T0, T1, T2, T3 \
                 ) \
  ACCUMULATE_3R( \
              I0, I1, I2, \
              T0, T1, T2 \
             ) \
  VFMADDSUB231PD(ZMM(T3), ZMM(30), ZMM(I3)) \

// Transposing 4 ZMM vectors, each containing 4 complex numbers
// Thus, it is a 4x4 transpose
#define TRANSPOSE_4x4( \
                      R0, R1, R2, R3, \
                      T0, T1, T2, T3 \
                     ) \
  VSHUFF64X2(IMM(0x88), ZMM(R1), ZMM(R0), ZMM(26)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R1), ZMM(R0), ZMM(27)) \
  VSHUFF64X2(IMM(0x88), ZMM(R3), ZMM(R2), ZMM(28)) \
  VSHUFF64X2(IMM(0xDD), ZMM(R3), ZMM(R2), ZMM(29)) \
  VSHUFF64X2(IMM(0x88), ZMM(28), ZMM(26), ZMM(T0)) \
  VSHUFF64X2(IMM(0xDD), ZMM(28), ZMM(26), ZMM(T2)) \
  VSHUFF64X2(IMM(0x88), ZMM(29), ZMM(27), ZMM(T1)) \
  VSHUFF64X2(IMM(0xDD), ZMM(29), ZMM(27), ZMM(T3))

/*
   crc:
     | | | |         --------    | | | |
     | | | |   +=    --------    | | | | ...
     | | | |         --------    | | | |
     | | | |             :       | | | |

   Assumptions:
   - A is row stored;
   - B is column-stored;
   Therefore, this (c)olumn-preferential kernel is well-suited for contiguous
   (d)ot product with vector loads from A and B.

   NOTE: The rrc case of input storage scheme is handled by transposing the
         operation in the framework layer(i.e C^T = B^T x A^T, thus converting the
         operands into a crc storage scheme).
*/

void bli_zgemmsup_cd_zen4_asm_12x4m
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
    uint64_t n_left = n0 % NR;

    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 12x?m kernels, as needed.
    if ( n_left )
    {
      dcomplex*  cij = c;
      dcomplex*  bj  = b;
      dcomplex*  ai  = a;

      if ( 2 <= n_left )
      {
        const dim_t nr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_12x2m
        (
          conja, conjb, m0, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0, beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );

        cij += 2 * cs_c0;
        bj += 2 * cs_b0;
        n_left -= 2;
      }
      if ( 1 == n_left )
      {
        // Call to GEMV, which internally uses DOTXF kernels(var1)
        bli_zgemv_ex
        (
          BLIS_NO_TRANSPOSE, conjb, m0, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0,
          beta, cij, rs_c0, cntx, NULL
        );
      }
      return;
    }

    // Main kernel
    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    if ( m_iter == 0 ) goto consider_edge_cases;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    /*
        Register clobber list and usage(for GPRs) :
        BEGIN_ASM

            R10 - Base addr of A
            RDX - Base addr of B
            R12 - Base addr of C

            R13 - Row stride of A
            R14 - Col stride of B
            RSI - Col stride of C

            R15 - n_iter(in 2, till 4)
                RBX - Copy of RDX
                RCX - Copy of R12
                R11 - m_iter(in MR, till MC)
                    R9 - m_inner_iter(in 4, till MR)
                        RAX - Copy of R10
                        R8  - k_iter and k_left
                            RDI - Offsetting of A(micro-tile calculation)
                            RAX gets k-based update
                            RBX gets k-based update
                        R8 - Used for broadcasting 1.0
                        R10 - Offset by 4 rows using rs_a
                        RCX - Offset by 64 bytes(col-major)
                        R9 - +4, !=12
                    R11 - Decrement
                RDX - Offset by 2 cols using cs_b
                R12 - Offset by 2 cols using cs_c
                R10 - Copy base addr of A
                R15 - +2, !=4
        END_ASM
    */

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(IMM(0), R15)
    LABEL(.ZLOOP_2J)          // Iterating in blocks of 2, until NR
    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(VAR(m_iter), R11)     // Iterating in steps of MR, until MC(m var)
    LABEL(.ZMLOOP)
    MOV(IMM(0), R9)
    LABEL(.ZLOOP_4I)          // Iterating in steps of 4, until MR
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next 4xKC block of A and 4x2 block of C
    */
    LEA(MEM(R10, R13, 4), R10)
    ADD(IMM(64), RCX)

    ADD(IMM(4), R9)
    CMP(IMM(12), R9)
    JNE(.ZLOOP_4I)

    DEC(R11)
    JNE(.ZMLOOP)

    LEA(MEM(RDX, R14, 2), RDX)
    LEA(MEM(R12, RSI, 2), R12)
    MOV(VAR(a), R10)

    ADD(IMM(2), R15)
    CMP(IMM(4), R15)
    JNE(.ZLOOP_2J)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_left]  "m" (m_left),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
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
      dcomplex* restrict cij = c + m_iter * MR * rs_c0;
      dcomplex* restrict ai  = a + m_iter * MR * rs_a0;
      dcomplex* restrict bj  = b;

      if ( 8 <= m_left )
      {
        const dim_t      mr_cur = 8;
        bli_zgemmsup_cd_zen4_asm_8x4
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 4 <= m_left )
      {
        const dim_t      mr_cur = 4;
        bli_zgemmsup_cd_zen4_asm_4x4
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 2 <= m_left )
      {
        const dim_t      mr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_2x4
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 1 == m_left )
      {
        // Call to GEMV, which internally uses AXPYF kernels(var2)
        bli_zgemv_ex
        (
          BLIS_TRANSPOSE, conja, k0, n0,
          alpha, bj, rs_b0, cs_b0, ai, cs_a0,
          beta, cij, cs_c0, cntx, NULL
        );
      }
    }
}

void bli_zgemmsup_cd_zen4_asm_12x2m
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
    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;
    uint64_t m_iter = m0 / MR; // To be used for MR loop in the kernel
    uint64_t m_left = m0 % MR; // To be used to dispatch ?x4m kernels

    if ( m_iter == 0 ) goto consider_edge_cases;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(VAR(m_iter), R11)     // Iterating in steps of MR, until MC(m var)
    LABEL(.ZMLOOP)
    MOV(IMM(0), R9)
    LABEL(.ZLOOP_4I)          // Iterating in steps of 4, until MR
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next 4xKC block of A and 4x2 block of C
    */
    LEA(MEM(R10, R13, 4), R10)
    ADD(IMM(64), RCX)

    ADD(IMM(4), R9)
    CMP(IMM(12), R9)
    JNE(.ZLOOP_4I)

    DEC(R11)
    JNE(.ZMLOOP)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [m_iter]  "m" (m_iter),
      [m_left]  "m" (m_left),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
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
      dcomplex* restrict cij = c + m_iter * MR * rs_c0;
      dcomplex* restrict ai  = a + m_iter * MR * rs_a0;
      dcomplex* restrict bj  = b;

      if ( 8 <= m_left )
      {
        const dim_t      mr_cur = 8;
        bli_zgemmsup_cd_zen4_asm_8x2
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 4 <= m_left )
      {
        const dim_t      mr_cur = 4;
        bli_zgemmsup_cd_zen4_asm_4x2
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 2 <= m_left )
      {
        const dim_t      mr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_2x2
        (
          conja, conjb, mr_cur, n0, k0, alpha,
          ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0,
          beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );
        cij += mr_cur * rs_c0; ai += mr_cur * rs_a0;
        m_left -= mr_cur;
      }
      if ( 1 == m_left )
      {
        // Call to GEMV, which internally uses AXPYF kernels(var2)
        bli_zgemv_ex
        (
          BLIS_TRANSPOSE, conja, k0, n0,
          alpha, bj, rs_b0, cs_b0, ai, cs_a0,
          beta, cij, cs_c0, cntx, NULL
        );
      }
    }
}

void bli_zgemmsup_cd_zen4_asm_8x4
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
    uint64_t n_left = n0 % NR;

    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 8x? kernels, as needed.
    if ( n_left )
    {
      dcomplex*  cij = c;
      dcomplex*  bj  = b;
      dcomplex*  ai  = a;

      if ( 2 <= n_left )
      {
        const dim_t nr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_8x2
        (
          conja, conjb, m0, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0, beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );

        cij += 2 * cs_c0;
        bj += 2 * cs_b0;
        n_left -= 2;
      }
      if ( 1 == n_left )
      {
        // Call to GEMV, which internally uses DOTXF kernels(var1)
        bli_zgemv_ex
        (
          BLIS_NO_TRANSPOSE, conjb, m0, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0,
          beta, cij, rs_c0, cntx, NULL
        );
      }
      return;
    }

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(IMM(0), R15)
    LABEL(.ZLOOP_2J)          // Iterating in blocks of 2, until NR
    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(IMM(0), R9)
    LABEL(.ZLOOP_4I)          // Iterating in steps of 4, until MR
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next 4xKC block of A and 4x2 block of C
    */
    LEA(MEM(R10, R13, 4), R10)
    ADD(IMM(64), RCX)

    ADD(IMM(4), R9)
    CMP(IMM(8), R9)
    JNE(.ZLOOP_4I)

    LEA(MEM(RDX, R14, 2), RDX)
    LEA(MEM(R12, RSI, 2), R12)
    MOV(VAR(a), R10)

    ADD(IMM(2), R15)
    CMP(IMM(4), R15)
    JNE(.ZLOOP_2J)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r12", "r13", "r14", "r15",
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

void bli_zgemmsup_cd_zen4_asm_4x4
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
    uint64_t n_left = n0 % NR;

    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 4x? kernels, as needed.
    if ( n_left )
    {
      dcomplex*  cij = c;
      dcomplex*  bj  = b;
      dcomplex*  ai  = a;

      if ( 2 <= n_left )
      {
        const dim_t nr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_4x2
        (
          conja, conjb, m0, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0, beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );

        cij += 2 * cs_c0;
        bj += 2 * cs_b0;
        n_left -= 2;
      }
      if ( 1 == n_left )
      {
        // Call to GEMV, which internally uses DOTXF kernels(var1)
        bli_zgemv_ex
        (
          BLIS_NO_TRANSPOSE, conjb, m0, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0,
          beta, cij, rs_c0, cntx, NULL
        );
      }
      return;
    }

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(IMM(0), R15)
    LABEL(.ZLOOP_2J)          // Iterating in blocks of 2, until NR
    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)

    LEA(MEM(RDX, R14, 2), RDX)
    LEA(MEM(R12, RSI, 2), R12)
    MOV(VAR(a), R10)

    ADD(IMM(2), R15)
    CMP(IMM(4), R15)
    JNE(.ZLOOP_2J)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r10", "r12", "r13", "r14", "r15",
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

void bli_zgemmsup_cd_zen4_asm_2x4
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
    uint64_t n_left = n0 % NR;

    // Checking whether this is a edge case in the n dimension.
    // If so, dispatch other 2x? kernels, as needed.
    if ( n_left )
    {
      dcomplex*  cij = c;
      dcomplex*  bj  = b;
      dcomplex*  ai  = a;

      if ( 2 <= n_left )
      {
        const dim_t nr_cur = 2;
        bli_zgemmsup_cd_zen4_asm_2x2
        (
          conja, conjb, m0, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0,
          bj, rs_b0, cs_b0, beta,
          cij, rs_c0, cs_c0,
          data, cntx
        );

        cij += 2 * cs_c0;
        bj += 2 * cs_b0;
        n_left -= 2;
      }
      if ( 1 == n_left )
      {
        // Call to GEMV, which internally uses DOTXF kernels(var1)
        bli_zgemv_ex
        (
          BLIS_NO_TRANSPOSE, conjb, m0, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0,
          beta, cij, rs_c0, cntx, NULL
        );
      }
      return;
    }

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in m-direction and k-direction
    uint64_t m_fringe_mask = 0xF; // Loaded onto k4 register
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(IMM(0), R15)
    LABEL(.ZLOOP_2J)          // Iterating in blocks of 2, until NR
    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 2 x KC block of A
    // KC x 2 block of B
    // 2 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_2x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_2x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(7) and ZMM(14) to ZMM(15) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_2R
    (
      6, 7,   // A(.)B[0] registers
      10, 11, // A(.)(permute(B[0])) registers
      22, 23
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_2R
    (
      14, 15, // A(.)B[1] registers
      18, 19, // A(.)(permute(B[1])) registers
      26, 27
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_2R
    (
      10, 11,
      22, 23
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_2R
    (
      18, 19,
      26, 27
    )

    /*
      The final result should be as follows :
      C(2x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(11)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13(with ZMM12 and ZMM13
             being 0) will arrange the elements in such a way that simple
             addition produces the result in a packed manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(2x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Setting the mask onto the mask register k4
    // This is used for loading/storing with C matrix
    MOV(var(m_fringe_mask), R9)
    KMOVQ(R9, K(4))
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    VMOVUPD(MEM(RCX), ZMM(30) MASK_K(4))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31) MASK_K(4))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX) MASK_K(4))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1) MASK_K(4))

    LABEL(.END)

    LEA(MEM(RDX, R14, 2), RDX)
    LEA(MEM(R12, RSI, 2), R12)
    MOV(VAR(a), R10)

    ADD(IMM(2), R15)
    CMP(IMM(4), R15)
    JNE(.ZLOOP_2J)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [m_fringe_mask]  "m" (m_fringe_mask),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r12", "r13", "r14", "r15",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7",
      "zmm8", "zmm9", "zmm10", "zmm11",
      "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23",
      "zmm24", "zmm25", "zmm26", "zmm27",
      "zmm28", "zmm29", "zmm30", "zmm31",
      "k3", "k4", "memory"
    )
}

void bli_zgemmsup_cd_zen4_asm_8x2
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
    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(IMM(0), R9)
    LABEL(.ZLOOP_4I)          // Iterating in steps of 4, until MR
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)
    /*
      Adjusting the addresses for loading the
      next 4xKC block of A and 4x2 block of C
    */
    LEA(MEM(R10, R13, 4), R10)
    ADD(IMM(64), RCX)

    ADD(IMM(4), R9)
    CMP(IMM(8), R9)
    JNE(.ZLOOP_4I)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r12", "r13", "r14",
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

void bli_zgemmsup_cd_zen4_asm_4x2
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
    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in k-direction
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 4 x KC block of A
    // KC x 2 block of B
    // 4 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_4x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_4x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_4R
    (
      6, 7, 8, 9,     // A(.)B[0] registers
      10, 11, 12, 13, // A(.)(permute(B[0])) registers
      22, 23, 24, 25
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_4R
    (
      14, 15, 16, 17, // A(.)B[1] registers
      18, 19, 20, 21, // A(.)(permute(B[1])) registers
      26, 27, 28, 29
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_4R
    (
      10, 11, 12, 13,
      22, 23, 24, 25
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_4R
    (
      18, 19, 20, 21,
      26, 27, 28, 29
    )

    /*
      The final result should be as follows :
      C(4x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))
               reduce(ZMM(12)) reduce(ZMM(20))
               reduce(ZMM(13)) reduce(ZMM(21))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(13)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13 will arrange the elements
             in such a way that simple addition produces the result in a packed
             manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(4x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    // Load two columns from C
    VMOVUPD(MEM(RCX), ZMM(30))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1))

    LABEL(.END)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r10", "r12", "r13", "r14",
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

void bli_zgemmsup_cd_zen4_asm_2x2
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
    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t cs_c   = cs_c0;

    uint64_t k_iter_16 = k0 / 16; // Unroll factor of 16
    uint64_t k_left_16 = k0 % 16;
    uint64_t k_iter_4 = k_left_16 / 4;
    uint64_t k_left = k_left_16 % 4;

    const double value = 1.0; // To be broadcasted and used for complex arithmetic
    const double *v = &value;

    // Setting the mask for fringe case in m-direction and k-direction
    uint64_t m_fringe_mask = 0xF; // Loaded onto k4 register
    uint64_t k_fringe_mask = (1 << 2 * k_left) - 1; // Loaded onto k3 register

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

    // Assembly code-section
    BEGIN_ASM()
    MOV(VAR(a), R10)          // R10 = base addr of A (MCXKC block)
    MOV(VAR(b), RDX)          // RDX = base addr of B (KCXNR block)
    MOV(VAR(c), R12)          // R12 = base addr of C (MCxNR block)

    MOV(VAR(rs_a), R13)
    LEA(MEM(, R13, 8), R13)
    LEA(MEM(, R13, 2), R13)   // R13 = sizeof(dcomplex)*rs_a

    MOV(VAR(cs_b), R14)
    LEA(MEM(, R14, 8), R14)
    LEA(MEM(, R14, 2), R14)   // R14 = sizeof(dcomplex)*cs_b

    MOV(VAR(cs_c), RSI)
    LEA(MEM(, RSI, 8), RSI)
    LEA(MEM(, RSI, 2), RSI)   // RSI = sizeof(dcomplex)*cs_c

    MOV(R12, RCX)             // RCX = addr of C for the MRxNR block
    MOV(RDX, RBX)             // RBX = addr of B for the KCxNR block
    MOV(R10, RAX)             // RAX = addr of A for the MRxKC block
    // Resetting all scratch registers for arithmetic and accumulation
    RESET_REGISTERS

    // At this point, the computation involves :
    // 2 x KC block of A
    // KC x 2 block of B
    // 2 x 2 block of C

    // Setting iterator for k
    MOV(var(k_iter_16), R8)
    TEST(R8, R8)
    JE(.ZKITER4)
    // Main loop for k(with unroll = 16)
    LABEL(.ZKITER16)

    // Computing the 4x2 micro-tiles with k in blocks of 16
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()
    MICRO_TILE_2x2_MAIN()

    DEC(R8)             // k_iter_16 -= 1
    JNZ(.ZKITER16)

    // Fringe loop for k(with unroll = 4)
    LABEL(.ZKITER4)
    MOV(VAR(k_iter_4), R8)
    TEST(R8, R8)
    JE(.ZKFRINGE)
    LABEL(.ZKITER4LOOP)

    // Computing the 4x2 micro-tiles with k in blocks of 4
    MICRO_TILE_2x2_MAIN()

    DEC(R8)             // k_iter_4 -= 1
    JNZ(.ZKITER4LOOP)

    // Remainder loop for k(with unroll = 4)
    LABEL(.ZKFRINGE)
    MOV(VAR(k_left), R8)
    TEST(R8, R8)
    JE(.ACCUMULATE)

    // Setting the mask onto the mask register k3
    MOV(var(k_fringe_mask), R8)
    KMOVQ(R8, K(3))

    // Computing the 4x2 micro-tile for k < 4
    MICRO_TILE_2x2_FRINGE(3)

    LABEL(.ACCUMULATE) // Accumulating A*B over 12 registers
    /*
        ZMM(6) to ZMM(9) and ZMM(14) to ZMM(17) contain the partial
        sums from A(.)B.
        ZMM(10) to ZMM(13) and ZMM(18) to ZMM(21) contain the partial
        sums from A(.)permute(B).
    */

    // Intermediate register for complex arithmetic
    MOV(VAR(v), R8)  // Used in fmaddsub instruction
    VBROADCASTSD(MEM(R8), ZMM(30)) // Broadcasting 1.0 over ZMM(29)

    /*
        The following shuffling is performed below :
        For one A(.)B and A(.)permute(B) pair(say, ZMM(6) and ZMM(10)),
        ZMM(6)  = AR1.BR1 AI1.BI1 AR2.BR2 AI2.BI2 ...
        ZMM(10) = AR1.BI1 AI1.BR1 AR2.BI2 AI2.BR2 ...

        Shuffle ZMM(6) and ZMM(10) as follows
        ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
        ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...
    */
    // Shuffling every pair computed with 4 rows of A and first column of B
    SHUFFLE_2R
    (
      6, 7,   // A(.)B[0] registers
      10, 11, // A(.)(permute(B[0])) registers
      22, 23
    )
    // Shuffling every pair computed with 4 rows of A and second column of B
    SHUFFLE_2R
    (
      14, 15, // A(.)B[1] registers
      18, 19, // A(.)(permute(B[1])) registers
      26, 27
    )
    /*
      The following reduction is performed post the shuffles
      ZMM(22) = AR1.BR1 AR1.BI1 AR2.BR2 AR2.BI2 ...
      ZMM(10) = AI1.BI1 AI1.BR1 AI2.BI2 AI2.BR2 ...

      ZMM(10) = ( AR1.BR1 - AI1.BI1 ) ( AR1.BI1 - AI1.BR1 ) ...
    */
    // Reduction from computations with first column of B(i.e, B[0])
    ACCUMULATE_2R
    (
      10, 11,
      22, 23
    )
    // Reduction from computations with second column of B(i.e, B[1])
    ACCUMULATE_2R
    (
      18, 19,
      26, 27
    )

    /*
      The final result should be as follows :
      C(2x2) = reduce(ZMM(10)) reduce(ZMM(18))
               reduce(ZMM(11)) reduce(ZMM(19))

      Complex numbers follow interleaved format.
      Thus, if ZMM(10)      = R1 I1 R2 I2 R3 I3 R4 I4
      Then  reduce(ZMM(10)) = (R1 + R2 + R3 + R4) (I1 + I2 + I3 + I4)

      NOTE : reduce(ZMM(10)) to reduce(ZMM(11)) form the first column of C
             A 4x4 128-lane transpose of ZMM10 to ZMM13(with ZMM12 and ZMM13
             being 0) will arrange the elements in such a way that simple
             addition produces the result in a packed manner.
    */
    // Transposing ZMM(10) to ZMM(13)(first column of C)
    TRANSPOSE_4x4
    (
      10, 11, 12, 13,
      6, 7, 8, 9
    )
    // Transposing ZMM(18) to ZMM(21)(first column of C)
    TRANSPOSE_4x4
    (
      18, 19, 20, 21,
      14, 15, 16, 17
    )

    /*
      Post transpose :
      C(2x2) = ZMM(6)   ZMM(14)
                 +        +
               ZMM(7)   ZMM(15)
                 +        +
               ZMM(8)   ZMM(16)
                 +        +
               ZMM(9)   ZMM(17)
    */
    VADDPD(ZMM(6), ZMM(7), ZMM(6))
    VADDPD(ZMM(8), ZMM(9), ZMM(8))
    VADDPD(ZMM(6), ZMM(8), ZMM(6))

    VADDPD(ZMM(14), ZMM(15), ZMM(14))
    VADDPD(ZMM(16), ZMM(17), ZMM(16))
    VADDPD(ZMM(14), ZMM(16), ZMM(14))

    // Alpha scaling
    LABEL(.ALPHA_SCALE)
    MOV(VAR(alpha_mul_type), AL)
    CMP(IMM(0xFF), AL) // Checking if alpha == -1
    JNE(.ALPHA_GENERAL)
    // Handling when alpha == -1
    VXORPD(ZMM(2), ZMM(2), ZMM(2)) // Resetting ZMM(2) to 0
    VSUBPD(ZMM(6), ZMM(2), ZMM(6))
    VSUBPD(ZMM(14), ZMM(2), ZMM(14))
    JMP(.BETA_SCALE)

    LABEL(.ALPHA_GENERAL)
    CMP(IMM(2), AL) // Checking if alpha == BLIS_MUL_DEFAULT
    JNE(.BETA_SCALE)
    MOV(VAR(alpha), RAX)
    VBROADCASTSD(MEM(RAX), ZMM(0))  // Alpha->real
    VBROADCASTSD(MEM(RAX, 8), ZMM(1)) // Alpha->imag

    VMULPD(ZMM(1), ZMM(6), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(6))

    VMULPD(ZMM(1), ZMM(14), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(14))

    // Beta scaling
    LABEL(.BETA_SCALE)
    // Setting the mask onto the mask register k4
    // This is used for loading/storing with C matrix
    MOV(var(m_fringe_mask), R9)
    KMOVQ(R9, K(4))
    // Beta scaling when C is column stored
    MOV(VAR(beta_mul_type), AL)
    CMP(IMM(0), AL)    // Checking if beta == 0
    JE(.STORE)
    VMOVUPD(MEM(RCX), ZMM(30) MASK_K(4))
    VMOVUPD(MEM(RCX, RSI, 1), ZMM(31) MASK_K(4))
    CMP(IMM(0x01), AL) // Checking if beta == 1
    JE(.ADD)
    CMP(IMM(0xFF), AL) // Checking if beta == -1
    JNE(.BETA_GENERAL)

    // Handling when beta is -1
    VSUBPD(ZMM(30), ZMM(6), ZMM(6))
    VSUBPD(ZMM(31), ZMM(14), ZMM(14))
    JMP(.STORE)

    LABEL(.BETA_GENERAL) // Checking if beta == BLIS_MUL_DEFAULT
    MOV(VAR(beta), RBX)
    VBROADCASTSD(MEM(RBX), ZMM(0))    // Beta->real
    VBROADCASTSD(MEM(RBX, 8), ZMM(1)) // Beta->imag

    VMULPD(ZMM(1), ZMM(30), ZMM(3))
    VPERMILPD(IMM(0x55), ZMM(3), ZMM(3))
    VFMADDSUB132PD(ZMM(0), ZMM(3), ZMM(30))

    VMULPD(ZMM(1), ZMM(31), ZMM(4))
    VPERMILPD(IMM(0x55), ZMM(4), ZMM(4))
    VFMADDSUB132PD(ZMM(0), ZMM(4), ZMM(31))

    LABEL(.ADD)
    // Handling when beta is -1
    VADDPD(ZMM(30), ZMM(6), ZMM(6))
    VADDPD(ZMM(31), ZMM(14), ZMM(14))

    LABEL(.STORE)
    VMOVUPD(ZMM(6), MEM(RCX) MASK_K(4))
    VMOVUPD(ZMM(14), MEM(RCX, RSI, 1) MASK_K(4))

    LABEL(.END)

    END_ASM(
    : // output operands (none)
    : // input operands
      [v]  "m" (v),
      [k_iter_16]  "m" (k_iter_16),
      [k_iter_4]  "m" (k_iter_4),
      [k_left]  "m" (k_left),
      [m_fringe_mask]  "m" (m_fringe_mask),
      [k_fringe_mask]  "m" (k_fringe_mask),
      [alpha_mul_type]  "m" (alpha_mul_type),
      [beta_mul_type]   "m" (beta_mul_type),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [b]      "m" (b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_a]   "m" (rs_a),
      [cs_b]   "m" (cs_b),
      [cs_c]   "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r12", "r13", "r14",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7",
      "zmm8", "zmm9", "zmm10", "zmm11",
      "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23",
      "zmm24", "zmm25", "zmm26", "zmm27",
      "zmm28", "zmm29", "zmm30", "zmm31",
      "k3", "k4", "memory"
    )
}