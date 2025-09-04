/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "bli_gemmsup_rd_zen4_asm_s6x64.h"

#define NR 64


void bli_sgemmsup_rd_zen4_asm_6x64m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t n_left = n0 % NR;      // n0 is expected to be n0<=NR

    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if ( n_left )
    {
        float* restrict cij = c;
        float* restrict bj  = b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;

            bli_sgemmsup_rd_zen4_asm_6x48m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;

            bli_sgemmsup_rd_zen4_asm_6x32m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;

            bli_sgemmsup_rd_zen_asm_6x16m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            bli_sgemmsup_rd_zen_asm_6x8m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            bli_sgemmsup_rd_zen_asm_6x4m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen_asm_6x2m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }
        if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, m0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
        return;
    }

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea(mem(   , r15, 1 ), rsi)         // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;


    mov( var( m_iter ), r11 )           // ii = m_iter;
    label( .SLOOP3X4I )                 // LOOP OVER ii = [ m_iter ... 1 0 ]

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_a
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_a

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_8 )


    label( .K_LOOP_ITER32 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER32 )


    label( .CONSIDER_K_ITER_8 )
    mov( var( k_iter8 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1 )


    label( .K_LOOP_ITER8 )
    // ITER 0
    // Load row from A using ymm registers
    // Upper 256-bit lanes are cleared for the
    // zmm counterpart
    vmovups(         ( rax ), ymm0 )
    vmovups( ( rax,  r8, 1 ), ymm1 )
    vmovups( ( rax,  r8, 2 ), ymm2 )
    vmovups( ( rax, r10, 1 ), ymm3 )
    vmovups( ( rax,  r8, 4 ), ymm4 )
    vmovups( ( rax, rdi, 1 ), ymm5 )
    add( imm( 8*4 ), rax )

    // Load column from B using ymm registers
    // Upper 256-bit lane is cleared for the
    // zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovups(        ( rbx ), ymm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), ymm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), ymm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), ymm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 8*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER8 )


    label( .CONSIDER_K_LEFT_1 )
    mov( var( k_left1 ), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )


    label( .K_LOOP_LEFT1 )

    // Load row from A using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    vmovss(         ( rax ), xmm0 )
    vmovss( ( rax,  r8, 1 ), xmm1 )
    vmovss( ( rax,  r8, 2 ), xmm2 )
    vmovss( ( rax, r10, 1 ), xmm3 )
    vmovss( ( rax,  r8, 4 ), xmm4 )
    vmovss( ( rax, rdi, 1 ), xmm5 )
    add( imm( 1*4 ), rax )

    // Load column from B using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovss(        ( rbx ), xmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovss( ( rbx, r9, 1 ), xmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovss( ( rbx, r9, 2 ), xmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovss( ( rbx, r13, 1 ), xmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 1*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_LEFT1 )


    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    // The horizontal sum of each ZMM register has the result for a single
    // element of the C Matrix.
    // ZMM_TO_YMM adds the upper half of ZMM registers to the lower half of
    // the respective ZMM registers, thus having the result in the lower half of
    // ZMM registers which is equivalent to its respective YMM counterpart.
    // ymm = lo(zmm) + hi(zmm)
    // zmm8 = z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15
    // ymm0 = z8 z9 z10 z11 z12 z13 z14 z15
    // ymm8 = z0 z1  z2  z3  z4  z5  z6  z7
    // ymm0 = ymm0 + ymm8
    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    // Accumulates the results by horizontally adding the YMM registers,
    // and having the final result in xmm registers.
    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     				// Scaling the result of A*B with alpha

    C_STOR       					// Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     				// Scaling the result of A*B with alpha

    C_STOR       					// Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     				// Scaling the result of A*B with alpha

    C_STOR_BZ       				// Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     				// Scaling the result of A*B with alpha

    C_STOR_BZ       				// Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 6*rs_c

    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 6*rs_a

    dec( r11 )
    jne( .SLOOP3X4I )                   // iterate again if ii != 0.

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter8]  "m" (k_iter8),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25",
      "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 64;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen4_asm_5x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen4_asm_4x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen4_asm_3x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen4_asm_2x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen4_asm_1x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen4_asm_6x48m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(   , r15, 1 ), rsi )       // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;


    mov( var( m_iter ), r11 )           // ii = m_iter;
    label( .SLOOP3X4I )                 // LOOP OVER ii = [ m_iter ... 1 0 ]

    lea( mem( r14 ), rax )              // load c to rcx
    lea( mem( r12 ), rcx )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )

    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_8 )


    label( .K_LOOP_ITER32 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER32 )


    label( .CONSIDER_K_ITER_8 )
    mov( var( k_iter8 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1 )


    label( .K_LOOP_ITER8 )
    // ITER 0
    // Load row from A using ymm registers
    // Upper 256-bit lanes are cleared for the
    // zmm counterpart
    vmovups(         ( rax ), ymm0 )
    vmovups( ( rax,  r8, 1 ), ymm1 )
    vmovups( ( rax,  r8, 2 ), ymm2 )
    vmovups( ( rax, r10, 1 ), ymm3 )
    vmovups( ( rax,  r8, 4 ), ymm4 )
    vmovups( ( rax, rdi, 1 ), ymm5 )
    add( imm( 8*4 ), rax )

    // Load column from B using ymm registers
    // Upper 256-bit lane is cleared for the
    // zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovups(        ( rbx ), ymm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), ymm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), ymm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), ymm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 8*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER8 )


    label( .CONSIDER_K_LEFT_1 )
    mov( var( k_left1 ), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )


    label( .K_LOOP_LEFT1 )

    // Load row from A using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    vmovss(         ( rax ), xmm0 )
    vmovss( ( rax,  r8, 1 ), xmm1 )
    vmovss( ( rax,  r8, 2 ), xmm2 )
    vmovss( ( rax, r10, 1 ), xmm3 )
    vmovss( ( rax,  r8, 4 ), xmm4 )
    vmovss( ( rax, rdi, 1 ), xmm5 )
    add( imm( 1*4 ), rax )                 // a += 1*cs_b = 1*4;

    // Load column from B using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovss(        ( rbx ), xmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovss( ( rbx, r9, 1 ), xmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovss( ( rbx, r9, 2 ), xmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovss( ( rbx, r13, 1 ), xmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 1*4 ), rbx )                 // b += 1*rs_b = 1*4;

    dec( rsi )
    jne( .K_LOOP_LEFT1 )


    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    // The horizontal sum of each ZMM register has the result for a single
    // element of the C Matrix.
    // ZMM_TO_YMM adds the upper half of ZMM registers to the lower half of
    // the respective ZMM registers, thus having the result in the lower half of
    // ZMM registers which is equivalent to its respective YMM counterpart.
    // ymm = lo(zmm) + hi(zmm)
    // zmm8 = z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15
    // ymm0 = z8 z9 z10 z11 z12 z13 z14 z15
    // ymm8 = z0 z1  z2  z3  z4  z5  z6  z7
    // ymm0 = ymm0 + ymm8
    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    // Accumulates the results by horizontally adding the YMM registers,
    // and having the final result in xmm registers.
    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 6*rs_c

    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 6*rs_a

    dec( r11 )
    jne( .SLOOP3X4I )                    // iterate again if ii != 0.

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter8]  "m" (k_iter8),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25",
      "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 48;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen4_asm_5x48
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen4_asm_4x48
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen4_asm_3x48
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen4_asm_2x48
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen4_asm_1x48
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen4_asm_6x32m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b


    mov( imm(0), r15 )                  // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(   , r15, 1 ), rsi )       // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;


    mov( var( m_iter ), r11 )           // ii = m_iter;
    label( .SLOOP3X4I )                 // LOOP OVER ii = [ m_iter ... 1 0 ]

    lea( mem( r14 ), rax )              // load c to rcx
    lea( mem( r12 ), rcx )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_8 )

    label( .K_LOOP_ITER32 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    vmovups( ( rax, rdi, 1 ), zmm5 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER32 )


    label( .CONSIDER_K_ITER_8 )
    mov( var( k_iter8 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1 )


    label( .K_LOOP_ITER8 )
    // ITER 0
    // Load row from A using ymm registers
    // Upper 256-bit lanes are cleared for the
    // zmm counterpart
    vmovups(         ( rax ), ymm0 )
    vmovups( ( rax,  r8, 1 ), ymm1 )
    vmovups( ( rax,  r8, 2 ), ymm2 )
    vmovups( ( rax, r10, 1 ), ymm3 )
    vmovups( ( rax,  r8, 4 ), ymm4 )
    vmovups( ( rax, rdi, 1 ), ymm5 )
    add( imm( 8*4 ), rax )

    // Load column from B using ymm registers
    // Upper 256-bit lane is cleared for the
    // zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovups(        ( rbx ), ymm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovups( ( rbx, r9, 1 ), ymm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovups( ( rbx, r9, 2 ), ymm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovups( ( rbx, r13, 1 ), ymm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 8*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER8 )


    label( .CONSIDER_K_LEFT_1 )
    mov( var( k_left1 ), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )


    label( .K_LOOP_LEFT1 )

    // Load row from A using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    vmovss(         ( rax ), xmm0 )
    vmovss( ( rax,  r8, 1 ), xmm1 )
    vmovss( ( rax,  r8, 2 ), xmm2 )
    vmovss( ( rax, r10, 1 ), xmm3 )
    vmovss( ( rax,  r8, 4 ), xmm4 )
    vmovss( ( rax, rdi, 1 ), xmm5 )
    add( imm( 1*4 ), rax )                 // a += 1*cs_b = 1*4;

    // Load column from B using xmm registers
    // Upper 256-bit lanes and the upper 224
    // bits of the lower 256-bit lane are cleared
    // for the zmm counterpart
    // Thus, we can re-use the VFMA6 macro
    vmovss(        ( rbx ), xmm6 )
    VFMA6(  8,  9, 10, 20, 21, 22 )

    vmovss( ( rbx, r9, 1 ), xmm6 )
    VFMA6( 11, 12, 13, 23, 24, 25 )

    vmovss( ( rbx, r9, 2 ), xmm6 )
    VFMA6( 14, 15, 16, 26, 27, 28 )

    vmovss( ( rbx, r13, 1 ), xmm6 )
    VFMA6( 17, 18, 19, 29, 30, 31 )

    add( imm( 1*4 ), rbx )                 // b += 1*rs_b = 1*4;

    dec( rsi )
    jne( .K_LOOP_LEFT1 )


    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )

    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    // The horizontal sum of each ZMM register has the result for a single
    // element of the C Matrix.
    // ZMM_TO_YMM adds the upper half of ZMM registers to the lower half of
    // the respective ZMM registers, thus having the result in the lower half of
    // ZMM registers which is equivalent to its respective YMM counterpart.
    // ymm = lo(zmm) + hi(zmm)
    // zmm8 = z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15
    // ymm0 = z8 z9 z10 z11 z12 z13 z14 z15
    // ymm8 = z0 z1  z2  z3  z4  z5  z6  z7
    // ymm0 = ymm0 + ymm8
    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    // Accumulates the results by horizontally adding the YMM registers,
    // and having the final result in xmm registers.
    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C

    ZMM_TO_YMM( 20, 21, 22, 23,  4,  5,  6,  7 )
    ZMM_TO_YMM( 24, 25, 26, 27,  8,  9, 10, 11 )
    ZMM_TO_YMM( 28, 29, 30, 31, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 6*rs_c

    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 6*rs_a

    dec( r11 )
    jne( .SLOOP3X4I )                    // iterate again if ii != 0.

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )


    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter8]  "m" (k_iter8),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25",
      "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 32;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen4_asm_5x32
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen4_asm_4x32
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen4_asm_3x32
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen4_asm_2x32
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen4_asm_1x32
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}
