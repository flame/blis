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

#include "bli_gemmsup_rv_zen_s6x64.h"

void bli_sgemmsup_rv_zen_asm_5x48_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop

    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3( 7,  8,  9, 10 )
    ALPHA_SCALE3( 7, 12, 13, 14 )
    ALPHA_SCALE3( 7, 16, 17, 18 )
    ALPHA_SCALE3( 7, 20, 21, 22 )
    ALPHA_SCALE3( 7, 24, 25, 26 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C3( 4,  8,  9, 10 )
    UPDATE_C3( 4, 12, 13, 14 )
    UPDATE_C3( 4, 16, 17, 18 )
    UPDATE_C3( 4, 20, 21, 22 )
    UPDATE_C3( 4, 24, 25, 26 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16(  8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16(  9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 24 )
    UPDATE_C_1X16( 25 )
    UPDATE_C_1X16( 26 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ(  8,  9, 10 )
    UPDATE_C3_BZ( 12, 13, 14 )
    UPDATE_C3_BZ( 16, 17, 18 )
    UPDATE_C3_BZ( 20, 21, 22 )
    UPDATE_C3_BZ( 24, 25, 26 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ(  8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ(  9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 24 )
    UPDATE_C_1X16_BZ( 25 )
    UPDATE_C_1X16_BZ( 26 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_5x32_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2( 7,  8,  9 )
    ALPHA_SCALE2( 7, 12, 13 )
    ALPHA_SCALE2( 7, 16, 17 )
    ALPHA_SCALE2( 7, 20, 21 )
    ALPHA_SCALE2( 7, 24, 25 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C2( 4,  8,  9 )
    UPDATE_C2( 4, 12, 13 )
    UPDATE_C2( 4, 16, 17 )
    UPDATE_C2( 4, 20, 21 )
    UPDATE_C2( 4, 24, 25 )
    jmp(.SDONE)

    label( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 24 )
    UPDATE_C_1X16( 25 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ(  8,  9 )
    UPDATE_C2_BZ( 12, 13 )
    UPDATE_C2_BZ( 16, 17 )
    UPDATE_C2_BZ( 20, 21 )
    UPDATE_C2_BZ( 24, 25 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 24 )
    UPDATE_C_1X16_BZ( 25 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_5x16_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA1( 5, 24 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA1( 5, 24 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA1( 5, 24 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA1( 5, 24 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA1( 5, 24 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1( 7,  8 )
    ALPHA_SCALE1( 7, 12 )
    ALPHA_SCALE1( 7, 16 )
    ALPHA_SCALE1( 7, 20 )
    ALPHA_SCALE1( 7, 24 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C1( 4,  8 )
    UPDATE_C1( 4, 12 )
    UPDATE_C1( 4, 16 )
    UPDATE_C1( 4, 20 )
    UPDATE_C1( 4, 24 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16(  8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 24 )
    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C1_BZ( 8 )
    UPDATE_C1_BZ( 12 )
    UPDATE_C1_BZ( 16 )
    UPDATE_C1_BZ( 20 )
    UPDATE_C1_BZ( 24 )
    jmp(.SDONE)

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )


    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 24 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_3x48_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3( 7,  8,  9, 10 )
    ALPHA_SCALE3( 7, 12, 13, 14 )
    ALPHA_SCALE3( 7, 16, 17, 18 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C3( 4,  8,  9, 10 )
    UPDATE_C3( 4, 12, 13, 14 )
    UPDATE_C3( 4, 16, 17, 18 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16(  8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16(  9, 13 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 10, 14 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 16 )
    UPDATE_C_1X16( 17 )
    UPDATE_C_1X16( 18 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ(  8,  9, 10 )
    UPDATE_C3_BZ( 12, 13, 14 )
    UPDATE_C3_BZ( 16, 17, 18 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ(  8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ(  9, 13 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 10, 14 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 16 )
    UPDATE_C_1X16_BZ( 17 )
    UPDATE_C_1X16_BZ( 18 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_3x32_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2( 7,  8,  9 )
    ALPHA_SCALE2( 7, 12, 13 )
    ALPHA_SCALE2( 7, 16, 17 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C2( 4,  8,  9 )
    UPDATE_C2( 4, 12, 13 )
    UPDATE_C2( 4, 16, 17 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16( 8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 9, 13 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 16 )
    UPDATE_C_1X16( 17 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ(  8,  9 )
    UPDATE_C2_BZ( 12, 13 )
    UPDATE_C2_BZ( 16, 17 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ( 8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 9, 13 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 16 )
    UPDATE_C_1X16_BZ( 17 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_3x16_avx512
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1( 7,  8 )
    ALPHA_SCALE1( 7, 12 )
    ALPHA_SCALE1( 7, 16 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C1( 4,  8 )
    UPDATE_C1( 4, 12 )
    UPDATE_C1( 4, 16 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16( 8, 12 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 16 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C1_BZ(  8 )
    UPDATE_C1_BZ( 12 )
    UPDATE_C1_BZ( 16 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ( 8, 12 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )             // load cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c *= sizeof(dt) => cs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 16 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}
