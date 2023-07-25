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

#define MR 6

/*
   rrr:
     --------        ------        --------
     --------        ------        --------
     --------   +=   ------ ...    --------
     --------        ------        --------
     --------        ------            :
     --------        ------            :
   Assumptions:
   - B is row-stored;
   - A is row-stored;
   - m0 and n0 are at most MR (6) and NR (64), respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.
*/
void bli_sgemmsup_rv_zen_asm_6x64n_avx512
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
    uint64_t m_left = m0 % MR;      // m0 is expected to be m0<=MR

    if ( m_left ) {
        float* restrict cij = c;
        float* restrict bj  = b;
        float* restrict ai  = a;

        if ( 5 <= m_left ) {
            bli_sgemmsup_rv_zen_asm_5x64n_avx512(
              conja, conjb, m_left, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            return;
        }

        if ( 4 <= m_left ) {
            bli_sgemmsup_rv_zen_asm_4x64n_avx512(
              conja, conjb, m_left, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            return;
        }

        if ( 3 <= m_left ) {
            bli_sgemmsup_rv_zen_asm_3x64n_avx512(
              conja, conjb, m_left, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            return;
        }

        if ( 2 <= m_left ) {
            bli_sgemmsup_rv_zen_asm_2x64n_avx512(
              conja, conjb, m_left, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            return;
        }

        if ( 1 <= m_left ) {
            bli_sgemmsup_rv_zen_asm_1x64n_avx512(
              conja, conjb, m_left, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            return;
        }
    }

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

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

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4( 7, 16, 17, 18, 19 )
    ALPHA_SCALE4( 7, 20, 21, 22, 23 )
    ALPHA_SCALE4( 7, 24, 25, 26, 27 )
    ALPHA_SCALE4( 7, 28, 29, 30, 31 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )
    UPDATE_C4( 4, 12, 13, 14, 15 )
    UPDATE_C4( 4, 16, 17, 18, 19 )
    UPDATE_C4( 4, 20, 21, 22, 23 )
    UPDATE_C4( 4, 24, 25, 26, 27 )
    UPDATE_C4( 4, 28, 29, 30, 31 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16(  8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16(  9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 11, 15, 19, 23 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    lea( mem( rcx, r10, 4 ), rcx )
    TRANSPOSE_2X16( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 25, 29 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 26, 30 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 27, 31 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ(  8,  9, 10, 11 )
    UPDATE_C4_BZ( 12, 13, 14, 15 )
    UPDATE_C4_BZ( 16, 17, 18, 19 )
    UPDATE_C4_BZ( 20, 21, 22, 23 )
    UPDATE_C4_BZ( 24, 25, 26, 27 )
    UPDATE_C4_BZ( 28, 29, 30, 31 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16_BZ(  8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ(  9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 11, 15, 19, 23 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    lea( mem( rcx, r10, 4 ), rcx )
    TRANSPOSE_2X16_BZ( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 25, 29 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 26, 30 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 27, 31 )

    jmp( .SDONE )                     // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )          // load ps_b4; rdx = ps_b4
    mov( var( bbuf ), rbx )           // load b
    add( rdx, rbx )                   // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )           // load cs_c; rdx = cs_c
    lea( mem( , rdx, 4 ), rdx )       // rdx = cs_c*sizeof(dt) => rdx = cs_c*4
    lea( mem( , rdx, 8 ), rdx )       // rdx = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )       // rdx = rdx * 8 = cs_c * 8 * 8
                                      // => rdx = cs_c * 64
    mov( var( cbuf ), rcx )           // load address of c
    add( rdx, rcx )                   // c += rs_c * MR
    mov( rcx, var( cbuf ) )           // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
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
    if ( n_left )
    {
        const dim_t mr_cur = 6;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_6x48m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_6x32m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_6x16m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_6x8m
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_6x4m
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_6x2m
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 6 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 6;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_5x64n_avx512
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

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )

    add( r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )       // i = k_left;
    test( rsi, rsi )                // check i via logical AND.
    je( .SPOSTACCUM )               // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.


    label( .K_LEFT_LOOP )

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 5 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )

    add( r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4( 7, 16, 17, 18, 19 )
    ALPHA_SCALE4( 7, 20, 21, 22, 23 )
    ALPHA_SCALE4( 7, 24, 25, 26, 27 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )
    UPDATE_C4( 4, 12, 13, 14, 15 )
    UPDATE_C4( 4, 16, 17, 18, 19 )
    UPDATE_C4( 4, 20, 21, 22, 23 )
    UPDATE_C4( 4, 24, 25, 26, 27 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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
    TRANSPOSE_4X16( 11, 15, 19, 23 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )
    lea( mem( , rdi, 4 ), rdi )
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 24 )
    UPDATE_C_1X16( 25 )
    UPDATE_C_1X16( 26 )
    UPDATE_C_1X16( 27 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ(  8,  9, 10, 11 )
    UPDATE_C4_BZ( 12, 13, 14, 15 )
    UPDATE_C4_BZ( 16, 17, 18, 19 )
    UPDATE_C4_BZ( 20, 21, 22, 23 )
    UPDATE_C4_BZ( 24, 25, 26, 27 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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
    TRANSPOSE_4X16_BZ( 11, 15, 19, 23 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )
    lea( mem( , rdi, 4 ), rdi )
    lea( mem( rcx, rdi, 4 ), rcx )
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 24 )
    UPDATE_C_1X16_BZ( 25 )
    UPDATE_C_1X16_BZ( 26 )
    UPDATE_C_1X16_BZ( 27 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )    // load ps_b4
    mov( var( bbuf ), rbx )     // load b
    add( rdx, rbx )             // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )
    lea( mem( , rdx, 4 ), rdx )
    lea( mem( , rdx, 8 ), rdx )     // rdx  = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )     // rdx  = rdx * 8 = cs_c * 8 * 8 => rdx = cs_c * 64
    mov( var( cbuf ), rcx )              // load address of c
    add( rdx, rcx )                    // c += rs_c * MR
    mov( rcx, var( cbuf ) )              // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
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

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( n_left )
    {
        const dim_t mr_cur = 5;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_5x48_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_5x32_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_5x16_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_5x8
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_5x4
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_5x2
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 5 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 5;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_4x64n_avx512
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

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )

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

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4( 7, 16, 17, 18, 19 )
    ALPHA_SCALE4( 7, 20, 21, 22, 23 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )
    UPDATE_C4( 4, 12, 13, 14, 15 )
    UPDATE_C4( 4, 16, 17, 18, 19 )
    UPDATE_C4( 4, 20, 21, 22, 23 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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
    TRANSPOSE_4X16( 11, 15, 19, 23 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ(  8,  9, 10, 11 )
    UPDATE_C4_BZ( 12, 13, 14, 15 )
    UPDATE_C4_BZ( 16, 17, 18, 19 )
    UPDATE_C4_BZ( 20, 21, 22, 23 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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
    TRANSPOSE_4X16_BZ( 11, 15, 19, 23 )

    jmp( .SDONE )                     // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )          // load ps_b4; rdx = ps_b4
    mov( var( bbuf ), rbx )           // load b
    add( rdx, rbx )                   // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )           // load cs_c; rdx = cs_c
    lea( mem( , rdx, 4 ), rdx )       // rdx = cs_c*sizeof(dt) => rdx = cs_c*4
    lea( mem( , rdx, 8 ), rdx )       // rdx = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )       // rdx = rdx * 8 = cs_c * 8 * 8
                                      // => rdx = cs_c * 64
    mov( var( cbuf ), rcx )           // load address of c
    add( rdx, rcx )                   // c += rs_c * MR
    mov( rcx, var( cbuf ) )           // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
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
    if ( n_left )
    {
        const dim_t mr_cur = 4;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_4x48m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_4x32m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_4x16m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_4x8
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_4x4
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_4x2
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 4 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 4;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_3x64n_avx512
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

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )

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

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4( 7, 16, 17, 18, 19 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )
    UPDATE_C4( 4, 12, 13, 14, 15 )
    UPDATE_C4( 4, 16, 17, 18, 19 )

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
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 11, 15 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )
    mov( var( rs_c ), rdi )
    lea( mem( , rdi, 4 ), rdi )
    lea( mem( rcx, rdi, 2 ), rcx )
    mov( var( cs_c ), rdi )                // load rs_c
    lea( mem( , rdi, 4 ), rdi )            // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16( 16 )
    UPDATE_C_1X16( 17 )
    UPDATE_C_1X16( 18 )
    UPDATE_C_1X16( 19 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ(  8,  9, 10, 11 )
    UPDATE_C4_BZ( 12, 13, 14, 15 )
    UPDATE_C4_BZ( 16, 17, 18, 19 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_2X16_BZ(  8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ(  9, 13 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 10, 14 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 11, 15 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), rdi )             // load rs_c; rdi = rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c*sizeof(dt) => rdi = rs_c*4
    lea( mem( rcx, rdi, 2 ), rcx )      // c += rdi * 2
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    UPDATE_C_1X16_BZ( 16 )
    UPDATE_C_1X16_BZ( 17 )
    UPDATE_C_1X16_BZ( 18 )
    UPDATE_C_1X16_BZ( 19 )

    jmp( .SDONE )                     // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )          // load ps_b4
    mov( var( bbuf ), rbx )           // load b
    add( rdx, rbx )                   // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )           // load cs_c; rdx = cs_c
    lea( mem( , rdx, 4 ), rdx )       // rdx = cs_c*sizeof(dt) => rdx = cs_c*4
    lea( mem( , rdx, 8 ), rdx )       // rdx = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )       // rdx = rdx * 8 = cs_c * 8 * 8
                                      // => rdx = cs_c * 64
    mov( var( cbuf ), rcx )           // load address of c
    add( rdx, rcx )                   // c += rs_c * MR
    mov( rcx, var( cbuf ) )           // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
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

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( n_left )
    {
        const dim_t mr_cur = 3;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_3x48_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_3x32_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_3x16_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_3x8
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_3x4
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_3x2
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 3 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 3;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_2x64n_avx512
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

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )

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

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4( 7, 12, 13, 14, 15 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )
    UPDATE_C4( 4, 12, 13, 14, 15 )

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
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 11, 15 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ(  8,  9, 10, 11 )
    UPDATE_C4_BZ( 12, 13, 14, 15 )

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
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 11, 15 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )    // load ps_b4
    mov( var( bbuf ), rbx )     // load b
    add( rdx, rbx )             // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )
    lea( mem( , rdx, 4 ), rdx )
    lea( mem( , rdx, 8 ), rdx )     // rdx  = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )     // rdx  = rdx * 8 = cs_c * 8 * 8 => rdx = cs_c * 64
    mov( var( cbuf ), rcx )              // load address of c
    add( rdx, rcx )                    // c += rs_c * MR
    mov( rcx, var( cbuf ) )              // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
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
    if ( n_left )
    {
        const dim_t mr_cur = 2;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_2x48m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_2x32m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_2x16m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_2x8
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_2x4
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_2x2
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 2 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 2;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_1x64n_avx512
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

    uint64_t n_iter = n0 / 64;
    uint64_t n_left = n0 % 64;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of B and convert it to units of bytes.
    uint64_t ps_b   = bli_auxinfo_ps_b( data );
    uint64_t ps_b4  = ps_b * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( n_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov( var( rs_a ), r8 )          // load rs_a
    lea( mem( , r8, 4 ), r8 )       // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( rs_b ), r9 )          // load rs_b
    lea( mem( , r9, 4 ), r9 )       // rs_b *= sizeof(dt) => rs_b *= 4
    mov( var( cs_a ), r10 )         // load cs_a
    lea( mem( , r10, 4 ), r10 )     // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r8, r8, 2 ), r13 )    // r13 = 3 * rs_a
    lea( mem( r8, r8, 4 ), r15 )    // r15 = 5 * rs_a

    mov( var( n_iter ), r11 )       // load n_iter

    label( .N_LOOP_ITER )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float)

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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

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

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 3 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7,  8,  9, 10, 11 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4,  8,  9, 10, 11 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    UPDATE_C_1X16(  8 )
    UPDATE_C_1X16(  9 )
    UPDATE_C_1X16( 10 )
    UPDATE_C_1X16( 11 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C4_BZ( 8, 9, 10, 11 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    UPDATE_C_1X16_BZ(  8 )
    UPDATE_C_1X16_BZ(  9 )
    UPDATE_C_1X16_BZ( 10 )
    UPDATE_C_1X16_BZ( 11 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_b4 ), rdx )    // load ps_b4
    mov( var( bbuf ), rbx )     // load b
    add( rdx, rbx )             // b += ps_b4
    mov( rbx, var( bbuf ) )

    mov( var( cs_c ), rdx )           // load cs_c; rdx = cs_c
    lea( mem( , rdx, 4 ), rdx )       // rdx = cs_c*sizeof(dt) => rdx = cs_c*4
    lea( mem( , rdx, 8 ), rdx )       // rdx = cs_c * 8
    lea( mem( , rdx, 8 ), rdx )       // rdx = rdx * 8 = cs_c * 8 * 8
                                      // => rdx = cs_c * 64
    mov( var( cbuf ), rcx )           // load address of c
    add( rdx, rcx )                   // c += rs_c * MR
    mov( rcx, var( cbuf ) )           // store updated c

    dec( r11 )
    jne( .N_LOOP_ITER )

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
      [ps_b4]  "m" (ps_b4),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [n_iter] "m" (n_iter),
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

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( n_left )
    {
        const dim_t mr_cur = 1;
        const dim_t j_edge = n0 - ( dim_t )n_left;

        uint64_t ps_b   = bli_auxinfo_ps_b( data );

        float* restrict cij = c + j_edge*cs_c;
        float* restrict bj  = b + n_iter * ps_b;
        float* restrict ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_1x48m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_1x32m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_1x16m_avx512
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_1x8
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_1x4
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_1x2
            (
              conja,conjb,mr_cur,nr_cur,k0,
              alpha,ai,rs_a0,cs_a0,
              bj,rs_b0,cs_b0,beta,
              cij,rs_c0,cs_c0,
              data,cntx
            );
            cij += nr_cur*cs_c0;
            bj  += nr_cur*cs_b0;
            n_left -= nr_cur;
        }

        if ( 1 <= n_left )
        {
            const dim_t nr_cur = 1;
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 1 * rs_a0 )
            {
                bli_sgemv_ex
                (
                  BLIS_NO_TRANSPOSE, conjb, m0, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 2;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr_cur*rs_c0;
                    ai_ii  += ps_a0;
                }
            }
            n_left -= nr_cur;
        }
    }
}
