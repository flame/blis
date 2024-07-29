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

#include "bli_gemmsup_rv_zen_s6x64.h"

#define NR 64

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
void bli_sgemmsup_rv_zen_asm_6x64m_avx512
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
        float* cij = c;
        float* bj  = b;
        float* ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen_asm_6x48m_avx512
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen_asm_6x32m_avx512
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen_asm_6x16m_avx512
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;
            bli_sgemmsup_rv_zen_asm_6x8m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;
            bli_sgemmsup_rv_zen_asm_6x4m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;
            bli_sgemmsup_rv_zen_asm_6x2m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
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

                // Since A is packed into row panels,
                // we must use a loop over gemv.
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
                    cij_ii += mr_cur * rs_c0;
                    ai_ii += ps_a0;
                }
            }
            n_left -= nr_cur;
        }

        if ( n0 / NR == 0 )
        {
            return;
        }
    }

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    mov( var( m_iter ), r11 )       // load m_iter


    label( .M_LOOP_ITER )

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    // C Prefetch
    cmp( imm( 4 ), rdi )
    jz( .SPOSTPFETCH )  // haven't added col-prefetch cases


    label( .SROWPFETCH )
    lea( mem( rcx, rdi, 2 ), rdx )
    lea( mem( rdx, rdi, 1 ), rdx )

    prefetch( 0, mem( rcx,         7*8 ) )
    prefetch( 0, mem( rcx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rcx, rdi, 2, 7*8 ) )
    prefetch( 0, mem( rdx,         7*8 ) )
    prefetch( 0, mem( rdx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rdx, rdi, 2, 7*8 ) )


    label( .SPOSTPFETCH )

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
    VFMA4( 4,  8,  9, 10, 11)
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
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
    VFMA4( 4,  8,  9, 10, 11)
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
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
    VFMA4( 4,  8,  9, 10, 11)
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
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
    VFMA4( 4,  8,  9, 10, 11)
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop

    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )              // i = k_left;
	  test( rsi, rsi )                     // check i via logical AND.
	  je( .SPOSTACCUM )                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

    label( .K_LEFT_LOOP )
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11)
    vbroadcastss( mem( rax,  r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
    vbroadcastss( mem( rax,  r8, 2 ), zmm6 )
    VFMA4( 6, 16, 17, 18, 19 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA4( 4, 20, 21, 22, 23 )
    vbroadcastss( mem( rax,  r8, 4 ), zmm5 )
    VFMA4( 5, 24, 25, 26, 27 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA4( 6, 28, 29, 30, 31 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )                           // i -= 1;
	  jne( .K_LEFT_LOOP )                   // iterate again if i != 0.

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
     *
     * |-----------------------------------|       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |        |        |        |        |       |       16x4       |  16x2  |
     * |  4x16  |  4x16  |  4x16  |  4x16  |       |                  |        |
     * |        |        |        |        |       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |-----------------------------------|  ->   |       16x4       |  16x2  |
     * |        |        |        |        |       |                  |        |
     * |  2x16  |  2x16  |  2x16  |  2x16  |       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |-----------------------------------|       |       16x4       |  16x2  |
     *                                             |                  |        |
     *                                             |------------------|--------|
     *                                             |                  |        |
     *                                             |       16x4       |  16x2  |
     *                                             |                  |        |
     *                                             |------------------|--------|
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 11, 15, 19, 23 )
    add( rdi, rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c


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

    UPDATE_C4_BZ( 8, 9, 10, 11 )
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
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 11, 15, 19, 23 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 25, 29 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 26, 30 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 27, 31 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_a4 ), rdx )            // load panel stride of a; rdx = ps_a4
    mov( var( abuf ), rax )             // load address of a
    add( rdx, rax )                     // a += ps_a4
    mov( rax, var( abuf ) )             // store updated a

    mov( var( rs_c ), rdi )             // load rs_c; rdi = rs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem(    , rdi, 2 ), rdx )      // rdx = rs_c * 2
    lea( mem( rdx, rdi, 4 ), rdx )      // rdx = rdi * 4 => rdx = rs_c * 6
    mov( var( cbuf ), rcx )             // load address of c
    add( rdx, rcx )                     // c += rs_c * 6(MR)
    mov( rcx, var( cbuf ) )             // store updated c

    dec( r11 )
    jne( .M_LOOP_ITER )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
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
      [m_iter] "m" (m_iter),
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
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge * rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen_asm_4x64m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen_asm_2x64m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen_asm_1x64m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_6x48m_avx512
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

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    mov( var( m_iter ), r11 )       // load m_iter


    label( .M_LOOP_ITER )

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    // C Prefetch
    lea( mem( rcx, rdi, 2 ), rdx )
    lea( mem( rdx, rdi, 1 ), rdx )

    cmp( imm( 4 ), rdi )
    jz( .SPOSTPFETCH )  // haven't added col-prefetch cases


    label( .SROWPFETCH )
    prefetch( 0, mem( rcx,         7*8 ) )
    prefetch( 0, mem( rcx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rcx, rdi, 2, 7*8 ) )
    prefetch( 0, mem( rdx,         7*8 ) )
    prefetch( 0, mem( rdx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rdx, rdi, 2, 7*8 ) )


    label( .SPOSTPFETCH )

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

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA3( 6, 28, 29, 30 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA3( 6, 28, 29, 30 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA3( 6, 28, 29, 30 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA3( 6, 28, 29, 30 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop


    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )              // i = k_left;
	  test( rsi, rsi )                     // check i via logical AND.
	  je( .SPOSTACCUM )                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

    label( .K_LEFT_LOOP )
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA3( 5, 24, 25, 26 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA3( 6, 28, 29, 30 )

    add(  r9, rbx )
    add( r10, rax )

    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label(.SPOSTACCUM)
    // Scaling A * B with alpha.
    ALPHA_SCALE3( 7, 8, 9, 10 )
    ALPHA_SCALE3( 7, 12, 13, 14 )
    ALPHA_SCALE3( 7, 16, 17, 18 )
    ALPHA_SCALE3( 7, 20, 21, 22 )
    ALPHA_SCALE3( 7, 24, 25, 26 )
    ALPHA_SCALE3( 7, 28, 29, 30 )

    mov( var( beta ), rdx )     // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )      // check if beta = 0
    je( .SBETAZERO )            // jump to beta = 0 case

    cmp( imm( 4 ), rdi )        // set ZF of (4*rs_c) == 4.
    jz( .SCOLSTORED )           // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C3( 4, 8, 9, 10 )
    UPDATE_C3( 4, 12, 13, 14 )
    UPDATE_C3( 4, 16, 17, 18 )
    UPDATE_C3( 4, 20, 21, 22 )
    UPDATE_C3( 4, 24, 25, 26 )
    UPDATE_C3( 4, 28, 29, 30 )

    jmp( .SDONE )               // jump to the end

    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )

    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c
   
    TRANSPOSE_2X16( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 25, 29 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 26, 30 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ( 8, 9, 10 )
    UPDATE_C3_BZ( 12, 13, 14 )
    UPDATE_C3_BZ( 16, 17, 18 )
    UPDATE_C3_BZ( 20, 21, 22 )
    UPDATE_C3_BZ( 24, 25, 26 )
    UPDATE_C3_BZ( 28, 29, 30 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    /* Transposing 4x16 tiles to 16x4 tiles */
    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 25, 29 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 26, 30 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_a4 ), rdx )            // load panel stride of a
    mov( var( abuf ), rax )             // load address of a
    add( rdx, rax )                     // a += ps_a4
    mov( rax, var( abuf ) )             // store updated a

    mov( var( rs_c ), rdi )             // load rs_c; rdi = rs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem(    , rdi, 2 ), rdx )      // rdx = rs_c * 2
    lea( mem( rdx, rdi, 4 ), rdx )      // rdx = rdi * 4 => rdx = rs_c * 6
    mov( var( cbuf ), rcx )             // load address of c
    add( rdx, rcx )                     // c += rs_c * 6(MR)
    mov( rcx, var( cbuf ) )             // store updated c

    dec( r11 )
    jne( .M_LOOP_ITER )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
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
      [m_iter] "m" (m_iter),
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
    if (m_left)
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen_asm_4x48m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen_asm_2x48m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen_asm_1x48m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_6x32m_avx512
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

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    mov( var( m_iter ), r11 )       // load m_iter


    label( .M_LOOP_ITER )

    INIT_REG

    mov( var(abuf), rax )   // load address of a
    mov( var(bbuf), rbx )   // load address of b
    mov( var(cbuf), rcx )   // load address of c

    // C Prefetch
    lea( mem( rcx, rdi, 2 ), rdx )
    lea( mem( rdx, rdi, 1 ), rdx )

    cmp( imm( 4 ), rdi )
    jz( .SPOSTPFETCH )  // haven't added col-prefetch cases


    label( .SROWPFETCH )
    prefetch( 0, mem( rcx,         7*8 ) )
    prefetch( 0, mem( rcx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rcx, rdi, 2, 7*8 ) )
    prefetch( 0, mem( rdx,         7*8 ) )
    prefetch( 0, mem( rdx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rdx, rdi, 2, 7*8 ) )


    label( .SPOSTPFETCH )

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )   // load k_iter
    test( rsi,rsi )
    je( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA2( 6, 28, 29 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA2( 6, 28, 29 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA2( 6, 28, 29 )

    add( r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA2( 6, 28, 29 )

    add( r9, rbx )
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

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA2( 5, 24, 25 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA2( 6, 28, 29 )

    add( r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop

    label(.SPOSTACCUM)
    // Scaling A * B with alpha.
    ALPHA_SCALE2( 7, 8, 9 )
    ALPHA_SCALE2( 7, 12, 13 )
    ALPHA_SCALE2( 7, 16, 17 )
    ALPHA_SCALE2( 7, 20, 21 )
    ALPHA_SCALE2( 7, 24, 25 )
    ALPHA_SCALE2( 7, 28, 29 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C2( 4, 8, 9 )
    UPDATE_C2( 4, 12, 13 )
    UPDATE_C2( 4, 16, 17 )
    UPDATE_C2( 4, 20, 21 )
    UPDATE_C2( 4, 24, 25 )
    UPDATE_C2( 4, 28, 29 )
    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load cs_c; rdi = cs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = cs_c*sizeof(dt) => rdi = cs_c*4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * cs_c

    /* Transposing 4x16 tiles to 16x4 tiles */
    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c


    TRANSPOSE_2X16( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 25, 29 )

    jmp( .SDONE )                       // jump to the end


    label(.SBETAZERO)

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ( 8, 9 )
    UPDATE_C2_BZ( 12, 13 )
    UPDATE_C2_BZ( 16, 17 )
    UPDATE_C2_BZ( 20, 21 )
    UPDATE_C2_BZ( 24, 25 )
    UPDATE_C2_BZ( 28, 29 )

    jmp( .SDONE )                       // jump to the end


    label(.SCOLSTORBZ)
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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ( 24, 28 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 25, 29 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_a4 ), rdx )            // load panel stride of a
    mov( var( abuf ), rax )             // load address of a
    add( rdx, rax )                     // a += ps_a4
    mov( rax, var( abuf ) )             // store updated a

    mov( var( rs_c ), rdi )             // load rs_c; rdi = rs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem(    , rdi, 2 ), rdx )      // rdx = rs_c * 2
    lea( mem( rdx, rdi, 4 ), rdx )      // rdx = rdi * 4 => rdx = rs_c * 6
    mov( var( cbuf ), rcx )             // load address of c
    add( rdx, rcx )                     // c += rs_c * 6(MR)
    mov( rcx, var( cbuf ) )             // store updated c

    dec( r11 )
    jne( .M_LOOP_ITER )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
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
      [m_iter] "m" (m_iter),
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
    if (m_left)
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen_asm_4x32m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen_asm_2x32m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen_asm_1x32m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_6x16m_avx512
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

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    mov( var( m_iter ), r11 )       // load m_iter


    label( .M_LOOP_ITER )

    INIT_REG

    mov( var( abuf ), rax )         // load address of a
    mov( var( bbuf ), rbx )         // load address of b
    mov( var( cbuf ), rcx )         // load address of c

    // C Prefetch
    lea( mem( rcx, rdi, 2 ), rdx )
    lea( mem( rdx, rdi, 1 ), rdx )

    cmp( imm( 4 ), rdi )
    jz( .SPOSTPFETCH )  // haven't added col-prefetch cases


    label( .SROWPFETCH )
    prefetch( 0, mem( rcx,         7*8 ) )
    prefetch( 0, mem( rcx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rcx, rdi, 2, 7*8 ) )
    prefetch( 0, mem( rdx,         7*8 ) )
    prefetch( 0, mem( rdx, rdi, 1, 7*8 ) )
    prefetch( 0, mem( rdx, rdi, 2, 7*8 ) )


    label( .SPOSTPFETCH )

    mov( var( alpha ), rdx )        // load address of alpha
    vbroadcastss( ( rdx ), zmm7 )

    mov( var( k_iter ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSID_K_LEFT )


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LOOP_ITER )
    // ITER 0
    // Load a row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA1( 5, 24 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA1( 6, 28 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load a row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA1( 5, 24 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA1( 6, 28 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load a row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA1( 5, 24 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA1( 6, 28 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load a row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA1( 5, 24 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA1( 6, 28 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LOOP_ITER )     // if rsi != 0, repeat k-loop

    label( .CONSID_K_LEFT )

    mov( var( k_left ), rsi )              // i = k_left;
	  test( rsi, rsi )                     // check i via logical AND.
	  je( .SPOSTACCUM )                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

    label( .K_LEFT_LOOP )
    // Load a row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 6 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )
    vbroadcastss( mem( rax, r8, 4 ), zmm5 )
    VFMA1( 5, 24 )
    vbroadcastss( mem( rax, r15, 1 ), zmm6 )
    VFMA1( 6, 28 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop

    label(.SPOSTACCUM)
    // Scaling A * B with alpha.
    ALPHA_SCALE1( 7, 8 )
    ALPHA_SCALE1( 7, 12 )
    ALPHA_SCALE1( 7, 16 )
    ALPHA_SCALE1( 7, 20 )
    ALPHA_SCALE1( 7, 24 )
    ALPHA_SCALE1( 7, 28 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C1( 4, 8 )
    UPDATE_C1( 4, 12 )
    UPDATE_C1( 4, 16 )
    UPDATE_C1( 4, 20 )
    UPDATE_C1( 4, 24 )
    UPDATE_C1( 4, 28 )

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

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16( 24, 28 )

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
    UPDATE_C1_BZ( 28 )

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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( rs_c ), r12 )             // load rs_c; r12 = rs_c
    lea( mem( , r12, 4 ), r12 )         // r12 = rs_c*sizeof(dt) => r12 = rs_c*4
    lea( mem( rcx, r12, 4 ), rcx )      // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ( 24, 28 )

    jmp( .SDONE )                       // jump to the end


    label( .SDONE )

    mov( var( ps_a4 ), rdx )            // load panel stride of a
    mov( var( abuf ), rax )             // load address of a
    add( rdx, rax )                     // a += ps_a4
    mov( rax, var( abuf ) )             // store updated a

    mov( var( rs_c ), rdi )             // load rs_c; rdi = rs_c
    lea( mem(    , rdi, 4 ), rdi )      // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem(    , rdi, 2 ), rdx )      // rdx = rs_c * 2
    lea( mem( rdx, rdi, 4 ), rdx )      // rdx = rdi * 4 => rdx = rs_c * 6
    mov( var( cbuf ), rcx )             // load address of c
    add( rdx, rcx )                     // c += rs_c * 6(MR)
    mov( rcx, var( cbuf ) )             // store updated c

    dec( r11 )
    jne( .M_LOOP_ITER )

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
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
      [m_iter] "m" (m_iter),
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
    if (m_left)
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter*ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen_asm_4x16m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen_asm_2x16m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen_asm_1x16m_avx512
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen_asm_4x64m_avx512
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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
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
    VFMA4( 4,  8,  9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
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
    VFMA4( 4,  8,  9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
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
    VFMA4( 4,  8,  9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
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


    // The k-loop iterates over 4 rows of B, and broadcasts of each row of A.
    label( .K_LEFT_LOOP )

    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA4( 5, 12, 13, 14, 15)
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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 11, 15, 19, 23 )

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
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_4x48m_avx512
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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )

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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA3( 6, 16, 17, 18 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA3( 4, 20, 21, 22 )

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

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C3( 4, 8, 9, 10 )
    UPDATE_C3( 4, 12, 13, 14 )
    UPDATE_C3( 4, 16, 17, 18 )
    UPDATE_C3( 4, 20, 21, 22 )
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

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 10, 14, 18, 22 )
    lea( mem( rcx, r12, 4 ), rcx )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ( 8, 9, 10 )
    UPDATE_C3_BZ( 12, 13, 14 )
    UPDATE_C3_BZ( 16, 17, 18 )
    UPDATE_C3_BZ( 20, 21, 22 )

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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 10, 14, 18, 22 )

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
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_4x32m_avx512
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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )

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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA2( 6, 16, 17 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA2( 4, 20, 21 )

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

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C2( 4, 8, 9 )
    UPDATE_C2( 4, 12, 13 )
    UPDATE_C2( 4, 16, 17 )
    UPDATE_C2( 4, 20, 21 )

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

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16( 9, 13, 17, 21 )
    lea( mem( rcx, r12, 4 ), rcx )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ( 8, 9 )
    UPDATE_C2_BZ( 12, 13 )
    UPDATE_C2_BZ( 16, 17 )
    UPDATE_C2_BZ( 20, 21 )

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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ( 9, 13, 17, 21 )

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
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_4x16m_avx512
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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )

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

    // Broadcast 4 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )
    vbroadcastss( mem( rax, r8, 2 ), zmm6 )
    VFMA1( 6, 16 )
    vbroadcastss( mem( rax, r13, 1 ), zmm4 )
    VFMA1( 4, 20 )

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

    TRANSPOSE_4X16( 8, 12, 16, 20 )
    lea( mem( rcx, r12, 4 ), rcx )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C1_BZ( 8 )
    UPDATE_C1_BZ( 12 )
    UPDATE_C1_BZ( 16 )
    UPDATE_C1_BZ( 20 )

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

    TRANSPOSE_4X16_BZ( 8, 12, 16, 20 )

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
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x64m_avx512
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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4,  8,  9, 10, 11 )
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

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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

    TRANSPOSE_2X16_BZ(  8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ(  9, 13 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 10, 14 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 11, 15 )

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
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x48m_avx512
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )

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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4,  8,  9, 10 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA3( 5, 12, 13, 14 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3( 7,  8,  9, 10 )
    ALPHA_SCALE3( 7, 12, 13, 14 )

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

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
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

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ(  8, 9, 10 )
    UPDATE_C3_BZ( 12, 13, 14 )

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

    TRANSPOSE_2X16_BZ(  8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ(  9, 13 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 10, 14 )

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
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x32m_avx512
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )

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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4,  8,  9 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA2( 5, 12, 13 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2( 7,  8,  9 )
    ALPHA_SCALE2( 7, 12, 13 )

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

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16( 8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16( 9, 13 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ(  8,  9 )
    UPDATE_C2_BZ( 12, 13 )

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

    TRANSPOSE_2X16_BZ( 8, 12 )
    lea( mem( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ( 9, 13 )

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
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x16m_avx512
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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )

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

    // Broadcast 2 elements from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4,  8 )
    vbroadcastss( mem( rax, r8, 1 ), zmm5 )
    VFMA1( 5, 12 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1( 7,  8 )
    ALPHA_SCALE1( 7, 12 )

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

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16( 8, 12 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C1_BZ(  8 )
    UPDATE_C1_BZ( 12 )

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

    TRANSPOSE_2X16_BZ( 8, 12 )
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
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_1x64m_avx512
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
    // Load 4 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )
    vmovups( 0xc0( rbx ), zmm3 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA4( 4, 8, 9, 10, 11 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4( 7, 8, 9, 10, 11 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C4( 4, 8, 9, 10, 11 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

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

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ(  8 )
    UPDATE_C_1X16_BZ(  9 )
    UPDATE_C_1X16_BZ( 10 )
    UPDATE_C_1X16_BZ( 11 )

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

void bli_sgemmsup_rv_zen_asm_1x48m_avx512
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )
    vmovups( 0x80( rbx ), zmm2 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )

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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA3( 4, 8, 9, 10 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3( 7, 8, 9, 10 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C3( 4, 8, 9, 10 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16(  8 )
    UPDATE_C_1X16(  9 )
    UPDATE_C_1X16( 10 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C3_BZ( 8, 9, 10 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ(  8 )
    UPDATE_C_1X16_BZ(  9 )
    UPDATE_C_1X16_BZ( 10 )

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

void bli_sgemmsup_rv_zen_asm_1x32m_avx512
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups(     ( rbx ), zmm0 )
    vmovups( 0x40( rbx ), zmm1 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )

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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA2( 4, 8, 9 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2( 7, 8, 9 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C2( 4, 8, 9 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16( 8 )
    UPDATE_C_1X16( 9 )
    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C2_BZ( 8, 9 )
    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 8 )
    UPDATE_C_1X16_BZ( 9 )

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

void bli_sgemmsup_rv_zen_asm_1x16m_avx512
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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 1
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 2
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )

    add(  r9, rbx )
    add( r10, rax )

    // ITER 3
    // Load 1 row from B matrix.
    vmovups( ( rbx ), zmm0 )

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )

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

    // Broadcast 1 element from a row of A & do VFMA with rows of B.
    vbroadcastss( ( rax ), zmm4 )
    VFMA1( 4, 8 )

    add(  r9, rbx )
    add( r10, rax )
    dec( rsi )
    jne( .K_LEFT_LOOP )     // if rsi != 0, repeat k-loop


    label( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1( 7, 8 )

    mov( var( beta ), rdx )         // load address of beta
    vbroadcastss( ( rdx ), zmm4 )

    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm4 )          // check if beta = 0
    je( .SBETAZERO )                // jump to beta = 0 case

    cmp( imm(4), rdi )              // set ZF if (4*rs_c) == 4
    jz( .SCOLSTORED )               // jump to column storage case


    label( .SROWSTORED )

    UPDATE_C1( 4, 8 )

    jmp( .SDONE )               // jump to the end


    label( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rdi = rs_c *= sizeof(dt) => rs_c *= 4
    lea( mem( rdi, rdi, 2 ), r12 )      // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16( 8 )

    jmp( .SDONE )                       // jump to the end


    label( .SBETAZERO )

    cmp( imm( 4 ), rdi )                // set ZF if (4*rs_c) == 4.
    jz( .SCOLSTORBZ )                   // jump to column storage case


    label( .SROWSTORBZ )

    UPDATE_C1_BZ( 8 )

    jmp( .SDONE )                       // jump to the end


    label( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov( var( cbuf ), rcx )             // load address of c
    mov( var( cs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float)
    lea( mem( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ( 8 )

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
