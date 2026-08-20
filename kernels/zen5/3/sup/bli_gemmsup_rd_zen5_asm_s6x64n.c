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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

#include "bli_gemmsup_rd_zen5_asm_s6x64.h"

void bli_sgemmsup_rd_zen5_asm_6x64n
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
    // This function handles m0 <= 6 rows of C.
    // The asm path below is used only when m0 == 6.
    uint64_t m_left = m0 % 6;

    // For m0 < 6, decompose the row edge into 3x64n, 2x64n, and finally gemv
    // for the last row.
    if ( m_left )
    {
        float* restrict cij = c;
        float* restrict bj  = b;
        float* restrict ai  = a;

        if ( 5 <= m_left )
        {
            const dim_t mr_cur = 5;

            bli_sgemmsup_rd_zen5_asm_5x64n
            (
              conja, conjb, mr_cur, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen5_asm_4x64n
            (
              conja, conjb, mr_cur, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 3 <= m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_3x64n
            (
              conja, conjb, mr_cur, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_2x64n
            (
              conja, conjb, mr_cur, n0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == m_left )
        {
            bli_sgemv_ex
            (
              BLIS_TRANSPOSE, conja, k0, n0,
              alpha, bj, rs_b0, cs_b0, ai, cs_a0,
              beta, cij, cs_c0, cntx, NULL
            );
        }
        return;
    }

    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Control reaches here when the main 4-column asm loop is skipped or when
    // only the right-edge columns remain.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        const dim_t      mr_cur = 6;
        const dim_t      j_edge = n0 - ( dim_t )n_left;

      // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        float* restrict cij = c + j_edge*cs_c;
        float* restrict ai  = a;
        float* restrict bj  = b + j_edge*cs_b;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_6x3m
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_6x2m
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles a single full 6x4 tile in the m dimension.
    if ( n_iter == 0 ) return;

    float *abuf = a;
    float *bbuf = b + ( n_iter - 1 ) * 4 * cs_b;
    float *cbuf = c + ( n_iter - 1 ) * 4 * cs_c;

    // Main 6x4 microkernel over the full top-left rectangle:
    // Handles exactly 6 rows and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    label ( .SLOOP3X4I )                                           // Single 6-row block

    mov ( var ( abuf ), rdx )                                      // load base address of a
    mov ( var ( bbuf ), r14 )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    mov ( var ( n_iter ), r15 )                                    // jj = n_iter;
    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a 
    lea ( mem ( r10, r8, 2 ), rdi )                                // rdi = 5 * rs_a 
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( r12, rcx )                                               // rcx = base of the current 6x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    prefetchw0 ( mem ( rcx, r11, 4 ) )                             // C row 4
    lea ( mem ( rcx, r11, 2 ), rax )                               // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 3
    lea ( mem ( rcx, r11, 4 ), rax )                               // rax = rcx + 4 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 5
    mov ( rdx, rax )                                               // restart A at the top of this 4-col block
    mov ( r14, rbx )                                               // restart B at the top of this 4-col block
    
    // zmm8-zmm31 accumulate a 6x4 tile.
    INIT_ACCUM_4COL

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 6 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 6x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 6 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 6x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 6 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 6x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    add ( imm ( 16*4 ), rbx )    

    label ( .CONSIDER_K_LEFT_1 )
    mov ( var ( k_left1 ), rsi )
    test ( rsi, rsi )
    je ( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case because 
    // in practice this is faster on zen5 
    cmp ( imm ( 8 ), rsi )
    jle ( .K_FLOATS_LEFT_LE_8 )

    label ( .K_FLOATS_LEFT_GT_8 )
    // Masked ZMM tail for the final 1-15 k values.
    vmovups (         mem ( rax ), ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), ZMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), ZMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, r10, 1 ), ZMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ), ZMM ( 4 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, rdi, 1 ), ZMM ( 5 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // When operating on <= 8 remaining elements, use masked YMM
    // registers for the tail path rather than handling each element
    // individually. This avoids a wasteful element-by-element loop
    // and keeps the tail processing as a single masked vector FMA
    // sequence on the remaining elements.
    vmovups (         mem ( rax ), YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), YMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, r10, 1 ), YMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ), YMM ( 4 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, rdi, 1 ), YMM ( 5 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA6 ( 7, 11, 12, 13, 23, 24, 25 )
    
    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )
    
    label ( .POST_ACCUM )

    // Preload alpha into xmm7 to avoid redundant memory reloads.
    mov ( var ( alpha ), rax )                                     // load address of alpha
    vbroadcastss ( ( rax ), xmm7 )                                 // xmm7 = alpha 
    mov ( var ( beta ), rax )                                      // load address of beta
    vbroadcastss ( ( rax ), xmm0 )
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm0 )                                        // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 6x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 6x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                     // scale the first 3 rows by alpha 

    // Preload beta into xmm9 now that zmm9 is free ( consumed by ZMM_REDUCE_4 ).
    mov ( var ( beta ), rax )
    vbroadcastss ( ( rax ), xmm9 )                                 // xmm9 = beta 
  
    C_STOR ( r11, 9, 4, 5, 6 )                                     // update the first 3 rows of the 6x4 tile

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                             // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 11 )                            // xmm11 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]
    ZMM_REDUCE_4 ( 22, 25, 28, 31, 14 )                            // xmm14 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), sum ( zmm31 )]

    ALPHA_SCALE ( 7, 8, 11, 14 )                                   // scale the next 3 rows by alpha 

    C_STOR_CONT ( r11, 9, 8, 11, 14 )                              // update the next 3 rows 

    jmp ( .SDONE )

    // Reduce the 6x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                     // scale the first 3 rows by alpha 
    
    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the first 3 rows without reading C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                             // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 11 )                            // xmm11 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]
    ZMM_REDUCE_4 ( 22, 25, 28, 31, 14 )                            // xmm14 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), sum ( zmm31 )]

    ALPHA_SCALE ( 7, 8, 11, 14 )                                   // scale the next 3 rows by alpha 

    C_STOR_BZ_CONT ( r11, 8, 11, 14 )                              // store the next 3 rows 

    label ( .SDONE )

    sub ( imm ( 4*4 ), r12 )                                       // r12 -= 4 columns 
    lea ( mem ( , r9, 4 ), rsi )                                   // rsi = 4 * cs_b_bytes
    sub ( rsi, r14 )                                               // r14 -= 4 * cs_b_bytes 

    dec ( r15 )
    jne ( .SLOOP3X4J )                                             // iterate again if ii != 0.

    end_asm (
    :                                                              // output operands ( none )
    :                                                              // input operands
      [iter_1_mask] "m" ( iter_1_mask ),
      [k_iter64] "m" ( k_iter64 ),
      [k_iter32] "m" ( k_iter32 ),
      [k_iter16] "m" ( k_iter16 ),
      [k_left1]  "m" ( k_left1 ),
      [rs_a]     "m" ( rs_a ),
      [cs_b]     "m" ( cs_b ),
      [alpha]    "m" ( alpha ),
      [beta]     "m" ( beta ),
      [rs_c]     "m" ( rs_c ),
      [n_iter]   "m" ( n_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
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
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_5x64n
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
    // This kernel handles 5 rows of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Control reaches here when the main 4-column asm loop is skipped or when
    // only the right-edge columns remain.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        const dim_t      mr_cur = 5;
        const dim_t      j_edge = n0 - ( dim_t )n_left;

      // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        float* restrict cij = c + j_edge*cs_c;
        float* restrict ai  = a;
        float* restrict bj  = b + j_edge*cs_b;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_5x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_5x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles full 5x4 tiles.
    if ( n_iter == 0 ) return;

    float *abuf = a;
    float *bbuf = b + ( n_iter - 1 ) * 4 * cs_b;
    float *cbuf = c + ( n_iter - 1 ) * 4 * cs_c;

    // Main 5x4 microkernel over the full top-left rectangle:
    // Handles 5 rows and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes
    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( abuf ), rdx )                                      // load base address of a
    mov ( var ( bbuf ), r14 )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    mov ( var ( n_iter ), r15 )                                    // jj = n_iter;
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( r12, rcx )                                               // rcx = base of the current 5x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    prefetchw0 ( mem ( rcx, r11, 4 ) )                             // C row 4
    lea ( mem ( rcx, r11, 2 ), rax )                               // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 3
    mov ( rdx, rax )                                               // restart A at the top of this 4-col block
    mov ( r14, rbx )                                               // restart B at the top of this 4-col block
    
    // zmm8-zmm19 accumulate a 5x4 tile.
    INIT_ACCUM_5x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    add ( imm ( 16*4 ), rbx )    
    
    label ( .CONSIDER_K_LEFT_1 )
    mov ( var ( k_left1 ), rsi )
    test ( rsi, rsi )
    je ( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case because 
    // in practice this is faster on zen5 
    cmp ( imm ( 8 ), rsi )
    jle ( .K_FLOATS_LEFT_LE_8 )

    label ( .K_FLOATS_LEFT_GT_8 )
    // Masked ZMM tail for the final 1-15 k values.
    vmovups (         mem ( rax ),  ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ),  ZMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ),  ZMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r10, 1 ), ZMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ),  ZMM ( 4 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // When operating on <= 8 remaining elements, use masked YMM
    // registers for the tail path rather than handling each element
    // individually. This avoids a wasteful element-by-element loop
    // and keeps the tail processing as a single masked vector FMA
    // sequence on the remaining elements.
    vmovups (         mem ( rax ),  YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ),  YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ),  YMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r10, 1 ), YMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ),  YMM ( 4 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6, 8, 9, 10, 20, 26 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 21, 27 )
    
    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6, 14, 15, 16, 23, 29 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 17, 18, 19, 24, 30 )

    label ( .POST_ACCUM )
   
    // Preload alpha into xmm7 to avoid redundant memory reloads.
    mov ( var ( alpha ), rax )                                     // load address of alpha
    vbroadcastss ( ( rax ), xmm7 )                                 // xmm7 = alpha 
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 3x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                     // scale the 3 rows by alpha 

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the 3x4 tile in C

    ZMM_REDUCE_4 (  20, 21, 23, 24, 8 )                            // xmm8  = [sum ( zmm20 ),  sum ( zmm21 ), sum ( zmm23 ), sum ( zmm24 )]
    ZMM_REDUCE_4 (  26, 27, 29, 30, 11 )                           // xmm11  = [sum ( zmm26 ),  sum ( zmm27 ), sum ( zmm29 ), sum ( zmm30 )]

    ALPHA_SCALE2 ( 7, 8, 11 )                                      // scale the next 2 rows by alpha 

    C_STOR2_CONT ( r11, 31, 8, 11 )                                // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 3x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                     // scale the 3 rows by alpha 
    
    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the 3x4 tile without reading C

    ZMM_REDUCE_4 (  20, 21, 23, 24, 8 )                            // xmm8  = [sum ( zmm20 ),  sum ( zmm21 ), sum ( zmm23 ), sum ( zmm24 )]
    ZMM_REDUCE_4 (  26, 27, 29, 30, 11 )                           // xmm11  = [sum ( zmm26 ),  sum ( zmm27 ), sum ( zmm29 ), sum ( zmm30 )]

    ALPHA_SCALE2 ( 7, 8, 11 )                                      // scale the next 2 rows by alpha 

    C_STOR_BZ2_CONT ( r11, 8, 11 )                                 // store the next 2 rows 

    label ( .SDONE )

    sub ( imm ( 4*4 ), r12 )                                       // r12 -= 4 columns 

    lea ( mem ( , r9, 4 ), rsi )                                   // rsi = 4 * cs_b_bytes
    sub ( rsi, r14 )                                               // r14 -= 4 * cs_b_bytes 

    dec ( r15 )
    jne ( .SLOOP3X4J )                                             // iterate again if jj != 0.

    end_asm (
    :                                                              // output operands ( none )
    :                                                              // input operands
      [iter_1_mask] "m" ( iter_1_mask ),
      [k_iter64] "m" ( k_iter64 ),
      [k_iter32] "m" ( k_iter32 ),
      [k_iter16] "m" ( k_iter16 ),
      [k_left1]  "m" ( k_left1 ),
      [rs_a]     "m" ( rs_a ),
      [cs_b]     "m" ( cs_b ),
      [alpha]    "m" ( alpha ),
      [beta]     "m" ( beta ),
      [rs_c]     "m" ( rs_c ),
      [n_iter]   "m" ( n_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_4x64n
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
    // This kernel handles 4 rows of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Control reaches here when the main 4-column asm loop is skipped or when
    // only the right-edge columns remain.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        const dim_t      mr_cur = 4;
        const dim_t      j_edge = n0 - ( dim_t )n_left;

      // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        float* restrict cij = c + j_edge*cs_c;
        float* restrict ai  = a;
        float* restrict bj  = b + j_edge*cs_b;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_4x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_4x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles full 4x4 tiles.
    if ( n_iter == 0 ) return;

    float *abuf = a;
    float *bbuf = b + ( n_iter - 1 ) * 4 * cs_b;
    float *cbuf = c + ( n_iter - 1 ) * 4 * cs_c;

    // Main 4x4 microkernel over the full top-left rectangle:
    // Handles 4 rows and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes
    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( abuf ), rdx )                                      // load base address of a
    mov ( var ( bbuf ), r14 )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    mov ( var ( n_iter ), r15 )                                    // jj = n_iter;
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( r12, rcx )                                               // rcx = base of the current 4x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    lea ( mem ( rcx, r11, 2 ), rax )                               // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 3
    mov ( rdx, rax )                                               // restart A at the top of this 4-col block
    mov ( r14, rbx )                                               // restart B at the top of this 4-col block
    
    // zmm8-zmm20 accumulate a 4x4 tile.
    INIT_ACCUM_4x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 4 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 4x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( ( rbx, r9, 2 ), zmm21 )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm6 )
    VFMA4 ( 6, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )    
    
    label ( .CONSIDER_K_LEFT_1 )
    mov ( var ( k_left1 ), rsi )
    test ( rsi, rsi )
    je ( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case because 
    // in practice this is faster on zen5 
    cmp ( imm ( 8 ), rsi )
    jle ( .K_FLOATS_LEFT_LE_8 )

    label ( .K_FLOATS_LEFT_GT_8 )
    // Masked ZMM tail for the final 1-15 k values.
    vmovups (         mem ( rax ),  ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ),  ZMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ),  ZMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r10, 1 ), ZMM ( 3 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6, 17, 18, 19, 29 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // When operating on <= 8 remaining elements, use masked YMM
    // registers for the tail path rather than handling each element
    // individually. This avoids a wasteful element-by-element loop
    // and keeps the tail processing as a single masked vector FMA
    // sequence on the remaining elements.
    // Perform a masked FMA operation on the remaining elements
    vmovups (         mem ( rax ),  YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ),  YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ),  YMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r10, 1 ), YMM ( 3 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6, 8, 9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )
    
    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA4 ( 21, 14, 15, 16, 26 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6, 17, 18, 19, 29 )

    label ( .POST_ACCUM )
   
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 3x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha 

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the 3x4 tile in C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                             // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]

    ALPHA_SCALE1 ( 30, 8 )                                         // scale the 4 elements by alpha 

    C_STOR1_CONT ( r11, 31, 8 )                                    // update the next row 

    jmp ( .SDONE )

    // Reduce the 3x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha 

    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the 3x4 tile without reading C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                             // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]

    ALPHA_SCALE1 ( 30, 8 )                                         // scale the 4 elements by alpha 

    C_STOR_BZ1_CONT ( r11, 8 )                                     // store the next row 

    label ( .SDONE )

    sub ( imm ( 4*4 ), r12 )                                       // r12 -= 4 columns 

    lea ( mem ( , r9, 4 ), rsi )                                   // rsi = 4 * cs_b_bytes
    sub ( rsi, r14 )                                               // r14 -= 4 * cs_b_bytes 

    dec ( r15 )
    jne ( .SLOOP3X4J )                                             // iterate again if jj != 0.

    end_asm (
    :                                                              // output operands ( none )
    :                                                              // input operands
      [iter_1_mask] "m" ( iter_1_mask ),
      [k_iter64] "m" ( k_iter64 ),
      [k_iter32] "m" ( k_iter32 ),
      [k_iter16] "m" ( k_iter16 ),
      [k_left1]  "m" ( k_left1 ),
      [rs_a]     "m" ( rs_a ),
      [cs_b]     "m" ( cs_b ),
      [alpha]    "m" ( alpha ),
      [beta]     "m" ( beta ),
      [rs_c]     "m" ( rs_c ),
      [n_iter]   "m" ( n_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )    
}

void bli_sgemmsup_rd_zen5_asm_3x64n
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
    // This kernel handles 3 rows of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Control reaches here when the main 4-column asm loop is skipped or when
    // only the right-edge columns remain.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        const dim_t      mr_cur = 3;
        const dim_t      j_edge = n0 - ( dim_t )n_left;

      // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        float* restrict cij = c + j_edge*cs_c;
        float* restrict ai  = a;
        float* restrict bj  = b + j_edge*cs_b;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_3x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_3x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles full 3x4 tiles.
    if ( n_iter == 0 ) return;

    float *abuf = a;
    float *bbuf = b + ( n_iter - 1 ) * 4 * cs_b;
    float *cbuf = c + ( n_iter - 1 ) * 4 * cs_c;

    // Main 3x4 microkernel over the full top-left rectangle:
    // Handles 3 rows and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( abuf ), rdx )                                      // load base address of a
    mov ( var ( bbuf ), r14 )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    mov ( var ( n_iter ), r15 )                                    // jj = n_iter;
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( r12, rcx )                                               // rcx = base of the current 3x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    mov ( rdx, rax )                                               // restart A at the top of this 4-col block
    mov ( r14, rbx )                                               // restart B at the top of this 4-col block
    
    // zmm8-zmm19 accumulate a 3x4 tile.
    INIT_ACCUM_3x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 3x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA3 ( 21, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )    
    
    label ( .CONSIDER_K_LEFT_1 )
    mov ( var ( k_left1 ), rsi )
    test ( rsi, rsi )
    je ( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case because 
    // in practice this is faster on zen5 
    cmp ( imm ( 8 ), rsi )
    jle ( .K_FLOATS_LEFT_LE_8 )

    label ( .K_FLOATS_LEFT_GT_8 )
    // Masked ZMM tail for the final 1-15 k values.
    vmovups (         mem ( rax ), ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), ZMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), ZMM ( 2 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 20 MASK_KZ ( 1 ) ) )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA3 ( 21, 17, 18, 19 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // When operating on <= 8 remaining elements, use masked YMM
    // registers for the tail path rather than handling each element
    // individually. This avoids a wasteful element-by-element loop
    // and keeps the tail processing as a single masked vector FMA
    // sequence on the remaining elements.
    vmovups (         mem ( rax ), YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), YMM ( 2 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA3 ( 6, 8, 9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )
    
    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 20 MASK_KZ ( 1 ) ) )
    VFMA3 ( 20, 14, 15, 16 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA3 ( 21, 17, 18, 19 )

    label ( .POST_ACCUM )
   
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 3x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the 3x4 tile in C

    jmp ( .SDONE )

    // Reduce the 3x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the 3x4 tile without reading C

    label ( .SDONE )

    sub ( imm ( 4*4 ), r12 )                                       // r12 -= 4 columns 

    lea ( mem ( , r9, 4 ), rsi )                                   // rsi = 4 * cs_b_bytes
    sub ( rsi, r14 )                                               // r14 -= 4 * cs_b_bytes 

    dec ( r15 )
    jne ( .SLOOP3X4J )                                             // iterate again if jj != 0.

    end_asm (
    :                                                              // output operands ( none )
    :                                                              // input operands
      [iter_1_mask] "m" ( iter_1_mask ),
      [k_iter64] "m" ( k_iter64 ),
      [k_iter32] "m" ( k_iter32 ),
      [k_iter16] "m" ( k_iter16 ),
      [k_left1]  "m" ( k_left1 ),
      [rs_a]     "m" ( rs_a ),
      [cs_b]     "m" ( cs_b ),
      [alpha]    "m" ( alpha ),
      [beta]     "m" ( beta ),
      [rs_c]     "m" ( rs_c ),
      [n_iter]   "m" ( n_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_2x64n
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
    // This kernel handles 2 rows of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Control reaches here when the main 4-column asm loop is skipped or when
    // only the right-edge columns remain.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        const dim_t      mr_cur = 2;
        const dim_t      j_edge = n0 - ( dim_t )n_left;

      // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        float* restrict cij = c + j_edge*cs_c;
        float* restrict ai  = a;
        float* restrict bj  = b + j_edge*cs_b;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_2x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_2x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles full 2x4 tiles.
    if ( n_iter == 0 ) return;

    float *abuf = a;
    float *bbuf = b + ( n_iter - 1 ) * 4 * cs_b;
    float *cbuf = c + ( n_iter - 1 ) * 4 * cs_c;

    // Main 2x4 microkernel over the full top-left rectangle:
    // Handles 2 rows and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( abuf ), rdx )                                      // load base address of a
    mov ( var ( bbuf ), r14 )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    mov ( var ( n_iter ), r15 )                                    // jj = n_iter;
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( r12, rcx )                                               // rcx = base of the current 2x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    mov ( rdx, rax )                                               // restart A at the top of this 4-col block
    mov ( r14, rbx )                                               // restart B at the top of this 4-col block
    
    // zmm8-zmm18 accumulate a 2x4 tile.
    INIT_ACCUM_2x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 2 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 2x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 2 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 2x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 2 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 2x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( ( rbx, r9, 2 ), zmm20 )
    VFMA2 ( 20, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm21 )
    VFMA2 ( 21, 17, 18 )

    add ( imm ( 16*4 ), rbx )    

    label ( .CONSIDER_K_LEFT_1 )
    mov ( var ( k_left1 ), rsi )
    test ( rsi, rsi )
    je ( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case because 
    // in practice this is faster on zen5 
    cmp ( imm ( 8 ), rsi )
    jle ( .K_FLOATS_LEFT_LE_8 )

    label ( .K_FLOATS_LEFT_GT_8 )
    // Masked ZMM tail for the final 1-15 k values.
    vmovups (         mem ( rax ), ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), ZMM ( 1 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA2 ( 6, 8, 9 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 20 MASK_KZ ( 1 ) ) )
    VFMA2 ( 20, 14, 15 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA2 ( 21, 17, 18 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // When operating on <= 8 remaining elements, use masked YMM
    // registers for the tail path rather than handling each element
    // individually. This avoids a wasteful element-by-element loop
    // and keeps the tail processing as a single masked vector FMA
    // sequence on the remaining elements.
    vmovups (         mem ( rax ), YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), YMM ( 1 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA2 ( 6, 8, 9 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA2 ( 7, 11, 12 )
    
    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 20 MASK_KZ ( 1 ) ) )
    VFMA2 ( 20, 14, 15 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 21 MASK_KZ ( 1 ) ) )
    VFMA2 ( 21, 17, 18 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 2x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 2x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5 <- rows 0..1, cols 0..3.
    // C_STOR2 then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR2 ( r11, 31, 4, 5 )                                      // update the 2x4 tile in C

    jmp ( .SDONE )

    // Reduce the 2x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_BZ2 ( r11, 4, 5 )                                       // store the 2x4 tile without reading C

    label ( .SDONE )

    sub ( imm ( 4*4 ), r12 )                                       // r12 -= 4 columns 
    lea ( mem ( , r9, 4 ), rsi )                                   // rsi = 4 * cs_b_bytes
    sub ( rsi, r14 )                                               // r14 -= 4 * cs_b_bytes 

    dec ( r15 )
    jne ( .SLOOP3X4J )                                             // iterate again if jj != 0.

    end_asm (
    :                                                              // output operands ( none )
    :                                                              // input operands
      [iter_1_mask] "m" ( iter_1_mask ),
      [k_iter64] "m" ( k_iter64 ),
      [k_iter32] "m" ( k_iter32 ),
      [k_iter16] "m" ( k_iter16 ),
      [k_left1]  "m" ( k_left1 ),
      [rs_a]     "m" ( rs_a ),
      [cs_b]     "m" ( cs_b ),
      [alpha]    "m" ( alpha ),
      [beta]     "m" ( beta ),
      [rs_c]     "m" ( rs_c ),
      [n_iter]   "m" ( n_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm17", "ymm18",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}
