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

void bli_sgemmsup_rd_zen5_asm_6x64m
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
    // Main loop handles 4 columns at a time; 1-3 columns fall through to n_left dispatch.
    uint64_t n_left = n0 % 4;

    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;
    uint64_t b_step_base = 0;

    uint64_t n_iter = n0 / 4;
    uint64_t n_main_loop = n0 - n_left;
    uint64_t j_step_end = 0;
    uint64_t j_step_start = n_main_loop >= 1 ? ( ( n_main_loop - 1 ) / 16 ) * 16 : 0;

    // Main loop handles 6 rows at a time; 1-5 rows fall through to m_left dispatch.
    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;
    // The decomposition into edge kernels is handled as follows:
    // Example, m0 = 7 and n0 = 5:
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [ asm ][ asm ][ asm ][ asm ][ n_left ]
    //   [          m_left          ][ n_left ]
    // The asm handles all multiples of 6 rows and 4 columns in the top-left of the matrix
    // Next, any remaining m_left rows are handled by a separate edge microkernel after the
    // assembly section which still processes 4 columns at a time in the n dimension.
    // Finally, any remaining n_left columns are handled by a separate edge microkernel
    // that processes the remaining right side columns.
    // n_left dispatch: handle the right edge columns that remain after the asm
    // kernel covers all full 4-column blocks.
    // This path owns the entire right-edge strip across rows [0, m0 ), including
    // the bottom-right corner that the x64 m_left kernels intentionally skip.
    // Computes the n_left part of C.
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    //   [ - ][ - ][ - ][ - ][ n_left ]
    if ( n_left )
    {
        float* restrict cij = c;
        float* restrict bj  = b;
        float* restrict ai  = a;

        // Start at column 4 * n_iter, i.e. the first column not covered by asm.
        cij += n_main_loop*cs_c0;
        bj  += n_main_loop*cs_b0;

        if ( 3 == n_left )
        {
            const dim_t nr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_6x3m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_6x2m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, m0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
    }

    // The asm microkernel only handles full 6x4 tiles.
    // If either dimension is already in edge territory, skip straight to the
    // C dispatch below.
    if ( m_iter == 0 || n_iter == 0 ) goto consider_edge_cases;

    // Main 6x4 microkernel over the full top-left rectangle:
    // Handles all rows in multiples of 6, and columns in multiples of 4.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [ asm ][ asm ][ asm ][ asm ][ - ]
    //   [          -               ][ - ]
    begin_asm()

    mov ( var ( rs_a ), r8 )                                         // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                      // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                         // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                      // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                   // r13 = 3 * cs_b in bytes
    lea ( mem (  r8, r8, 2 ), r10 )                                  // r10 = 3 * rs_a

    mov ( var ( iter_1_mask ), esi )                                 // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    // Outer J-step loop: process 16 columns ( 4 groups of 4 ) at a time.
    // For each 16-col step, sweep all m_iter row-blocks ( full SLOOP3X4I ),
    // then advance to next 16-col step. This is done so that writes happen to the full
    // cache line in the C matrix
    label ( .SLOOP3X4J_STEP )                                        // Outer loop: 16-col steps
    mov ( var ( j_step_start ), r15 )                                // r15 = j_step_start
    mov ( imm ( 16 ), rsi )                                          // rsi = 16
    add ( r15, rsi )                                                 // rsi = j_step_start + 16 ( desired end )
    mov ( var ( n_main_loop ), rax )                                 // rax = n_main_loop ( max bound )
    cmp ( rax, rsi )                                                 // compare rsi vs n_main_loop
    cmovg ( rax, rsi )                                               // if rsi > n_main_loop, rsi = n_main_loop
    mov ( rsi, var ( j_step_end ) )                                  // j_step_end = min ( j_step_start + 16, n_main_loop )

    mov ( var ( m_iter ), r11 )                                      // ii = m_iter;
    mov ( var ( abuf ), r14 )                                        // load base address of a
    mov ( var ( cbuf ), r12 )                                        // load base address of c

    // Compute B base for this 16-col step once, outside the I-loop.
    mov ( var ( bbuf ), rdx )
    sub ( imm ( 4 ), rsi )                                           // rsi = j_step_end - 4
    imul ( r9, rsi )                                                 // rsi = ( j_step_end - 4 ) * cs_b_bytes
    add ( rsi, rdx )                                                 // rdx = B base for rightmost 4-col group
    mov ( rdx, var ( b_step_base ) )                                 // save B base for I-loop reuse

    label ( .SLOOP3X4I )                                             // Middle loop over 6-row blocks of C/A

    // B base for this 16-col step: bbuf + j_step_start * cs_b_bytes
    mov ( var ( b_step_base ), rdx )                                 // reload B base for the beginning of the I-loop
    mov ( var ( j_step_end ), r15 )
    sub ( imm ( 4 ), r15 )                                           // r15 = j_step_end - 4

    label ( .SLOOP3X4J )                                             // Inner loop over 4-column output tiles

    // rcx = base of the current 6x4 output tile in C.
    lea ( mem ( r12, r15, 4 ), rcx )                                 // rcx = c_row_base + jj * sizeof ( float )

    // Prefetch the rows of C that will be written to in this loop
    mov ( var ( rs_c ), rsi )                                        // rsi = rs_c
    lea ( mem ( , rsi, 4 ), rsi )                                    // rsi = rs_c * sizeof ( float )
    prefetchw0 ( mem ( rcx ) )                                       // C row 0
    prefetchw0 ( mem ( rcx, rsi, 1 ) )                               // C row 1
    prefetchw0 ( mem ( rcx, rsi, 2 ) )                               // C row 2
    prefetchw0 ( mem ( rcx, rsi, 4 ) )                               // C row 4
    lea ( mem ( rcx, rsi, 2 ), rax )                                 // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 3 = rcx + 3 * rs_c
    lea ( mem ( rcx, rsi, 4 ), rax )                                 // rax = rcx + 4 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 5 = rcx + 5 * rs_c

    mov ( r14, rax )                                                 // restart A at the top of this row-block
    mov ( rdx, rbx )                                                 // load b to rbx
    lea ( mem ( r10, r8, 2 ), rdi )                                  // rdi = 5 * rs_a

    // zmm8-zmm31 accumulate a 6x4 tile.
    INIT_ACCUM_4COL

    mov ( var ( k_iter64 ), rsi )                                    // number of 64-float k blocks
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

    mov ( var ( k_iter32 ), rsi )                                    // number of remaining 32-float k blocks
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
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

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

    vmovups (  mem ( rbx, r9, 2 ), ZMM ( 6 MASK_KZ ( 1 ) ) )
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

    vmovups (  mem ( rbx, r9, 2 ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6, 14, 15, 16, 26, 27, 28 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA6 ( 7, 17, 18, 19, 29, 30, 31 )

    label ( .POST_ACCUM )

    // Preload alpha into xmm7 to avoid redundant memory reloads.
    // xmm7 survives ZMM_REDUCE calls ( they only clobber zmm0-3 ).
    mov ( var ( alpha ), rax )                                       // load address of alpha
    vbroadcastss ( ( rax ), xmm7 )                                   // xmm7 = alpha 
    mov ( var ( beta ), rax )                                        // load address of beta
    vbroadcastss ( ( rax ), xmm0 )
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm0 )                                          // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 6x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 6x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                               // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                               // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                               // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                       // scale the first 3 rows by alpha 

    // Preload beta into xmm9 now that zmm9 is free ( consumed by ZMM_REDUCE_4 ).
    mov ( var ( beta ), rax )
    vbroadcastss ( ( rax ), xmm9 )                                   // xmm9 = beta 
    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )

    C_STOR ( rdi, 9, 4, 5, 6 )                                       // update the first 3 rows of the 6x4 tile

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                               // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 11 )                              // xmm11 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]
    ZMM_REDUCE_4 ( 22, 25, 28, 31, 14 )                              // xmm14 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), sum ( zmm31 )]

    ALPHA_SCALE ( 7, 8, 11, 14 )                                     // scale the next 3 rows by alpha 

    C_STOR_CONT ( rdi, 9, 8, 11, 14 )                                // update the next 3 rows 

    jmp ( .SDONE )

    // Reduce the 6x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                               // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                               // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                               // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 7, 4, 5, 6 )                                       // scale the first 3 rows by alpha 

    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )
    C_STOR_BZ ( rdi, 4, 5, 6 )                                       // store the first 3 rows without reading C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 8 )                               // xmm8  = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 11 )                              // xmm11 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]
    ZMM_REDUCE_4 ( 22, 25, 28, 31, 14 )                              // xmm14 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), sum ( zmm31 )]

    ALPHA_SCALE ( 7, 8, 11, 14 )                                     // scale the next 3 rows by alpha 

    C_STOR_BZ_CONT ( rdi, 8, 11, 14 )                                // store the next 3 rows 

    label ( .SDONE )

    // Advance to the previous 4-column block of B/C ( R-to-L within block ).
    lea ( mem ( , r9, 4 ), rsi )                                     // rsi = 4 * cs_b_bytes
    sub ( rsi, rdx )                                                 // rdx -= 4 * cs_b
    sub ( imm ( 4 ), r15 )
    cmp ( var ( j_step_start ), r15 )
    jge ( .SLOOP3X4J )                                               // iterate within current 16-col step

    // Finished all 4 J-iterations for this row-block; step A and C down by 6 rows.
    mov ( var ( rs_c ), rdi )                                        // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )                                    // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r12, rdi, 2 ), r12 )
    lea ( mem ( r12, rdi, 4 ), r12 )                                 // c_ii = r12 += 6*rs_c

    lea ( mem ( r14, r8,  2 ), r14 )
    lea ( mem ( r14, r8,  4 ), r14 )                                 // a_ii = r14 += 6*rs_a

    dec ( r11 )
    jne ( .SLOOP3X4I )                                               // iterate again if ii != 0.

    // Advance j_step_start by 16 and loop if more columns remain.
    mov ( var ( j_step_start ), r15 )
    sub ( imm ( 16 ), r15 )
    mov ( r15, var ( j_step_start ) )
    test ( r15, r15 )
    jns ( .SLOOP3X4J_STEP )                                          // loop while j_step_start >= 0 i.e. if it is not-signed

    end_asm (
    :                                                                // output operands
    :                                                                // input operands
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
      [n_main_loop]   "m" ( n_main_loop ),
      [m_iter]   "m" ( m_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf ),
      [j_step_start] "m" ( j_step_start ),
      [j_step_end]   "m" ( j_step_end ),
      [b_step_base]  "m" ( b_step_base )
    :                                                                // register clobber list
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

    consider_edge_cases:

    // Control reaches here in two cases:
    // 1. asm finished the full-tile rectangle and only edge work remains.
    // 2. asm was skipped because m0 < 6 or n0 < 4.
    // m_left dispatch: handle the bottom edge rows that remain after the asm
    // kernel covers all full 6-row blocks.
    // The x64 callees receive n0, but internally they only compute the 4-col
    // columns [0, 4 * n_iter ). They do not consume the 1-3 column tail.
    // Computes the m_left part of C.
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [ - ][ - ][ - ][ - ][ - ]
    //   [      m_left      ][ - ]
    if ( m_left && n_iter )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        // Start at row 6 * m_iter, i.e. the first row not covered by asm.
        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen5_asm_5x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen5_asm_4x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_3x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_2x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen5_asm_1x64
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen5_asm_6x3m
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
    // This kernel handles 6 rows and 3 columns of C.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 6 rows at a time; 1-5 rows fall through to m_left dispatch.
    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // The asm microkernel only handles full 6x3 tiles.
    // If m0 < 6, skip straight to the C dispatch below.
    if ( m_iter == 0 ) goto consider_edge_cases;

    // Main 6x3 microkernel over the full 6x3 rectangle:
    // Handles all rows in multiples of 6.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ][ asm ]
    //   [ asm ][ asm ][ asm ]
    //   [ asm ][ asm ][ asm ]
    //   [ asm ][ asm ][ asm ]
    //   [ asm ][ asm ][ asm ]
    //   [ asm ][ asm ][ asm ]
    //   [      m_left      ]
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                                         // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                      // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                         // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                      // cs_b *= sizeof ( dt ) => cs_b *= 4

    mov ( var ( iter_1_mask ), esi )                                 // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )
    kmovw ( esi, K ( 2 ) )

    mov ( var ( abuf ), r14 )                                        // load base address of a
    mov ( var ( bbuf ), rdx )                                        // load base address of b
    mov ( var ( cbuf ), r12 )                                        // load base address of c

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                       // load address of alpha
    vbroadcastss ( ( rsi ), xmm29 )                                  // xmm29 <- alpha 
    mov ( var ( beta ), rsi )                                        // load address of beta
    vbroadcastss ( ( rsi ), xmm30 )                                  // xmm30 <- beta 

    lea ( mem ( r8, r8, 2 ), r10 )                                   // r10 = 3 * rs_a 
    mov ( var ( m_iter ), r11 )                                      // ii = m_iter;
    label ( .SLOOP3X4I )                                             // Outer loop over 6-row blocks of C/A

    mov ( r12, rcx )                                                 // rcx = base of the current 6x3 output tile in C

    // Prefetch the rows of C that will be written to in this loop
    mov ( var ( rs_c ), rsi )                                        // rsi = rs_c
    lea ( mem ( , rsi, 4 ), rsi )                                    // rsi = rs_c * sizeof ( float )
    prefetchw0 ( mem ( rcx ) )                                       // C row 0
    prefetchw0 ( mem ( rcx, rsi, 1 ) )                               // C row 1
    prefetchw0 ( mem ( rcx, rsi, 2 ) )                               // C row 2
    prefetchw0 ( mem ( rcx, rsi, 4 ) )                               // C row 4
    lea ( mem ( rcx, rsi, 2 ), rax )                                 // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 3 = rcx + 3 * rs_c
    lea ( mem ( rcx, rsi, 4 ), rax )                                 // rax = rcx + 4 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 5 = rcx + 5 * rs_c

    mov ( r14, rax )                                                 // restart A at the top of this row-block
    mov ( rdx, rbx )                                                 // load b to rbx

    lea ( mem ( r10, r8, 2 ), rdi )                                  // rdi = 5 * rs_a ( rdi is reused by the store path, so recompute each iteration )

    // zmm8-zmm28 accumulate a 6x3 tile.
    INIT_ACCUM_3COL

    mov ( var ( k_iter64 ), rsi )                                    // number of 64-float k blocks
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

    // Load one 16-float vector from each of the 3 columns of B and update the 6x3 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                    // number of remaining 32-float k blocks
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

    // Load one 16-float vector from each of the 3 columns of B and update the 6x3 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // The previous k-loop decomposition used iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( ( rbx, r9, 2 ), zmm18 )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

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
    // Masked ZMM tail for 1-15 remaining k values.
    vmovups (         mem ( rax ), ZMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), ZMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), ZMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, r10, 1 ), ZMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ), ZMM ( 4 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, rdi, 1 ), ZMM ( 5 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 17 MASK_KZ ( 1 ) ) )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 18 MASK_KZ ( 1 ) ) )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // For tails of <= 8 elements, use masked YMM operations for the
    // remainder instead of looping element-by-element and issuing
    // per-element FMAs, which is wasteful for this kernel.
    // Perform a masked FMA operation on the remaining elements.
    vmovups (         mem ( rax ), YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), YMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, r10, 1 ), YMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ), YMM ( 4 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, rdi, 1 ), YMM ( 5 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 17 MASK_KZ ( 1 ) ) )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 18 MASK_KZ ( 1 ) ) )
    VFMA6 ( 18, 14, 15, 16, 26, 27, 28 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm30 )                                         // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 6x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 6x3 tile.
    // ZMM_REDUCE_3 folds 3 such accumulators across k and packs the 3 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..2.
    // C_STOR_MASKED then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                   // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ),  *]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                   // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ),  *]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                   // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), *]

    ALPHA_SCALE ( 29, 4, 5, 6 )                                      // scale the first 3 rows by alpha 

    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )
    C_STOR_MASKED ( rdi, 30, 4, 5, 6 )                               // update the first 3 rows of the 6x3 tile

    ZMM_REDUCE_3 ( 20, 23, 26, 17 )                                  // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), *]
    ZMM_REDUCE_3 ( 21, 24, 27, 18 )                                  // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), *]
    ZMM_REDUCE_3 ( 22, 25, 28, 19 )                                  // xmm19 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), *]

    ALPHA_SCALE ( 29, 17, 18, 19 )                                   // scale the next 3 rows by alpha 

    C_STOR_MASKED_CONT ( rdi,  30, 17, 18, 19 )                      // update the next 3 rows 

    jmp ( .SDONE )

    // Reduce the 6x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                   // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ),  *]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                   // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ),  *]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                   // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), *]

    ALPHA_SCALE ( 29, 4, 5, 6 )                                      // scale the first 3 rows by alpha 

    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )
    C_STOR_BZ_MASKED ( rdi, 4, 5, 6 )                                // store the first 3 rows without reading C

    ZMM_REDUCE_3 ( 20, 23, 26, 17 )                                  // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), *]
    ZMM_REDUCE_3 ( 21, 24, 27, 18 )                                  // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), *]
    ZMM_REDUCE_3 ( 22, 25, 28, 19 )                                  // xmm19 = [sum ( zmm22 ), sum ( zmm25 ), sum ( zmm28 ), *]

    ALPHA_SCALE ( 29, 17, 18, 19 )                                   // scale the next 3 rows by alpha 

    C_STOR_BZ_MASKED_CONT ( rdi, 17, 18, 19 )                        // store the next 3 rows 

    label ( .SDONE )

    mov ( var ( rs_c ), rdi )                                        // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )                                    // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r12, rdi, 2 ), r12 )
    lea ( mem ( r12, rdi, 4 ), r12 )                                 // c_ii = r12 += 6*rs_c

    lea ( mem ( r14, r8,  2 ), r14 )
    lea ( mem ( r14, r8,  4 ), r14 )                                 // a_ii = r14 += 6*rs_a

    dec ( r11 )
    jne ( .SLOOP3X4I )                                               // iterate again if ii != 0.

    end_asm (
    :                                                                // output operands ( none )
    :                                                                // input operands
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
      [m_iter]   "m" ( m_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                                // register clobber list
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
      "memory", "k1", "k2"
    )

    consider_edge_cases:

    // Control reaches here in two cases:
    // 1. asm finished the full 6x3 tile work and only bottom-edge rows remain.
    // 2. asm was skipped because m0 < 6.
    // m_left dispatch: handle the bottom edge rows that remain after the asm
    // kernel covers all full 6-row blocks.
    // Computes the m_left part of C.
    //   [ - ][ - ][ - ]
    //   [ - ][ - ][ - ]
    //   [ - ][ - ][ - ]
    //   [ - ][ - ][ - ]
    //   [ - ][ - ][ - ]
    //   [ - ][ - ][ - ]
    //   [  m_left area ]
    if ( m_left )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        // Start at row 6 * m_iter, i.e. the first row not covered by asm.
        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen5_asm_5x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen5_asm_4x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_3x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_2x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen5_asm_1x3
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen5_asm_6x2m
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
    // This kernel handles 6 rows and 2 columns of C.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 6 rows at a time; 1-5 rows fall through to m_left dispatch.
    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // The asm microkernel only handles full 6x2 tiles.
    // If m0 < 6, skip straight to the C dispatch below.
    if ( m_iter == 0 ) goto consider_edge_cases;

    // Main 6x2 microkernel over the full 6x2 rectangle:
    // Handles all rows in multiples of 6.
    // -------------------------------------------------------------------------
    // Computes the asm part of C.
    //   [ asm ][ asm ]
    //   [ asm ][ asm ]
    //   [ asm ][ asm ]
    //   [ asm ][ asm ]
    //   [ asm ][ asm ]
    //   [ asm ][ asm ]
    //   [   m_left   ]
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                                         // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                      // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                         // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                      // cs_b *= sizeof ( dt ) => cs_b *= 4

    mov ( var ( iter_1_mask ), esi )                                 // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), r14 )                                        // load base address of a
    mov ( var ( bbuf ), rdx )                                        // load base address of b
    mov ( var ( cbuf ), r12 )                                        // load base address of c

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                       // load address of alpha
    vbroadcastss ( ( rsi ), xmm14 )                                  // xmm14 <- alpha 
    mov ( var ( beta ), rsi )                                        // load address of beta
    vbroadcastss ( ( rsi ), xmm15 )                                  // xmm15 <- beta 

    lea ( mem ( r8, r8, 2 ), r10 )                                   // r10 = 3 * rs_a 
    mov ( var ( m_iter ), r11 )                                      // ii = m_iter;
    label ( .SLOOP3X4I )                                             // Outer loop over 6-row blocks of C/A

    mov ( r12, rcx )                                                 // rcx = base of the current 6x2 output tile in C

    // Prefetch the rows of C that will be written to in this loop
    mov ( var ( rs_c ), rsi )                                        // rsi = rs_c
    lea ( mem ( , rsi, 4 ), rsi )                                    // rsi = rs_c * sizeof ( float )
    prefetchw0 ( mem ( rcx ) )                                       // C row 0
    prefetchw0 ( mem ( rcx, rsi, 1 ) )                               // C row 1
    prefetchw0 ( mem ( rcx, rsi, 2 ) )                               // C row 2
    prefetchw0 ( mem ( rcx, rsi, 4 ) )                               // C row 4
    lea ( mem ( rcx, rsi, 2 ), rax )                                 // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 3 = rcx + 3 * rs_c
    lea ( mem ( rcx, rsi, 4 ), rax )                                 // rax = rcx + 4 * rs_c
    prefetchw0 ( mem ( rax, rsi, 1 ) )                               // C row 5 = rcx + 5 * rs_c

    mov ( r14, rax )                                                 // restart A at the top of this row-block
    mov ( rdx, rbx )                                                 // load b to rbx

    lea ( mem ( r10, r8, 2 ), rdi )                                  // rdi = 5 * rs_a ( rdi is reused by the store path, so recompute each iteration )

    // zmm8-zmm25 accumulate a 6x2 tile.
    INIT_ACCUM_2COL

    mov ( var ( k_iter64 ), rsi )                                    // number of 64-float k blocks
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

    // Load one 16-float vector from each of the 2 columns of B and update the 6x2 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                    // number of remaining 32-float k blocks
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

    // Load one 16-float vector from each of the 2 columns of B and update the 6x2 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

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

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // The previous k-loop decomposition used iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    vmovups ( ( rax, rdi, 1 ), zmm5 )
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups ( ( rbx, r9, 1 ), zmm17 )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

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

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 17 MASK_KZ ( 1 ) ) )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp ( .POST_ACCUM ) 

    label ( .K_FLOATS_LEFT_LE_8 )
    // For tails of <= 8 elements, use masked YMM operations for the
    // remainder instead of looping element-by-element and issuing
    // per-element FMAs, which is wasteful for this kernel.
    // Perform a masked FMA operation on the remaining elements.
    vmovups (         mem ( rax ), YMM ( 0 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 1 ), YMM ( 1 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 2 ), YMM ( 2 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, r10, 1 ), YMM ( 3 MASK_KZ ( 1 ) ) )
    vmovups (  mem ( rax, r8, 4 ), YMM ( 4 MASK_KZ ( 1 ) ) )
    vmovups ( mem ( rax, rdi, 1 ), YMM ( 5 MASK_KZ ( 1 ) ) )

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA6 ( 6,  8,  9, 10, 20, 21, 22 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 17 MASK_KZ ( 1 ) ) )
    VFMA6 ( 17, 11, 12, 13, 23, 24, 25 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm15 )                                         // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 6x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 6x2 tile.
    // ZMM_REDUCE_2 folds 2 such accumulators across k and packs the 2 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..1.
    // C_STOR_2_FLOATS then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_2 (  8, 11, 4 )                                       // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ),  *, *]
    ZMM_REDUCE_2 (  9, 12, 5 )                                       // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ),  *, *]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                       // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), *, *]

    ALPHA_SCALE ( 14, 4, 5, 6 )                                      // scale the first 3 rows by alpha 

    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )
    C_STOR_2_FLOATS ( rdi, 15, 4, 5, 6 )                             // update the first 3 rows of the 6x2 tile

    ZMM_REDUCE_2 ( 20, 23, 17 )                                      // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), *, *]
    ZMM_REDUCE_2 ( 21, 24, 18 )                                      // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), *, *]
    ZMM_REDUCE_2 ( 22, 25, 19 )                                      // xmm19 = [sum ( zmm22 ), sum ( zmm25 ), *, *]

    ALPHA_SCALE ( 14, 17, 18, 19 )                                   // scale the next 3 rows by alpha 

    C_STOR_2_FLOATS_CONT ( rdi, 15, 17, 18, 19 )                     // update the next 3 rows 

    jmp ( .SDONE )

    // Reduce the 6x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                       // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ),  *, *]
    ZMM_REDUCE_2 (  9, 12, 5 )                                       // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ),  *, *]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                       // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), *, *]

    ALPHA_SCALE ( 14, 4, 5, 6 )                                      // scale the first 3 rows by alpha 

    mov ( var ( rs_c ), rdi )                                        // rdi = rs_c  
    lea ( mem ( , rdi, 4 ), rdi )                                    // rdi = rs_c * sizeof ( float )
    C_STOR_BZ_2_FLOATS ( rdi, 4, 5, 6 )                              // store the first 3 rows without reading C

    ZMM_REDUCE_2 ( 20, 23, 17 )                                      // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), *, *]
    ZMM_REDUCE_2 ( 21, 24, 18 )                                      // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), *, *]
    ZMM_REDUCE_2 ( 22, 25, 19 )                                      // xmm19 = [sum ( zmm22 ), sum ( zmm25 ), *, *]

    ALPHA_SCALE ( 14, 17, 18, 19 )                                   // scale the next 3 rows by alpha 

    C_STOR_BZ_2_FLOATS_CONT ( rdi, 17, 18, 19 )                      // store the next 3 rows 

    label ( .SDONE )

    mov ( var ( rs_c ), rdi )                                        // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )                                    // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r12, rdi, 2 ), r12 )
    lea ( mem ( r12, rdi, 4 ), r12 )                                 // c_ii = r12 += 6*rs_c

    lea ( mem ( r14, r8,  2 ), r14 )
    lea ( mem ( r14, r8,  4 ), r14 )                                 // a_ii = r14 += 6*rs_a

    dec ( r11 )
    jne ( .SLOOP3X4I )                                               // iterate again if ii != 0.

    end_asm (
    :                                                                // output operands ( none )
    :                                                                // input operands
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
      [m_iter]   "m" ( m_iter ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                                // register clobber list
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

    consider_edge_cases:

    // Control reaches here in two cases:
    // 1. asm finished the full 6x2 tile work and only bottom-edge rows remain.
    // 2. asm was skipped because m0 < 6.
    // m_left dispatch: handle the bottom edge rows that remain after the asm
    // kernel covers all full 6-row blocks.
    // Computes the m_left part of C.
    //   [ - ][ - ]
    //   [ - ][ - ]
    //   [ - ][ - ]
    //   [ - ][ - ]
    //   [ - ][ - ]
    //   [ - ][ - ]
    //   [ m_left ]
    if ( m_left )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        // Start at row 6 * m_iter, i.e. the first row not covered by asm.
        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 5 == m_left )
        {
            dim_t mr_cur = 5;
            bli_sgemmsup_rd_zen5_asm_5x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 4 == m_left )
        {
            const dim_t mr_cur = 4;

            bli_sgemmsup_rd_zen5_asm_4x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 3 == m_left )
        {
            const dim_t mr_cur = 3;

            bli_sgemmsup_rd_zen5_asm_3x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen5_asm_2x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
        else if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen5_asm_1x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}
