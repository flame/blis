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

void bli_sgemmsup_rd_zen5_asm_5x64
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
    // This file's x64 kernels only handle columns of C in groups of 4.
    // Any remaining 1-3 columns of C are handled elsewhere.
    // This kernel handles 5 rows of C over those 4-column groups.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns of C at a time; any remaining 1-3 columns of C
    // are intentionally not handled here.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;
    uint64_t n_main_loop = n0 - n_left;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // This kernel only computes columns of C in the range [0, 4 * n_iter ).
    // If n0 < 4, this kernel does no work and those columns of C are handled elsewhere.
    if ( n_iter == 0 ) return;

    // Main 5x4 microkernel over the full left rectangle handled by this kernel:
    // Handles 5 rows and columns in multiples of 4.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes
    lea ( mem ( r8, r8, 2 ), r10 )                                 // r10 = 3 * rs_a

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm28 )                                // xmm28 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( n_main_loop ), r15 )                               // r15 = n_main_loop
    sub ( imm ( 4 ), r15 )                                         // jj = n_main_loop - 4
    mov ( var ( abuf ), r14 )                                      // load base address of a 
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( var ( bbuf ), rdx )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    lea ( mem ( r12, r15, 4 ), r12 )                               // step to column jj within C

    lea ( mem (   , r15, 1 ), rsi )                                // rsi = r15 = 4*jj;
    imul ( r9, rsi )                                               // rsi *= cs_b;
    lea ( mem ( rdx, rsi, 1 ), rdx )                               // step to column jj within B

    mov ( r12, rcx )                                               // rcx = base of the current 5x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    prefetchw0 ( mem ( rcx, r11, 4 ) )                             // C row 4
    lea ( mem ( rcx, r11, 2 ), rax )                               // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 3
    mov ( r14, rax )                                               // restart A at the top of this 4-col block
    mov ( rdx, rbx )                                               // restart B at the top of this 4-col block

    // zmm8-zmm30 accumulate a 5x4 tile.
    INIT_ACCUM_5x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from each A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from each of the 5 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 5x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

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
    VFMA5 ( 6, 8, 9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

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
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

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
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 5 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 5x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    // Load one 16-float vector from each of the 5 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 5 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 5x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

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

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

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

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA5 ( 25, 17, 18, 19, 29, 30 )

    label ( .POST_ACCUM )

    // alpha is preloaded into xmm28 above the J-loop
    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 5x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // Each zmm accumulator holds 16 partial sums for one C ( i,j ) in the
    // current 5x4 tile.
    // ZMM_REDUCE_4 folds 4 such accumulators across k and packs the 4 final
    // column results for one row into one xmm register.
    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    // C_STOR then does beta * C + accum and writes those row vectors to C.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 28, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the first 3 rows of the 5x4 tile

    // xmm22/xmm25 <- rows 3..4, cols 0..3.
    ZMM_REDUCE_4 ( 20, 23, 26, 29, 22 )                            // xmm22 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 25 )                            // xmm25 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]

    ALPHA_SCALE2 ( 28, 22, 25 )                                    // scale the next 2 rows by alpha 

    C_STOR2_CONT ( r11, 31, 22, 25 )                               // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 5x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 28, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the first 3 rows without reading C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 22 )                            // xmm22 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]
    ZMM_REDUCE_4 ( 21, 24, 27, 30, 25 )                            // xmm25 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 ), sum ( zmm30 )]

    ALPHA_SCALE2 ( 28, 22, 25 )                                    // scale the next 2 rows by alpha 

    C_STOR_BZ2_CONT ( r11, 22, 25 )                                // store the next 2 rows 

    label ( .SDONE )

    sub ( imm ( 4 ), r15 )
    test ( r15, r15 )
    jns ( .SLOOP3X4J )                                             // iterate while jj >= 0

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
      [n_main_loop]   "m" ( n_main_loop ),
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
      "ymm20", "ymm21", "ymm23", "ymm24", "ymm26", "ymm27",
      "ymm29", "ymm30",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_4x64
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
    // This file's x64 kernels only handle columns of C in groups of 4.
    // Any remaining 1-3 columns of C are handled elsewhere.
    // This kernel handles 4 rows of C over those 4-column groups.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns of C at a time; any remaining 1-3 columns of C
    // are intentionally not handled here.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;
    uint64_t n_main_loop = n0 - n_left;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // This kernel only computes columns of C in the range [0, 4 * n_iter ).
    // If n0 < 4, this kernel does no work and those columns of C are handled elsewhere.
    if ( n_iter == 0 ) return;

    // Main 4x4 microkernel over the full rectangle handled by this kernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes
    lea ( mem ( r8, r8, 2 ), r10 )                                 // r10 = 3 * rs_a

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( n_main_loop ), r15 )                               // r15 = n_main_loop
    sub ( imm ( 4 ), r15 )                                         // jj = n_main_loop - 4
    mov ( var ( abuf ), r14 )                                      // load base address of a 
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( var ( bbuf ), rdx )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    lea ( mem ( r12, r15, 4 ), r12 )                               // step to column jj within C

    lea ( mem ( , r15, 1 ), rsi )                                  // rsi = r15 = 4*jj;
    imul ( r9, rsi )                                               // rsi *= cs_b;
    lea ( mem ( rdx, rsi, 1 ), rdx )                               // step to column jj within B

    mov ( r12, rcx )                                               // rcx = base of the current 4x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    lea ( mem ( rcx, r11, 2 ), rax )                               // rax = rcx + 2 * rs_c
    prefetchw0 ( mem ( rax, r11, 1 ) )                             // C row 3
    mov ( r14, rax )                                               // restart A at the top of this 4-col block
    mov ( rdx, rbx )                                               // restart B at the top of this 4-col block

    // zmm8-zmm29 accumulate a 4x4 tile.
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
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm6 )
    VFMA4 ( 6, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm7 )
    VFMA4 ( 7, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from each of the 4 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 4x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    // Load one 16-float vector from each of the 4 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from each of the 4 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 4x4 accumulators.
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA4 ( 25, 17, 18, 19, 29 )

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

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA4 ( 25, 17, 18, 19, 29 )

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

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA4 ( 22, 14, 15, 16, 26 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA4 ( 25, 17, 18, 19, 29 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 4x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the first 3 rows of the 4x4 tile

    // xmm21 <- row 3, cols 0..3.
    ZMM_REDUCE_4 ( 20, 23, 26, 29, 21 )                            // xmm21 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]

    ALPHA_SCALE1 ( 30, 21 )                                        // scale the next row by alpha 

    C_STOR1_CONT ( r11, 31, 21 )                                   // update the next row 

    jmp ( .SDONE )

    // Reduce the 4x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the first 3 rows without reading C

    ZMM_REDUCE_4 ( 20, 23, 26, 29, 21 )                            // xmm21 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 ), sum ( zmm29 )]

    ALPHA_SCALE1 ( 30, 21 )                                        // scale the next row by alpha 

    C_STOR_BZ1_CONT ( r11, 21 )                                    // store the next row 

    label ( .SDONE )

    sub ( imm ( 4 ), r15 )
    test ( r15, r15 )
    jns ( .SLOOP3X4J )                                             // iterate while jj >= 0

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
      [n_main_loop]   "m" ( n_main_loop ),
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
      "ymm20", "ymm23", "ymm26", "ymm29",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_3x64
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
    // This file's x64 kernels only handle columns of C in groups of 4.
    // Any remaining 1-3 columns of C are handled elsewhere.
    // This kernel handles 3 rows of C over those 4-column groups.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns of C at a time; any remaining 1-3 columns of C
    // are intentionally not handled here.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;
    uint64_t n_main_loop = n0 - n_left;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // This kernel only computes columns of C in the range [0, 4 * n_iter ).
    // If n0 < 4, this kernel does no work and those columns of C are handled elsewhere.
    if ( n_iter == 0 ) return;

    // Main 3x4 microkernel over the full rectangle handled by this kernel.
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

    mov ( var ( n_main_loop ), r15 )                               // r15 = n_main_loop
    sub ( imm ( 4 ), r15 )                                         // jj = n_main_loop - 4
    mov ( var ( abuf ), r14 )                                      // load base address of a 
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( var ( bbuf ), rdx )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    lea ( mem ( r12, r15, 4 ), r12 )                               // step to column jj within C

    lea ( mem (  , r15, 1 ), rsi )                                 // rsi = r15 = 4*jj;
    imul ( r9, rsi )                                               // rsi *= cs_b;
    lea ( mem ( rdx, rsi, 1 ), rdx )                               // step to column jj within B

    mov ( r12, rcx )                                               // rcx = base of the current 3x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    mov ( r14, rax )                                               // restart A at the top of this 4-col block
    mov ( rdx, rbx )                                               // restart B at the top of this 4-col block

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    // Load one 16-float vector from each of the 3 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA3 ( 25, 17, 18, 19 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA3 ( 25, 17, 18, 19 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA3 ( 22, 14, 15, 16 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA3 ( 25, 17, 18, 19 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // xmm4/xmm5/xmm6 <- rows 0..2, cols 0..3.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR ( r11, 31, 4, 5, 6 )                                    // update the 3 rows of the 3x4 tile

    jmp ( .SDONE )

    // Reduce the 3x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]
    ZMM_REDUCE_4 ( 10, 13, 16, 19, 6 )                             // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 ), sum ( zmm19 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_BZ ( r11, 4, 5, 6 )                                     // store the 3 rows without reading C

    label ( .SDONE )

    sub ( imm ( 4 ), r15 )
    test ( r15, r15 )
    jns ( .SLOOP3X4J )                                             // iterate while jj >= 0

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
      [n_main_loop]       "m" ( n_main_loop ),
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

void bli_sgemmsup_rd_zen5_asm_2x64
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
    // This file's x64 kernels only handle columns of C in groups of 4.
    // Any remaining 1-3 columns of C are handled elsewhere.
    // This kernel handles 2 rows of C over those 4-column groups.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns of C at a time; any remaining 1-3 columns of C
    // are intentionally not handled here.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;
    uint64_t n_main_loop = n0 - n_left;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // This kernel only computes columns of C in the range [0, 4 * n_iter ).
    // If n0 < 4, this kernel does no work and those columns of C are handled elsewhere.
    if ( n_iter == 0 ) return;

    // Main 2x4 microkernel over the full rectangle handled by this kernel.
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

    mov ( var ( n_main_loop ), r15 )                               // r15 = n_main_loop
    sub ( imm ( 4 ), r15 )                                         // jj = n_main_loop - 4
    mov ( var ( abuf ), r14 )                                      // load base address of a 
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( var ( bbuf ), rdx )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    lea ( mem ( r12, r15, 4 ), r12 )                               // step to column jj within C

    lea ( mem (  , r15, 1 ), rsi )                                 // rsi = r15 = 4*jj;
    imul ( r9, rsi )                                               // rsi *= cs_b;
    lea ( mem ( rdx, rsi, 1 ), rdx )                               // step to column jj within B

    mov ( r12, rcx )                                               // rcx = base of the current 2x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    mov ( r14, rax )                                               // restart A at the top of this 4-col block
    mov ( rdx, rbx )                                               // restart B at the top of this 4-col block

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    // Load one 16-float vector from each of the 2 rows of A.
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    vmovups ( ( rbx, r13, 1 ), zmm25 )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA2 ( 22, 14, 15 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA2 ( 25, 17, 18 )

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

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA2 ( 22, 14, 15 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA2 ( 25, 17, 18 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 2x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // xmm4/xmm5 <- rows 0..1, cols 0..3.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha
    
    C_STOR2 ( r11, 31, 4, 5 )                                      // update the 2 rows of the 2x4 tile

    jmp ( .SDONE )

    // Reduce the 2x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]
    ZMM_REDUCE_4 (  9, 12, 15, 18, 5 )                             // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 ), sum ( zmm18 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_BZ2 ( r11, 4, 5 )                                       // store the 2 rows without reading C

    label ( .SDONE )

    sub ( imm ( 4 ), r15 )
    test ( r15, r15 )
    jns ( .SLOOP3X4J )                                             // iterate while jj >= 0

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
      [n_main_loop]       "m" ( n_main_loop ),
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

void bli_sgemmsup_rd_zen5_asm_1x64
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
    // This file's x64 kernels only handle columns of C in groups of 4.
    // Any remaining 1-3 columns of C are handled elsewhere.
    // This kernel handles 1 row of C over those 4-column groups.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    // Main loop handles 4 columns of C at a time; any remaining 1-3 columns of C
    // are intentionally not handled here.
    uint64_t n_iter = n0 / 4;
    uint64_t n_left = n0 % 4;
    uint64_t n_main_loop = n0 - n_left;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // This kernel only computes columns of C in the range [0, 4 * n_iter ).
    // If n0 < 4, this kernel does no work and those columns of C are handled elsewhere.
    if ( n_iter == 0 ) return;

    // Main 1x4 microkernel over the full rectangle handled by this kernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4
    lea ( mem ( r9, r9, 2 ), r13 )                                 // r13 = 3 * cs_b in bytes
    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float )

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )

    mov ( var ( n_main_loop ), r15 )                               // r15 = n_main_loop
    sub ( imm ( 4 ), r15 )                                         // jj = n_main_loop - 4
    mov ( var ( abuf ), r14 )                                      // load base address of a 
    label ( .SLOOP3X4J )                                           // Inner loop over 4-column output tiles

    mov ( var ( bbuf ), rdx )                                      // load base address of b
    mov ( var ( cbuf ), r12 )                                      // load base address of c

    lea ( mem ( r12, r15, 4 ), r12 )                               // step to column jj within C

    lea ( mem (  , r15, 1 ), rsi )                                 // rsi = r15 = 4*jj;
    imul ( r9, rsi )                                               // rsi *= cs_b;
    lea ( mem ( rdx, rsi, 1 ), rdx )                               // step to column jj within B

    mov ( r12, rcx )                                               // rcx = base of the current 1x4 output tile in C
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    mov ( r14, rax )                                               // restart A at the top of this 4-col block
    mov ( rdx, rbx )                                               // restart B at the top of this 4-col block

    // zmm8/zmm11/zmm14/zmm17 accumulate a 1x4 tile.
    INIT_ACCUM_1x4

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // Each unrolled iteration consumes 16 floats from the A row and 16 floats from each B column.
    // Four such iterations make one 64-float k block.
    // ITER 0
    // Load one 16-float vector from the row of A.
    vmovups (         ( rax ), zmm0 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 1x4 accumulators.
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm0, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm0, zmm17 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm13 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vfmadd231ps ( ( rbx ), zmm13, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm13, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm13, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm13, zmm17 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm15 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vfmadd231ps ( ( rbx ), zmm15, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm15, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm15, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm15, zmm17 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm18 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vfmadd231ps ( ( rbx ), zmm18, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm18, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm18, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm18, zmm17 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // Two 16-float iterations cover the 32-float remainder block.
    // ITER 0
    // Load one 16-float vector from the row of A.
    vmovups (         ( rax ), zmm0 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 1x4 accumulators.
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm0, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm0, zmm17 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm13 )
    add ( imm ( 16*4 ), rax )

    // Load the next 16-float B vectors and continue accumulating.
    vfmadd231ps ( ( rbx ), zmm13, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm13, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm13, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm13, zmm17 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    // Load one 16-float vector from the row of A.
    vmovups (         ( rax ), zmm15 )
    add ( imm ( 16*4 ), rax )

    // Load one 16-float vector from each of the 4 columns of B and update the 1x4 accumulators.
    vfmadd231ps ( ( rbx ), zmm15, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm15, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm15, zmm14 )
    vfmadd231ps ( ( rbx, r13, 1 ), zmm15, zmm17 )

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

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6, 8 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA1 ( 22, 14 )

    vmovups ( mem ( rbx, r13, 1 ), ZMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA1 ( 25, 17 )

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

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6, 8 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA1 ( 22, 14 )

    vmovups ( mem ( rbx, r13, 1 ), YMM ( 25 MASK_KZ ( 1 ) ) )
    VFMA1 ( 25, 17 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 1x4 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    // xmm4 <- row 0, cols 0..3.
    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ), sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]

    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR1 ( r11, 31, 4 )                                         // update the row of the 1x4 tile

    jmp ( .SDONE )

    // Reduce the 1x4 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_4 (  8, 11, 14, 17, 4 )                             // xmm4 = [sum ( zmm8 ), sum ( zmm11 ), sum ( zmm14 ), sum ( zmm17 )]

    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR_BZ1 ( r11, 4 )                                          // store the row without reading C

    label ( .SDONE )

    sub ( imm ( 4 ), r15 )
    test ( r15, r15 )
    jns ( .SLOOP3X4J )                                             // iterate while jj >= 0

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
      [n_main_loop]       "m" ( n_main_loop ),
      [abuf]     "m" ( abuf ),
      [bbuf]     "m" ( bbuf ),
      [cbuf]     "m" ( cbuf )
    :                                                              // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10", "ymm11", "ymm13",
      "ymm14", "ymm17",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rd_zen5_asm_5x3
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
    // This kernel handles 5 rows and 3 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 5x3 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )
    kmovw ( esi, K ( 2 ) )                                         // k2 = mask for the 3 active C columns

    mov ( var ( abuf ), rax )                                      // load base address of a
    mov ( var ( bbuf ), rbx )                                      // load base address of b
    mov ( var ( cbuf ), rcx )                                      // load base address of c

    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    prefetchw0 ( mem ( rcx, r11, 4 ) )                             // C row 4
    lea ( mem ( rcx, r11, 2 ), rsi )                               // r11 = rcx + 2 * rs_c
    prefetchw0 ( mem ( rsi, r11, 1 ) )                             // C row 3

    INIT_ACCUM_5x3

    mov ( var ( k_iter64 ), rsi )                                  // number of 64-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // number of remaining 32-float k blocks
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

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
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

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

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

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

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA5 ( 22, 14, 15, 16, 26, 27 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 5x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                 // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_MASKED ( r11, 31, 4, 5, 6 )                             // update the first 3 rows of the 5x3 tile

    ZMM_REDUCE_3 ( 20, 23, 26, 17 )                                // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 )]
    ZMM_REDUCE_3 ( 21, 24, 27, 18 )                                // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 )]

    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_MASKED2_CONT ( r11, 31, 17, 18 )                        // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 5x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                 // xmm6  = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_BZ_MASKED ( r11, 4, 5, 6 )                              // store the first 3 rows without reading C

    ZMM_REDUCE_3 ( 20, 23, 26, 17 )                                // xmm17 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 )]
    ZMM_REDUCE_3 ( 21, 24, 27, 18 )                                // xmm18 = [sum ( zmm21 ), sum ( zmm24 ), sum ( zmm27 )]

    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_BZ_MASKED2_CONT ( r11, 17, 18 )                         // store the next 2 rows 

    label ( .SDONE )

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
      "memory", "k1", "k2"
    )
}

void bli_sgemmsup_rd_zen5_asm_4x3
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
    // This kernel handles 4 rows and 3 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 4x3 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )
    kmovw ( esi, K ( 2 ) )                                         // k2 = mask for the 3 active C columns

    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    lea ( mem ( rcx, r11, 2 ), rsi )                               // r11 = rcx + 2 * rs_c
    prefetchw0 ( mem ( rsi, r11, 1 ) )                             // C row 3

    INIT_ACCUM_4x3

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

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
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA4 ( 22, 14, 15, 16, 26 )

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

    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA4 ( 22, 14, 15, 16, 26 )

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

    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA4 ( 22, 14, 15, 16, 26 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 4x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the first 2 rows by alpha 

    C_STOR_MASKED2 ( r11, 31, 4, 5 )                               // update the first 2 rows of the 4x3 tile
    
    ZMM_REDUCE_3 ( 10, 13, 16, 17 )                                // xmm17 = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]
    ZMM_REDUCE_3 ( 20, 23, 26, 18 )                                // xmm18 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 )]

    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_MASKED2_CONT ( r11, 31, 17, 18 )                        // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 4x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4  = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5  = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the first 2 rows by alpha 

    C_STOR_BZ_MASKED2 ( r11, 4, 5 )                                // store the first 2 rows without reading C

    ZMM_REDUCE_3 ( 10, 13, 16, 17 )                                // xmm17 = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]
    ZMM_REDUCE_3 ( 20, 23, 26, 18 )                                // xmm18 = [sum ( zmm20 ), sum ( zmm23 ), sum ( zmm26 )]

    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_BZ_MASKED2_CONT ( r11, 17, 18 )                         // store the next 2 rows 

    label ( .SDONE )

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
      "memory", "k1", "k2"
    )
}

void bli_sgemmsup_rd_zen5_asm_3x3
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
    // This kernel handles 3 rows and 3 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 3x3 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )
    kmovw ( esi, K ( 2 ) )                                         // k2 = mask for the 3 active C columns

    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2

    INIT_ACCUM_3x3

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )
  
    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

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
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA3 ( 22, 14, 15, 16 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA3 ( 22, 14, 15, 16 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA3 ( 22, 14, 15, 16 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                 // xmm6 = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_MASKED ( r11, 31, 4, 5, 6 )                             // update the 3 rows of the 3x3 tile

    jmp ( .SDONE )

    // Reduce the 3x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]
    ZMM_REDUCE_3 ( 10, 13, 16, 6 )                                 // xmm6 = [sum ( zmm10 ), sum ( zmm13 ), sum ( zmm16 )]

    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_BZ_MASKED ( r11, 4, 5, 6 )                              // store the 3 rows without reading C

    label ( .SDONE )

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
      "memory", "k1", "k2"
    )
}

void bli_sgemmsup_rd_zen5_asm_2x3
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
    // This kernel handles 2 rows and 3 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 2x3 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )                                         // k2 = keep the first 3 lanes for 3 columns of C
    kmovw ( esi, K ( 2 ) )

    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float )

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1

    INIT_ACCUM_2x3

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )
  
    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    vmovups ( ( rbx, r9, 2 ), zmm22 )
    VFMA2 ( 22, 14, 15 )

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

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA2 ( 22, 14, 15 )

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

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA2 ( 22, 14, 15 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 2x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_MASKED2 ( r11, 31, 4, 5 )                               // update the 2x3 tile in C

    jmp ( .SDONE )

    // Reduce the 2x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ),  sum ( zmm11 ), sum ( zmm14 )]
    ZMM_REDUCE_3 (  9, 12, 15, 5 )                                 // xmm5 = [sum ( zmm9 ),  sum ( zmm12 ), sum ( zmm15 )]

    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_BZ_MASKED2 ( r11, 4, 5 )                                // store the 2x3 tile without reading C

    label ( .SDONE )

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
      "memory", "k1", "k2"
    )
}

void bli_sgemmsup_rd_zen5_asm_1x3
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
    // This kernel handles 1 row and 3 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 1x3 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    mov ( imm ( 7 ), esi )                                         // k2 = keep the first 3 lanes for 3 columns of C
    kmovw ( esi, K ( 2 ) )

    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 
    prefetchw0 ( mem ( rcx ) )                                     // C row 0 

    INIT_ACCUM_1x3

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm0, zmm14 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm13 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm13, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm13, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm13, zmm14 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm15 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm15, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm15, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm15, zmm14 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm16 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm16, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm16, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm16, zmm14 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm0, zmm14 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm13 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm13, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm13, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm13, zmm14 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )
    vfmadd231ps ( ( rbx, r9, 2 ), zmm0, zmm14 )

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
    
    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6,  8 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

    vmovups ( mem ( rbx, r9, 2 ),  ZMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA1 ( 22, 14 )

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
    
    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6,  8 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

    vmovups ( mem ( rbx, r9, 2 ),  YMM ( 22 MASK_KZ ( 1 ) ) )
    VFMA1 ( 22, 14 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 1x3 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ), sum ( zmm11 ), sum ( zmm14 )]

    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR_MASKED1 ( r11, 31, 4 )                                  // update the 1x3 tile in C

    jmp ( .SDONE )

    // Reduce the 1x3 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_3 (  8, 11, 14, 4 )                                 // xmm4 = [sum ( zmm8 ), sum ( zmm11 ), sum ( zmm14 )]

    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR_BZ_MASKED1 ( r11, 4 )                                   // store the 1x3 tile without reading C

    label ( .SDONE )

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
      "memory", "k1", "k2"
    )
}

void bli_sgemmsup_rd_zen5_asm_5x2
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
    // This kernel handles 5 rows and 2 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 5x2 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    prefetchw0 ( mem ( rcx, r11, 4 ) )                             // C row 4
    lea ( mem ( rcx, r11, 2 ), rsi )                               // r11 = rcx + 2 * rs_c
    prefetchw0 ( mem ( rsi, r11, 1 ) )                             // C row 3

    INIT_ACCUM_5x2

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    vmovups ( ( rax,  r8, 4 ), zmm4 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

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
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

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
    
    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

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
    
    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA5 ( 6,  8,  9, 10, 20, 21 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA5 ( 7, 11, 12, 13, 23, 24 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 5x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                     // xmm6 = [sum ( zmm10 ), sum ( zmm13 )]
    
    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_2_FLOATS ( r11, 31, 4, 5, 6 )                           // update the first 3 rows of the 5x2 tile

    ZMM_REDUCE_2 ( 20, 23, 17 )                                    // xmm17 = [sum ( zmm20 ), sum ( zmm23 )]
    ZMM_REDUCE_2 ( 21, 24, 18 )                                    // xmm18 = [sum ( zmm21 ), sum ( zmm24 )]
    
    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_2_FLOATS2_CONT ( r11, 31, 17, 18 )                      // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 5x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                     // xmm6 = [sum ( zmm10 ), sum ( zmm13 )]
    
    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the first 3 rows by alpha 

    C_STOR_BZ_2_FLOATS ( r11, 4, 5, 6 )                            // store the first 3 rows without reading C

    ZMM_REDUCE_2 ( 20, 23, 17 )                                    // xmm17 = [sum ( zmm20 ), sum ( zmm23 )]
    ZMM_REDUCE_2 ( 21, 24, 18 )                                    // xmm18 = [sum ( zmm21 ), sum ( zmm24 )]
    
    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_BZ_2_FLOATS2_CONT ( r11, 17, 18 )                       // store the next 2 rows 

    label ( .SDONE )

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

void bli_sgemmsup_rd_zen5_asm_4x2
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
    // This kernel handles 4 rows and 2 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 4x2 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    lea ( mem (  r8, r8, 2 ), r10 )                                // r10 = 3 * rs_a

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2
    lea ( mem ( rcx, r11, 2 ), rsi )                               // r11 = rcx + 2 * rs_c
    prefetchw0 ( mem ( rsi, r11, 1 ) )                             // C row 3

    INIT_ACCUM_4x2

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    vmovups ( ( rax, r10, 1 ), zmm3 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

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
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA4 ( 7, 11, 12, 13, 23 )

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
    
    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

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
    
    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA4 ( 6,  8,  9, 10, 20 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA4 ( 7, 11, 12, 13, 23 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 3x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4  = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5  = [sum ( zmm9 ),  sum ( zmm12 )]
    
    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the first 2 rows by alpha 

    C_STOR_2_FLOATS2 ( r11, 31, 4, 5 )                             // update the first 2 rows of the 4x2 tile

    ZMM_REDUCE_2 ( 10, 13, 17 )                                    // xmm17 = [sum ( zmm10 ), sum ( zmm13 )]
    ZMM_REDUCE_2 ( 20, 23, 18 )                                    // xmm18 = [sum ( zmm20 ), sum ( zmm23 )]
    
    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_2_FLOATS2_CONT ( r11, 31, 17, 18 )                      // update the next 2 rows 

    jmp ( .SDONE )

    // Reduce the 4x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4  = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5  = [sum ( zmm9 ),  sum ( zmm12 )]
    
    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the first 2 rows by alpha 

    C_STOR_BZ_2_FLOATS2 ( r11, 4, 5 )                              // store the first 2 rows without reading C

    ZMM_REDUCE_2 ( 10, 13, 17 )                                    // xmm17 = [sum ( zmm10 ), sum ( zmm13 )]
    ZMM_REDUCE_2 ( 20, 23, 18 )                                    // xmm18 = [sum ( zmm20 ), sum ( zmm23 )]
    
    ALPHA_SCALE2 ( 30, 17, 18 )                                    // scale the next 2 rows by alpha 

    C_STOR_BZ_2_FLOATS2_CONT ( r11, 17, 18 )                       // store the next 2 rows 

    label ( .SDONE )

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

void bli_sgemmsup_rd_zen5_asm_3x2
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
    // This kernel handles 3 rows and 2 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 3x2 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1
    prefetchw0 ( mem ( rcx, r11, 2 ) )                             // C row 2

    INIT_ACCUM_3x2

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    vmovups ( ( rax,  r8, 2 ), zmm2 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

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
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA3 ( 6,  8,  9, 10 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA3 ( 7, 11, 12, 13 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

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
    VFMA3 ( 6,  8,  9, 10 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA3 ( 7, 11, 12, 13 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 2x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                     // xmm6 = [sum ( zmm10 ), sum ( zmm13 )]
    
    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_2_FLOATS ( r11, 31, 4, 5, 6 )                           // update the 3x2 tile in C

    jmp ( .SDONE )

    // Reduce the 3x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    ZMM_REDUCE_2 ( 10, 13, 6 )                                     // xmm6 = [sum ( zmm10 ), sum ( zmm13 )]
    
    ALPHA_SCALE ( 30, 4, 5, 6 )                                    // scale the 3 rows by alpha

    C_STOR_BZ_2_FLOATS ( r11, 4, 5, 6 )                            // store the 3x2 tile without reading C

    label ( .SDONE )

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

void bli_sgemmsup_rd_zen5_asm_2x2
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
    // This kernel handles 2 rows and 2 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 2x2 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 
    prefetchw0 ( mem ( rcx, r11, 1 ) )                             // C row 1

    INIT_ACCUM_2x2

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    vmovups ( ( rax,  r8, 1 ), zmm1 )
    
    add ( imm ( 16*4 ), rax )

    vmovups (        ( rbx ), zmm6 )
    VFMA2 ( 6, 8, 9 )

    vmovups ( ( rbx, r9, 1 ), zmm7 )
    VFMA2 ( 7, 11, 12 )

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

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 2x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    
    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_2_FLOATS2 ( r11, 31, 4, 5 )                             // update the 2x2 tile in C

    jmp ( .SDONE )

    // Reduce the 2x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ),  sum ( zmm11 )]
    ZMM_REDUCE_2 (  9, 12, 5 )                                     // xmm5 = [sum ( zmm9 ),  sum ( zmm12 )]
    
    ALPHA_SCALE2 ( 30, 4, 5 )                                      // scale the 2 rows by alpha

    C_STOR_BZ_2_FLOATS2 ( r11, 4, 5 )                              // store the 2x2 tile without reading C

    label ( .SDONE )

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

void bli_sgemmsup_rd_zen5_asm_1x2
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
    // This kernel handles 1 row and 2 columns of C.
    // Decompose k into 64-float, 32-float, 16-float, and masked tail work.
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = ( 1 << k_left1 ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    // Main 1x2 microkernel.
    begin_asm()

    mov ( var ( rs_a ), r8 )                                       // load rs_a
    lea ( mem ( , r8, 4 ), r8 )                                    // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( cs_b ), r9 )                                       // load cs_b
    lea ( mem ( , r9, 4 ), r9 )                                    // cs_b *= sizeof ( dt ) => cs_b *= 4

    // preload alpha and beta into vector registers that are unused in this kernel
    mov ( var ( alpha ), rsi )                                     // load address of alpha
    vbroadcastss ( ( rsi ), xmm30 )                                // xmm30 <- alpha 
    mov ( var ( beta ), rsi )                                      // load address of beta
    vbroadcastss ( ( rsi ), xmm31 )                                // xmm31 <- beta 

    mov ( var ( iter_1_mask ), esi )                               // k1 = lane mask for the final k tail
    kmovw ( esi, K ( 1 ) )
    
    mov ( var ( abuf ), rax )                                      // load address of a
    mov ( var ( bbuf ), rbx )                                      // load address of b
    mov ( var ( cbuf ), rcx )                                      // load address of c

    mov ( var ( rs_c ), r11 )                                      // r11 = rs_c
    lea ( mem ( , r11, 4 ), r11 )                                  // r11 = rs_c * sizeof ( float ) 

    prefetchw0 ( mem ( rcx ) )                                     // C row 0 

    INIT_ACCUM_1x2

    mov ( var ( k_iter64 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_32 )

    label ( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm12 )
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm12, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm12, zmm11 )

    add ( imm ( 16*4 ), rbx )

    // ITER 2
    vmovups (         ( rax ), zmm13 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm13, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm13, zmm11 )

    add ( imm ( 16*4 ), rbx )

    // ITER 3
    vmovups (         ( rax ), zmm14 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm14, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm14, zmm11 )

    add ( imm ( 16*4 ), rbx )

    dec ( rsi )
    jne ( .K_LOOP_ITER64 )

    label ( .CONSIDER_K_ITER_32 )

    mov ( var ( k_iter32 ), rsi )                                  // load k_iter
    test ( rsi, rsi )
    je ( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )

    add ( imm ( 16*4 ), rbx )

    // ITER 1
    vmovups (         ( rax ), zmm12 )
    
    add ( imm ( 16*4 ), rax )

    // load column from B
    vfmadd231ps ( ( rbx ), zmm12, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm12, zmm11 )

    add ( imm ( 16*4 ), rbx )

    label ( .CONSIDER_K_ITER_16 )
    mov ( var ( k_iter16 ), rsi )
    test ( rsi, rsi )
    je ( .CONSIDER_K_LEFT_1 )

    // One full 16-float step remains before the masked k tail.
    // ITER 0
    vmovups (         ( rax ), zmm0 )
    
    add ( imm ( 16*4 ), rax )

    vfmadd231ps ( ( rbx ), zmm0, zmm8 )
    vfmadd231ps ( ( rbx, r9, 1 ), zmm0, zmm11 )

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
    
    vmovups (         mem ( rbx ), ZMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6,  8 )

    vmovups (  mem ( rbx, r9, 1 ), ZMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

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
    
    vmovups (         mem ( rbx ), YMM ( 6 MASK_KZ ( 1 ) ) )
    VFMA1 ( 6,  8 )

    vmovups (  mem ( rbx, r9, 1 ), YMM ( 7 MASK_KZ ( 1 ) ) )
    VFMA1 ( 7, 11 )

    label ( .POST_ACCUM )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm31 )                                       // branch on beta == 0 to skip reading C
    je ( .POST_ACCUM_STOR_BZ )

    // Reduce the 1x2 accumulators, scale by alpha, then update C with beta * C + accum.
    label ( .POST_ACCUM_STOR )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ), sum ( zmm11 )]
    
    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR_2_FLOATS1 ( r11, 31, 4 )                                // update the 1x2 tile in C

    jmp ( .SDONE )

    // Reduce the 1x2 accumulators, scale by alpha, then store directly when beta == 0.
    label ( .POST_ACCUM_STOR_BZ )

    ZMM_REDUCE_2 (  8, 11, 4 )                                     // xmm4 = [sum ( zmm8 ), sum ( zmm11 )]
    
    ALPHA_SCALE1 ( 30, 4 )                                         // scale the row by alpha

    C_STOR_BZ_2_FLOATS1 ( r11, 4 )                                 // store the 1x2 tile without reading C

    label ( .SDONE )

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
