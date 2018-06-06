/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#define SGEMM_OUTPUT_GS_BETA_NZ \
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t" \
	"vmovss            %%xmm0, (%%rcx        )   \n\t" \
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%rsi,1)   \n\t" \
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t" \
	"vmovss            %%xmm0, (%%rcx,%%rsi,2)   \n\t" \
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r13  )   \n\t" \
	"vmovss            %%xmm2, (%%rcx,%%rsi,4)   \n\t" \
	"vpermilps  $0x39, %%xmm2,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r15  )   \n\t" \
	"vpermilps  $0x39, %%xmm1,  %%xmm2           \n\t" \
	"vmovss            %%xmm2, (%%rcx,%%r13,2)   \n\t" \
	"vpermilps  $0x39, %%xmm2,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r10  )   \n\t"


void bli_sgemmtrsm_u_zen_asm_6x16
     (
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a10,
       float*     restrict a11,
       float*     restrict b01,
       float*     restrict b11,
       float*     restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	uint64_t   k_iter = k / 4;
	uint64_t   k_left = k % 4;

	float*     beta   = bli_sm1;

	__asm__ volatile
	(
	"                                            \n\t"
	"vzeroall                                    \n\t" // zero all xmm/ymm registers.
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"                                            \n\t"
	"addq           $32 * 4, %%rbx               \n\t"
	"                                            \n\t" // initialize loop by pre-loading
	"vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"movq                %7, %%rcx               \n\t" // load address of b11
	"movq               $16, %%rdi               \n\t" // set rs_b = PACKNR = 16
	"leaq        (,%%rdi,4), %%rdi               \n\t" // rs_b *= sizeof(float)
	"                                            \n\t"
	"                                            \n\t" // NOTE: c11, rs_c, and cs_c aren't
	"                                            \n\t" // needed for a while, but we load
	"                                            \n\t" // them now to avoid stalling later.
	"movq                %8, %%r8                \n\t" // load address of c11
	"movq                %9, %%r9                \n\t" // load rs_c
	"leaq        (,%%r9 ,4), %%r9                \n\t" // rs_c *= sizeof(float)
	"movq               %10, %%r10               \n\t" // load cs_c
	"leaq        (,%%r10,4), %%r10               \n\t" // cs_c *= sizeof(float)
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0   64 * 4(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss       0 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       1 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastss       2 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       3 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastss       4 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       5 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovaps           -2 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps           -1 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vbroadcastss       6 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       7 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastss       8 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       9 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastss      10 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      11 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovaps            0 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps            1 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0   76 * 4(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss      12 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      13 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastss      14 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      15 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastss      16 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      17 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovaps            2 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps            3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vbroadcastss      18 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      19 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastss      20 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      21 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastss      22 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss      23 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq          $4 *  6 * 4, %%rax            \n\t" // a += 4*6  (unroll x mr)
	"addq          $4 * 16 * 4, %%rbx            \n\t" // b += 4*16 (unroll x nr)
	"                                            \n\t"
	"vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0   64 * 4(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss       0 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       1 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastss       2 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       3 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastss       4 *  4(%%rax), %%ymm2    \n\t"
	"vbroadcastss       5 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq          $1 *  6 * 4, %%rax            \n\t" // a += 1*6  (unroll x mr)
	"addq          $1 * 16 * 4, %%rbx            \n\t" // b += 1*16 (unroll x nr)
	"                                            \n\t"
	"vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm4..ymm15 = -a10 * b01
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of alpha
	"vbroadcastss    (%%rbx), %%ymm3             \n\t" // load alpha and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                $1, %%rsi               \n\t" // load cs_b = 1
	"leaq        (,%%rsi,4), %%rsi               \n\t" // cs_b *= sizeof(float)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,8), %%rdx               \n\t" // load address of b11 + 8*cs_b
	"                                            \n\t"
	"movq             %%rcx, %%r11               \n\t" // save rcx = b11        for later
	"movq             %%rdx, %%r14               \n\t" // save rdx = b11+8*cs_b for later
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // b11 := alpha * b11 - a10 * b01
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm4    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm5    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm6    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm7    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm8    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm9    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm10   \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm11   \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm12   \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm13   \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231ps      (%%rcx), %%ymm3, %%ymm14   \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231ps      (%%rdx), %%ymm3, %%ymm15   \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // prefetch c11
	"                                            \n\t"
#if 0
	"movq             %%r8,  %%rcx               \n\t" // load address of c11 from r8
	"                                            \n\t" // Note: r9 = rs_c * sizeof(float)
	"                                            \n\t"
	"leaq   (%%r9 ,%%r9 ,2), %%r13               \n\t" // r13 = 3*rs_c;
	"leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c11 + 3*rs_c;
	"                                            \n\t"
	"prefetcht0   0 * 8(%%rcx)                   \n\t" // prefetch c11 + 0*rs_c
	"prefetcht0   0 * 8(%%rcx,%%r9 )             \n\t" // prefetch c11 + 1*rs_c
	"prefetcht0   0 * 8(%%rcx,%%r9 ,2)           \n\t" // prefetch c11 + 2*rs_c
	"prefetcht0   0 * 8(%%rdx)                   \n\t" // prefetch c11 + 3*rs_c
	"prefetcht0   0 * 8(%%rdx,%%r9 )             \n\t" // prefetch c11 + 4*rs_c
	"prefetcht0   0 * 8(%%rdx,%%r9 ,2)           \n\t" // prefetch c11 + 5*rs_c
#endif
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // trsm computation begins here
	"                                            \n\t"
	"                                            \n\t" // Note: contents of b11 are stored as
	"                                            \n\t" // ymm4  ymm5  = ( beta00..07 ) ( beta08..0F )
	"                                            \n\t" // ymm6  ymm7  = ( beta10..17 ) ( beta18..1F )
	"                                            \n\t" // ymm8  ymm9  = ( beta20..27 ) ( beta28..2F )
	"                                            \n\t" // ymm10 ymm11 = ( beta30..37 ) ( beta38..3F )
	"                                            \n\t" // ymm12 ymm13 = ( beta40..47 ) ( beta48..4F )
	"                                            \n\t" // ymm14 ymm15 = ( beta50..57 ) ( beta58..5F )
	"                                            \n\t"
	"                                            \n\t"
	"movq         %6, %%rax                      \n\t" // load address of a11
	"                                            \n\t"
	"movq      %%r11, %%rcx                      \n\t" // recall address of b11
	"movq      %%r14, %%rdx                      \n\t" // recall address of b11+8*cs_b
	"                                            \n\t"
	"leaq   (%%rcx,%%rdi,4), %%rcx               \n\t" // rcx = b11 + (6-1)*rs_b
	"leaq   (%%rcx,%%rdi,1), %%rcx               \n\t"
	"leaq   (%%rdx,%%rdi,4), %%rdx               \n\t" // rdx = b11 + (6-1)*rs_b + 8*cs_b
	"leaq   (%%rdx,%%rdi,1), %%rdx               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0 -------------
	"                                            \n\t"
	"vbroadcastss (5+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = (1/alpha55)
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm14  \n\t" // ymm14 *= (1/alpha55)
	"vmulps           %%ymm0,  %%ymm15, %%ymm15  \n\t" // ymm15 *= (1/alpha55)
	"                                            \n\t"
	"vmovups          %%ymm14, (%%rcx)           \n\t" // store ( beta50..beta57 ) = ymm14
	"vmovups          %%ymm15, (%%rdx)           \n\t" // store ( beta58..beta5F ) = ymm15
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 1 -------------
	"                                            \n\t"
	"vbroadcastss (4+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha45
	"vbroadcastss (4+4*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha44)
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha45 * ymm14
	"vmulps           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha45 * ymm15
	"                                            \n\t"
	"vsubps           %%ymm2,  %%ymm12, %%ymm12  \n\t" // ymm12 -= ymm2
	"vsubps           %%ymm3,  %%ymm13, %%ymm13  \n\t" // ymm13 -= ymm3
	"                                            \n\t"
	"vmulps           %%ymm12, %%ymm1,  %%ymm12  \n\t" // ymm12 *= (1/alpha44)
	"vmulps           %%ymm13, %%ymm1,  %%ymm13  \n\t" // ymm13 *= (1/alpha44)
	"                                            \n\t"
	"vmovups          %%ymm12, (%%rcx)           \n\t" // store ( beta40..beta47 ) = ymm12
	"vmovups          %%ymm13, (%%rdx)           \n\t" // store ( beta48..beta4F ) = ymm13
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 2 -------------
	"                                            \n\t"
	"vbroadcastss (3+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha35
	"vbroadcastss (3+4*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha34
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha35 * ymm14
	"vmulps           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha35 * ymm15
	"                                            \n\t"
	"vbroadcastss (3+3*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = (1/alpha33)
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha34 * ymm12
	"vfmadd231ps      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha34 * ymm13
	"                                            \n\t"
	"vsubps           %%ymm2,  %%ymm10, %%ymm10  \n\t" // ymm10 -= ymm2
	"vsubps           %%ymm3,  %%ymm11, %%ymm11  \n\t" // ymm11 -= ymm3
	"                                            \n\t"
	"vmulps           %%ymm10, %%ymm0,  %%ymm10  \n\t" // ymm10 *= (1/alpha33)
	"vmulps           %%ymm11, %%ymm0,  %%ymm11  \n\t" // ymm11 *= (1/alpha33)
	"                                            \n\t"
	"vmovups          %%ymm10, (%%rcx)           \n\t" // store ( beta30..beta37 ) = ymm10
	"vmovups          %%ymm11, (%%rdx)           \n\t" // store ( beta38..beta3F ) = ymm11
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 3 -------------
	"                                            \n\t"
	"vbroadcastss (2+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha25
	"vbroadcastss (2+4*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha24
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha25 * ymm14
	"vmulps           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha25 * ymm15
	"                                            \n\t"
	"vbroadcastss (2+3*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha23
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha24 * ymm12
	"vfmadd231ps      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha24 * ymm13
	"                                            \n\t"
	"vbroadcastss (2+2*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha22)
	"                                            \n\t"
	"vfmadd231ps      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha23 * ymm10
	"vfmadd231ps      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha23 * ymm11
	"                                            \n\t"
	"vsubps           %%ymm2,  %%ymm8,  %%ymm8   \n\t" // ymm8 -= ymm2
	"vsubps           %%ymm3,  %%ymm9,  %%ymm9   \n\t" // ymm9 -= ymm3
	"                                            \n\t"
	"vmulps           %%ymm8,  %%ymm1,  %%ymm8   \n\t" // ymm8 *= (1/alpha33)
	"vmulps           %%ymm9,  %%ymm1,  %%ymm9   \n\t" // ymm9 *= (1/alpha33)
	"                                            \n\t"
	"vmovups          %%ymm8,  (%%rcx)           \n\t" // store ( beta20..beta27 ) = ymm8
	"vmovups          %%ymm9,  (%%rdx)           \n\t" // store ( beta28..beta2F ) = ymm9
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 4 -------------
	"                                            \n\t"
	"vbroadcastss (1+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha15
	"vbroadcastss (1+4*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha14
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha15 * ymm14
	"vmulps           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha15 * ymm15
	"                                            \n\t"
	"vbroadcastss (1+3*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha13
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha14 * ymm12
	"vfmadd231ps      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha14 * ymm13
	"                                            \n\t"
	"vbroadcastss (1+2*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha12
	"                                            \n\t"
	"vfmadd231ps      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha13 * ymm10
	"vfmadd231ps      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha13 * ymm11
	"                                            \n\t"
	"vbroadcastss (1+1*6)*4(%%rax), %%ymm0       \n\t" // ymm4 = (1/alpha11)
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm8,  %%ymm2   \n\t" // ymm2 += alpha12 * ymm8
	"vfmadd231ps      %%ymm1,  %%ymm9,  %%ymm3   \n\t" // ymm3 += alpha12 * ymm9
	"                                            \n\t"
	"vsubps           %%ymm2,  %%ymm6,  %%ymm6   \n\t" // ymm6 -= ymm2
	"vsubps           %%ymm3,  %%ymm7,  %%ymm7   \n\t" // ymm7 -= ymm3
	"                                            \n\t"
	"vmulps           %%ymm6,  %%ymm0,  %%ymm6   \n\t" // ymm6 *= (1/alpha44)
	"vmulps           %%ymm7,  %%ymm0,  %%ymm7   \n\t" // ymm7 *= (1/alpha44)
	"                                            \n\t"
	"vmovups          %%ymm6,  (%%rcx)           \n\t" // store ( beta10..beta17 ) = ymm6
	"vmovups          %%ymm7,  (%%rdx)           \n\t" // store ( beta18..beta1F ) = ymm7
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 5 -------------
	"                                            \n\t"
	"vbroadcastss (0+5*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha05
	"vbroadcastss (0+4*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha04
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha05 * ymm14
	"vmulps           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha05 * ymm15
	"                                            \n\t"
	"vbroadcastss (0+3*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha03
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha04 * ymm12
	"vfmadd231ps      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha04 * ymm13
	"                                            \n\t"
	"vbroadcastss (0+2*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = alpha02
	"                                            \n\t"
	"vfmadd231ps      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha03 * ymm10
	"vfmadd231ps      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha03 * ymm11
	"                                            \n\t"
	"vbroadcastss (0+1*6)*4(%%rax), %%ymm0       \n\t" // ymm0 = alpha01
	"                                            \n\t"
	"vfmadd231ps      %%ymm1,  %%ymm8,  %%ymm2   \n\t" // ymm2 += alpha02 * ymm8
	"vfmadd231ps      %%ymm1,  %%ymm9,  %%ymm3   \n\t" // ymm3 += alpha02 * ymm9
	"                                            \n\t"
	"vbroadcastss (0+0*6)*4(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha00)
	"                                            \n\t"
	"vfmadd231ps      %%ymm0,  %%ymm6,  %%ymm2   \n\t" // ymm2 += alpha01 * ymm6
	"vfmadd231ps      %%ymm0,  %%ymm7,  %%ymm3   \n\t" // ymm3 += alpha01 * ymm7
	"                                            \n\t"
	"vsubps           %%ymm2,  %%ymm4,  %%ymm4   \n\t" // ymm4 -= ymm2
	"vsubps           %%ymm3,  %%ymm5,  %%ymm5   \n\t" // ymm5 -= ymm3
	"                                            \n\t"
	"vmulps           %%ymm4,  %%ymm1,  %%ymm4   \n\t" // ymm4 *= (1/alpha00)
	"vmulps           %%ymm5,  %%ymm1,  %%ymm5   \n\t" // ymm5 *= (1/alpha00)
	"                                            \n\t"
	"vmovups          %%ymm4,  (%%rcx)           \n\t" // store ( beta00..beta07 ) = ymm4
	"vmovups          %%ymm5,  (%%rdx)           \n\t" // store ( beta08..beta0F ) = ymm5
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq             %%r8,  %%rcx               \n\t" // load address of c11 from r8
	"movq             %%r9,  %%rdi               \n\t" // load rs_c (in bytes) from r9
	"movq            %%r10,  %%rsi               \n\t" // load cs_c (in bytes) from r10
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,8), %%rdx               \n\t" // load address of c11 + 8*cs_c;
	"leaq   (%%rcx,%%rdi,4), %%r14               \n\t" // load address of c11 + 4*rs_c;
	"                                            \n\t"
	"                                            \n\t" // These are used in the macros below.
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
	"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
	"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $4, %%rsi                        \n\t" // set ZF if (4*cs_c) == 4.
	"jz      .SROWSTORED                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $4, %%rdi                        \n\t" // set ZF if (4*rs_c) == 4.
	"jz      .SCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // if neither row- or column-
	"                                            \n\t" // stored, use general case.
	".SGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm4,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm6,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm8,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm10, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm12, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm14, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c11 + 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm5,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm7,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm9,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm11, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm13, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm15, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SROWSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm6,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm7,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm8,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm9,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm12, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm13, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm14, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm15, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vunpcklps         %%ymm6,  %%ymm4,  %%ymm0  \n\t"
	"vunpcklps         %%ymm10, %%ymm8,  %%ymm1  \n\t"
	"vshufps    $0x4e, %%ymm1,  %%ymm0,  %%ymm2  \n\t"
	"vblendps   $0xcc, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"vblendps   $0x33, %%ymm2,  %%ymm1,  %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovups           %%xmm0, (%%rcx        )   \n\t" // store ( gamma00..gamma30 )
	"vmovups           %%xmm1, (%%rcx,%%rsi,1)   \n\t" // store ( gamma01..gamma31 )
	"vmovups           %%xmm2, (%%rcx,%%rsi,4)   \n\t" // store ( gamma04..gamma34 )
	"vmovups           %%xmm3, (%%rcx,%%r15  )   \n\t" // store ( gamma05..gamma35 )
	"                                            \n\t"
	"                                            \n\t"
	"vunpckhps         %%ymm6,  %%ymm4,  %%ymm0  \n\t"
	"vunpckhps         %%ymm10, %%ymm8,  %%ymm1  \n\t"
	"vshufps    $0x4e, %%ymm1,  %%ymm0,  %%ymm2  \n\t"
	"vblendps   $0xcc, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"vblendps   $0x33, %%ymm2,  %%ymm1,  %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovups           %%xmm0, (%%rcx,%%rsi,2)   \n\t" // store ( gamma02..gamma32 )
	"vmovups           %%xmm1, (%%rcx,%%r13  )   \n\t" // store ( gamma03..gamma33 )
	"vmovups           %%xmm2, (%%rcx,%%r13,2)   \n\t" // store ( gamma06..gamma36 )
	"vmovups           %%xmm3, (%%rcx,%%r10  )   \n\t" // store ( gamma07..gamma37 )
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,8), %%rcx               \n\t" // rcx += 8*cs_c
	"                                            \n\t"
	"vunpcklps         %%ymm14, %%ymm12, %%ymm0  \n\t"
	"vunpckhps         %%ymm14, %%ymm12, %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovlpd           %%xmm0, (%%r14        )   \n\t" // store ( gamma40..gamma50 )
	"vmovhpd           %%xmm0, (%%r14,%%rsi,1)   \n\t" // store ( gamma41..gamma51 )
	"vmovlpd           %%xmm1, (%%r14,%%rsi,2)   \n\t" // store ( gamma42..gamma52 )
	"vmovhpd           %%xmm1, (%%r14,%%r13  )   \n\t" // store ( gamma43..gamma53 )
	"vmovlpd           %%xmm2, (%%r14,%%rsi,4)   \n\t" // store ( gamma44..gamma54 )
	"vmovhpd           %%xmm2, (%%r14,%%r15  )   \n\t" // store ( gamma45..gamma55 )
	"vmovlpd           %%xmm3, (%%r14,%%r13,2)   \n\t" // store ( gamma46..gamma56 )
	"vmovhpd           %%xmm3, (%%r14,%%r10  )   \n\t" // store ( gamma47..gamma57 )
	"                                            \n\t"
	"leaq   (%%r14,%%rsi,8), %%r14               \n\t" // r14 += 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vunpcklps         %%ymm7,  %%ymm5,  %%ymm0  \n\t"
	"vunpcklps         %%ymm11, %%ymm9,  %%ymm1  \n\t"
	"vshufps    $0x4e, %%ymm1,  %%ymm0,  %%ymm2  \n\t"
	"vblendps   $0xcc, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"vblendps   $0x33, %%ymm2,  %%ymm1,  %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovups           %%xmm0, (%%rcx        )   \n\t" // store ( gamma08..gamma38 )
	"vmovups           %%xmm1, (%%rcx,%%rsi,1)   \n\t" // store ( gamma09..gamma39 )
	"vmovups           %%xmm2, (%%rcx,%%rsi,4)   \n\t" // store ( gamma0C..gamma3C )
	"vmovups           %%xmm3, (%%rcx,%%r15  )   \n\t" // store ( gamma0D..gamma3D )
	"                                            \n\t"
	"vunpckhps         %%ymm7,  %%ymm5,  %%ymm0  \n\t"
	"vunpckhps         %%ymm11, %%ymm9,  %%ymm1  \n\t"
	"vshufps    $0x4e, %%ymm1,  %%ymm0,  %%ymm2  \n\t"
	"vblendps   $0xcc, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"vblendps   $0x33, %%ymm2,  %%ymm1,  %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovups           %%xmm0, (%%rcx,%%rsi,2)   \n\t" // store ( gamma0A..gamma3A )
	"vmovups           %%xmm1, (%%rcx,%%r13  )   \n\t" // store ( gamma0B..gamma3B )
	"vmovups           %%xmm2, (%%rcx,%%r13,2)   \n\t" // store ( gamma0E..gamma3E )
	"vmovups           %%xmm3, (%%rcx,%%r10  )   \n\t" // store ( gamma0F..gamma3F )
	"                                            \n\t"
	//"leaq   (%%rcx,%%rsi,8), %%rcx               \n\t" // rcx += 8*cs_c
	"                                            \n\t"
	"vunpcklps         %%ymm15, %%ymm13, %%ymm0  \n\t"
	"vunpckhps         %%ymm15, %%ymm13, %%ymm1  \n\t"
	"                                            \n\t"
	"vextractf128 $0x1, %%ymm0, %%xmm2           \n\t"
	"vextractf128 $0x1, %%ymm1, %%xmm3           \n\t"
	"                                            \n\t"
	"vmovlpd           %%xmm0, (%%r14        )   \n\t" // store ( gamma48..gamma58 )
	"vmovhpd           %%xmm0, (%%r14,%%rsi,1)   \n\t" // store ( gamma49..gamma59 )
	"vmovlpd           %%xmm1, (%%r14,%%rsi,2)   \n\t" // store ( gamma4A..gamma5A )
	"vmovhpd           %%xmm1, (%%r14,%%r13  )   \n\t" // store ( gamma4B..gamma5B )
	"vmovlpd           %%xmm2, (%%r14,%%rsi,4)   \n\t" // store ( gamma4C..gamma5C )
	"vmovhpd           %%xmm2, (%%r14,%%r15  )   \n\t" // store ( gamma4D..gamma5D )
	"vmovlpd           %%xmm3, (%%r14,%%r13,2)   \n\t" // store ( gamma4E..gamma5E )
	"vmovhpd           %%xmm3, (%%r14,%%r10  )   \n\t" // store ( gamma4F..gamma5F )
	"                                            \n\t"
	//"leaq   (%%r14,%%rsi,8), %%r14               \n\t" // r14 += 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SDONE:                                     \n\t"
	"                                            \n\t"
	"vzeroupper                                  \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a10),    // 2
	  "m" (b01),    // 3
	  "m" (beta),   // 4
	  "m" (alpha),  // 5
	  "m" (a11),    // 6
	  "m" (b11),    // 7
	  "m" (c11),    // 8
	  "m" (rs_c),   // 9
	  "m" (cs_c)    // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}



#define DGEMM_OUTPUT_GS_BETA_NZ \
	"vextractf128  $1, %%ymm0,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm0,  (%%rcx        )  \n\t" \
	"vmovhpd           %%xmm0,  (%%rcx,%%rsi  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%rsi,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r13  )  \n\t" /*\
	"vextractf128  $1, %%ymm2,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm2,  (%%rcx,%%rsi,4)  \n\t" \
	"vmovhpd           %%xmm2,  (%%rcx,%%r15  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%r13,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r10  )  \n\t"*/

void bli_dgemmtrsm_u_zen_asm_6x8
(
    dim_t               k,
    double*    restrict alpha,
    double*    restrict a10,
    double*    restrict a11,
    double*    restrict b01,
    double*    restrict b11,
    double*    restrict c11, inc_t rs_c, inc_t cs_c,
    auxinfo_t* restrict data,
    cntx_t*    restrict cntx
)
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	uint64_t   k_iter = k / 4;
	uint64_t   k_left = k % 4;

	double*    beta   = bli_dm1;

	__asm__ volatile
	(
	"                                            \n\t"
	"vzeroall                                    \n\t" // zero all xmm/ymm registers.
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"                                            \n\t"
	"addq           $32 * 4, %%rbx               \n\t"
	"                                            \n\t" // initialize loop by pre-loading
	"vmovapd           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"movq                %7, %%rcx               \n\t" // load address of b11
	"movq                $8, %%rdi               \n\t" // set rs_b = PACKNR = 8
	"leaq        (,%%rdi,8), %%rdi               \n\t" // rs_b *= sizeof(double)
	"                                            \n\t"
	"                                            \n\t" // NOTE: c11, rs_c, and cs_c aren't
	"                                            \n\t" // needed for a while, but we load
	"                                            \n\t" // them now to avoid stalling later.
	"movq                %8, %%r8                \n\t" // load address of c11
	"movq                %9, %%r9                \n\t" // load rs_c
	"leaq        (,%%r9 ,8), %%r9                \n\t" // rs_c *= sizeof(double)
	"movq               %10, %%r10               \n\t" // load cs_c
	"leaq        (,%%r10,8), %%r10               \n\t" // cs_c *= sizeof(double)
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0   64 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd       4 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd           -2 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd           -1 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vbroadcastsd       6 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       7 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       8 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       9 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      10 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      11 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd            0 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd            1 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0   76 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd      12 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      13 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd      14 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      15 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      16 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      17 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd            2 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd            3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vbroadcastsd      18 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      19 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd      20 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      21 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      22 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd      23 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq           $4 * 6 * 8, %%rax            \n\t" // a += 4*6 (unroll x mr)
	"addq           $4 * 8 * 8, %%rbx            \n\t" // b += 4*8 (unroll x nr)
	"                                            \n\t"
	"vmovapd           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0   64 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd       4 *  8(%%rax), %%ymm2    \n\t"
	"vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq           $1 * 6 * 8, %%rax            \n\t" // a += 1*6 (unroll x mr)
	"addq           $1 * 8 * 8, %%rbx            \n\t" // b += 1*8 (unroll x nr)
	"                                            \n\t"
	"vmovapd           -4 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rbx), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm4..ymm15 = -a10 * b01
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of alpha
	"vbroadcastsd    (%%rbx), %%ymm3             \n\t" // load alpha and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                $1, %%rsi               \n\t" // set cs_b = 1
	"leaq        (,%%rsi,8), %%rsi               \n\t" // cs_b *= sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of b11 + 4*cs_b
	"                                            \n\t"
	"movq             %%rcx, %%r11               \n\t" // save rcx = b11        for later
	"movq             %%rdx, %%r14               \n\t" // save rdx = b11+4*cs_b for later
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // b11 := alpha * b11 - a10 * b01
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm4    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm5    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm6    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm7    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm8    \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm9    \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm10   \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm11   \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm12   \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm13   \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vfmsub231pd      (%%rcx), %%ymm3, %%ymm14   \n\t"
  //"addq      %%rdi, %%rcx                      \n\t"
	"vfmsub231pd      (%%rdx), %%ymm3, %%ymm15   \n\t"
  //"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // prefetch c11
	"                                            \n\t"
#if 0
	"movq             %%r8,  %%rcx               \n\t" // load address of c11 from r8
	"                                            \n\t" // Note: r9 = rs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%r9 ,%%r9 ,2), %%r13               \n\t" // r13 = 3*rs_c;
	"leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c11 + 3*rs_c;
	"                                            \n\t"
	"prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c11 + 0*rs_c
	"prefetcht0   7 * 8(%%rcx,%%r9 )             \n\t" // prefetch c11 + 1*rs_c
	"prefetcht0   7 * 8(%%rcx,%%r9 ,2)           \n\t" // prefetch c11 + 2*rs_c
	"prefetcht0   7 * 8(%%rdx)                   \n\t" // prefetch c11 + 3*rs_c
	"prefetcht0   7 * 8(%%rdx,%%r9 )             \n\t" // prefetch c11 + 4*rs_c
	"prefetcht0   7 * 8(%%rdx,%%r9 ,2)           \n\t" // prefetch c11 + 5*rs_c
#endif
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // trsm computation begins here
	"                                            \n\t"
	"                                            \n\t" // Note: contents of b11 are stored as
	"                                            \n\t" // ymm4  ymm5  = ( beta00..03 ) ( beta04..07 )
	"                                            \n\t" // ymm6  ymm7  = ( beta10..13 ) ( beta14..17 )
	"                                            \n\t" // ymm8  ymm9  = ( beta20..23 ) ( beta24..27 )
	"                                            \n\t" // ymm10 ymm11 = ( beta30..33 ) ( beta34..37 )
	"                                            \n\t" // ymm12 ymm13 = ( beta40..43 ) ( beta44..47 )
	"                                            \n\t" // ymm14 ymm15 = ( beta50..53 ) ( beta54..57 )
	"                                            \n\t"
	"                                            \n\t"
	"movq         %6, %%rax                      \n\t" // load address of a11
	"                                            \n\t"
	"movq      %%r11, %%rcx                      \n\t" // recall address of b11
	"movq      %%r14, %%rdx                      \n\t" // recall address of b11+4*cs_b
	"                                            \n\t"
	"leaq   (%%rcx,%%rdi,4), %%rcx               \n\t" // rcx = b11 + (6-1)*rs_b
	"leaq   (%%rcx,%%rdi,1), %%rcx               \n\t"
	"leaq   (%%rdx,%%rdi,4), %%rdx               \n\t" // rdx = b11 + (6-1)*rs_b + 4*cs_b
	"leaq   (%%rdx,%%rdi,1), %%rdx               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0 -------------
	"                                            \n\t"
	"vbroadcastsd (5+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = (1/alpha55)
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t" // ymm14 *= (1/alpha55)
	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t" // ymm15 *= (1/alpha55)
	"                                            \n\t"
	"vmovupd          %%ymm14, (%%rcx)           \n\t" // store ( beta50..beta53 ) = ymm14
	"vmovupd          %%ymm15, (%%rdx)           \n\t" // store ( beta54..beta57 ) = ymm15
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 1 -------------
	"                                            \n\t"
	"vbroadcastsd (4+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha45
	"vbroadcastsd (4+4*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha44)
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha45 * ymm14
	"vmulpd           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha45 * ymm15
	"                                            \n\t"
	"vsubpd           %%ymm2,  %%ymm12, %%ymm12  \n\t" // ymm12 -= ymm2
	"vsubpd           %%ymm3,  %%ymm13, %%ymm13  \n\t" // ymm13 -= ymm3
	"                                            \n\t"
	"vmulpd           %%ymm12, %%ymm1,  %%ymm12  \n\t" // ymm12 *= (1/alpha44)
	"vmulpd           %%ymm13, %%ymm1,  %%ymm13  \n\t" // ymm13 *= (1/alpha44)
	"                                            \n\t"
	"vmovupd          %%ymm12, (%%rcx)           \n\t" // store ( beta40..beta43 ) = ymm12
	"vmovupd          %%ymm13, (%%rdx)           \n\t" // store ( beta44..beta47 ) = ymm13
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 2 -------------
	"                                            \n\t"
	"vbroadcastsd (3+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha35
	"vbroadcastsd (3+4*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha34
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha35 * ymm14
	"vmulpd           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha35 * ymm15
	"                                            \n\t"
	"vbroadcastsd (3+3*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = (1/alpha33)
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha34 * ymm12
	"vfmadd231pd      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha34 * ymm13
	"                                            \n\t"
	"vsubpd           %%ymm2,  %%ymm10, %%ymm10  \n\t" // ymm10 -= ymm2
	"vsubpd           %%ymm3,  %%ymm11, %%ymm11  \n\t" // ymm11 -= ymm3
	"                                            \n\t"
	"vmulpd           %%ymm10, %%ymm0,  %%ymm10  \n\t" // ymm10 *= (1/alpha33)
	"vmulpd           %%ymm11, %%ymm0,  %%ymm11  \n\t" // ymm11 *= (1/alpha33)
	"                                            \n\t"
	"vmovupd          %%ymm10, (%%rcx)           \n\t" // store ( beta30..beta33 ) = ymm10
	"vmovupd          %%ymm11, (%%rdx)           \n\t" // store ( beta34..beta37 ) = ymm11
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 3 -------------
	"                                            \n\t"
	"vbroadcastsd (2+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha25
	"vbroadcastsd (2+4*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha24
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha25 * ymm14
	"vmulpd           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha25 * ymm15
	"                                            \n\t"
	"vbroadcastsd (2+3*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha23
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha24 * ymm12
	"vfmadd231pd      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha24 * ymm13
	"                                            \n\t"
	"vbroadcastsd (2+2*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha22)
	"                                            \n\t"
	"vfmadd231pd      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha23 * ymm10
	"vfmadd231pd      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha23 * ymm11
	"                                            \n\t"
	"vsubpd           %%ymm2,  %%ymm8,  %%ymm8   \n\t" // ymm8 -= ymm2
	"vsubpd           %%ymm3,  %%ymm9,  %%ymm9   \n\t" // ymm9 -= ymm3
	"                                            \n\t"
	"vmulpd           %%ymm8,  %%ymm1,  %%ymm8   \n\t" // ymm8 *= (1/alpha33)
	"vmulpd           %%ymm9,  %%ymm1,  %%ymm9   \n\t" // ymm9 *= (1/alpha33)
	"                                            \n\t"
	"vmovupd          %%ymm8,  (%%rcx)           \n\t" // store ( beta20..beta23 ) = ymm8
	"vmovupd          %%ymm9,  (%%rdx)           \n\t" // store ( beta24..beta27 ) = ymm9
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 4 -------------
	"                                            \n\t"
	"vbroadcastsd (1+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha15
	"vbroadcastsd (1+4*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha14
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha15 * ymm14
	"vmulpd           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha15 * ymm15
	"                                            \n\t"
	"vbroadcastsd (1+3*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha13
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha14 * ymm12
	"vfmadd231pd      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha14 * ymm13
	"                                            \n\t"
	"vbroadcastsd (1+2*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha12
	"                                            \n\t"
	"vfmadd231pd      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha13 * ymm10
	"vfmadd231pd      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha13 * ymm11
	"                                            \n\t"
	"vbroadcastsd (1+1*6)*8(%%rax), %%ymm0       \n\t" // ymm4 = (1/alpha11)
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm8,  %%ymm2   \n\t" // ymm2 += alpha12 * ymm8
	"vfmadd231pd      %%ymm1,  %%ymm9,  %%ymm3   \n\t" // ymm3 += alpha12 * ymm9
	"                                            \n\t"
	"vsubpd           %%ymm2,  %%ymm6,  %%ymm6   \n\t" // ymm6 -= ymm2
	"vsubpd           %%ymm3,  %%ymm7,  %%ymm7   \n\t" // ymm7 -= ymm3
	"                                            \n\t"
	"vmulpd           %%ymm6,  %%ymm0,  %%ymm6   \n\t" // ymm6 *= (1/alpha44)
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm7   \n\t" // ymm7 *= (1/alpha44)
	"                                            \n\t"
	"vmovupd          %%ymm6,  (%%rcx)           \n\t" // store ( beta10..beta13 ) = ymm6
	"vmovupd          %%ymm7,  (%%rdx)           \n\t" // store ( beta14..beta17 ) = ymm7
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t" // iteration 5 -------------
	"                                            \n\t"
	"vbroadcastsd (0+5*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha05
	"vbroadcastsd (0+4*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha04
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm2   \n\t" // ymm2 = alpha05 * ymm14
	"vmulpd           %%ymm0,  %%ymm15, %%ymm3   \n\t" // ymm3 = alpha05 * ymm15
	"                                            \n\t"
	"vbroadcastsd (0+3*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha03
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm12, %%ymm2   \n\t" // ymm2 += alpha04 * ymm12
	"vfmadd231pd      %%ymm1,  %%ymm13, %%ymm3   \n\t" // ymm3 += alpha04 * ymm13
	"                                            \n\t"
	"vbroadcastsd (0+2*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = alpha02
	"                                            \n\t"
	"vfmadd231pd      %%ymm0,  %%ymm10, %%ymm2   \n\t" // ymm2 += alpha03 * ymm10
	"vfmadd231pd      %%ymm0,  %%ymm11, %%ymm3   \n\t" // ymm3 += alpha03 * ymm11
	"                                            \n\t"
	"vbroadcastsd (0+1*6)*8(%%rax), %%ymm0       \n\t" // ymm0 = alpha01
	"                                            \n\t"
	"vfmadd231pd      %%ymm1,  %%ymm8,  %%ymm2   \n\t" // ymm2 += alpha02 * ymm8
	"vfmadd231pd      %%ymm1,  %%ymm9,  %%ymm3   \n\t" // ymm3 += alpha02 * ymm9
	"                                            \n\t"
	"vbroadcastsd (0+0*6)*8(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha00)
	"                                            \n\t"
	"vfmadd231pd      %%ymm0,  %%ymm6,  %%ymm2   \n\t" // ymm2 += alpha01 * ymm6
	"vfmadd231pd      %%ymm0,  %%ymm7,  %%ymm3   \n\t" // ymm3 += alpha01 * ymm7
	"                                            \n\t"
	"vsubpd           %%ymm2,  %%ymm4,  %%ymm4   \n\t" // ymm4 -= ymm2
	"vsubpd           %%ymm3,  %%ymm5,  %%ymm5   \n\t" // ymm5 -= ymm3
	"                                            \n\t"
	"vmulpd           %%ymm4,  %%ymm1,  %%ymm4   \n\t" // ymm4 *= (1/alpha00)
	"vmulpd           %%ymm5,  %%ymm1,  %%ymm5   \n\t" // ymm5 *= (1/alpha00)
	"                                            \n\t"
	"vmovupd          %%ymm4,  (%%rcx)           \n\t" // store ( beta00..beta03 ) = ymm4
	"vmovupd          %%ymm5,  (%%rdx)           \n\t" // store ( beta04..beta07 ) = ymm5
	"subq      %%rdi, %%rcx                      \n\t" // rcx -= rs_b
	"subq      %%rdi, %%rdx                      \n\t" // rdx -= rs_b
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq             %%r8,  %%rcx               \n\t" // load address of c11 from r8
	"movq             %%r9,  %%rdi               \n\t" // load rs_c (in bytes) from r9
	"movq            %%r10,  %%rsi               \n\t" // load cs_c (in bytes) from r10
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c11 + 4*cs_c;
	"leaq   (%%rcx,%%rdi,4), %%r14               \n\t" // load address of c11 + 4*rs_c;
	"                                            \n\t"
	"                                            \n\t" // These are used in the macros below.
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
  //"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
  //"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
	"jz      .DROWSTORED                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $8, %%rdi                        \n\t" // set ZF if (8*rs_c) == 8.
	"jz      .DCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // if neither row- or column-
	"                                            \n\t" // stored, use general case.
	".DGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm4,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm6,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm8,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm10, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c11 + 4*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm5,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm7,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm9,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm11, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c11 += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DROWSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm6,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm7,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm8,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm9,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm12, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm13, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm14, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm15, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vunpcklpd         %%ymm6,  %%ymm4,  %%ymm0  \n\t"
	"vunpckhpd         %%ymm6,  %%ymm4,  %%ymm1  \n\t"
	"vunpcklpd         %%ymm10, %%ymm8,  %%ymm2  \n\t"
	"vunpckhpd         %%ymm10, %%ymm8,  %%ymm3  \n\t"
	"vinsertf128 $0x1, %%xmm2,  %%ymm0,  %%ymm4  \n\t"
	"vinsertf128 $0x1, %%xmm3,  %%ymm1,  %%ymm6  \n\t"
	"vperm2f128 $0x31, %%ymm2,  %%ymm0,  %%ymm8  \n\t"
	"vperm2f128 $0x31, %%ymm3,  %%ymm1,  %%ymm10 \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm4,  (%%rcx        )   \n\t"
	"vmovupd          %%ymm6,  (%%rcx,%%rsi  )   \n\t"
	"vmovupd          %%ymm8,  (%%rcx,%%rsi,2)   \n\t"
	"vmovupd          %%ymm10, (%%rcx,%%r13  )   \n\t"
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rcx               \n\t"
	"                                            \n\t"
	"vunpcklpd         %%ymm14, %%ymm12, %%ymm0  \n\t"
	"vunpckhpd         %%ymm14, %%ymm12, %%ymm1  \n\t"
	"vextractf128         $0x1, %%ymm0,  %%xmm2  \n\t"
	"vextractf128         $0x1, %%ymm1,  %%xmm3  \n\t"
	"                                            \n\t"
	"vmovupd          %%xmm0,  (%%r14        )   \n\t"
	"vmovupd          %%xmm1,  (%%r14,%%rsi  )   \n\t"
	"vmovupd          %%xmm2,  (%%r14,%%rsi,2)   \n\t"
	"vmovupd          %%xmm3,  (%%r14,%%r13  )   \n\t"
	"                                            \n\t"
	"leaq   (%%r14,%%rsi,4), %%r14               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vunpcklpd         %%ymm7,  %%ymm5,  %%ymm0  \n\t"
	"vunpckhpd         %%ymm7,  %%ymm5,  %%ymm1  \n\t"
	"vunpcklpd         %%ymm11, %%ymm9,  %%ymm2  \n\t"
	"vunpckhpd         %%ymm11, %%ymm9,  %%ymm3  \n\t"
	"vinsertf128 $0x1, %%xmm2,  %%ymm0,  %%ymm5  \n\t"
	"vinsertf128 $0x1, %%xmm3,  %%ymm1,  %%ymm7  \n\t"
	"vperm2f128 $0x31, %%ymm2,  %%ymm0,  %%ymm9  \n\t"
	"vperm2f128 $0x31, %%ymm3,  %%ymm1,  %%ymm11 \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm5,  (%%rcx        )   \n\t"
	"vmovupd          %%ymm7,  (%%rcx,%%rsi  )   \n\t"
	"vmovupd          %%ymm9,  (%%rcx,%%rsi,2)   \n\t"
	"vmovupd          %%ymm11, (%%rcx,%%r13  )   \n\t"
	"                                            \n\t"
	//"leaq   (%%rcx,%%rsi,4), %%rcx               \n\t"
	"                                            \n\t"
	"vunpcklpd         %%ymm15, %%ymm13, %%ymm0  \n\t"
	"vunpckhpd         %%ymm15, %%ymm13, %%ymm1  \n\t"
	"vextractf128         $0x1, %%ymm0,  %%xmm2  \n\t"
	"vextractf128         $0x1, %%ymm1,  %%xmm3  \n\t"
	"                                            \n\t"
	"vmovupd          %%xmm0,  (%%r14        )   \n\t"
	"vmovupd          %%xmm1,  (%%r14,%%rsi  )   \n\t"
	"vmovupd          %%xmm2,  (%%r14,%%rsi,2)   \n\t"
	"vmovupd          %%xmm3,  (%%r14,%%r13  )   \n\t"
	"                                            \n\t"
	//"leaq   (%%r14,%%rsi,4), %%r14               \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
	"vzeroupper                                  \n\t"
	"                                            \n\t"


	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a10),    // 2
	  "m" (b01),    // 3
	  "m" (beta),   // 4
	  "m" (alpha),  // 5
	  "m" (a11),    // 6
	  "m" (b11),    // 7
	  "m" (c11),    // 8
	  "m" (rs_c),   // 9
	  "m" (cs_c)    // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}


