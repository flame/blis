/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Advanced Micro Devices, Inc.

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


void bli_sgemmtrsm_l_6x16
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

    uint64_t   k_iter = k / 4;
    uint64_t   k_left = k % 4;

    float *beta = (float*) ((BLIS_MINUS_ONE).buffer);

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
    "movq                %7, %%rcx               \n\t" // load address of c
    "movq                $16, %%rdi               \n\t" // load rs_c
    "leaq        (,%%rdi,4), %%rdi               \n\t" // rs_c *= sizeof(float)
    "                                            \n\t"
    "leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*rs_c;
    "leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c + 3*rs_c;
    "prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*rs_c
    "prefetcht0   7 * 8(%%rdx)                   \n\t" // prefetch c + 3*rs_c
    "prefetcht0   7 * 8(%%rdx,%%rdi)             \n\t" // prefetch c + 4*rs_c
    "prefetcht0   7 * 8(%%rdx,%%rdi,2)           \n\t" // prefetch c + 5*rs_c
    "                                            \n\t"
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
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of beta
    "movq         %5, %%rbx                      \n\t" // load address of alpha
    "vbroadcastss    (%%rax), %%ymm0             \n\t" // load beta and duplicate
    "vbroadcastss    (%%rbx), %%ymm3             \n\t" // load alpha and duplicate
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm4   \n\t" // scale by beta
    "vmulps           %%ymm0,  %%ymm5,  %%ymm5   \n\t"
    "vmulps           %%ymm0,  %%ymm6,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm7,  %%ymm7   \n\t"
    "vmulps           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
    "vmulps           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
    "vmulps           %%ymm0,  %%ymm10, %%ymm10  \n\t"
    "vmulps           %%ymm0,  %%ymm11, %%ymm11  \n\t"
    "vmulps           %%ymm0,  %%ymm12, %%ymm12  \n\t"
    "vmulps           %%ymm0,  %%ymm13, %%ymm13  \n\t"
    "vmulps           %%ymm0,  %%ymm14, %%ymm14  \n\t"
    "vmulps           %%ymm0,  %%ymm15, %%ymm15  \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                $1, %%rsi               \n\t" // load cs_c
    "leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = cs_c * sizeof(float)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,8), %%rdx               \n\t" // load address of c +  8*cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // generate the bll data
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm4   \n\t" // mutiply by alpha & accumulate into ymm4 to 15
    "addq      %%rdi, %%rcx                      \n\t" // increment the pointer by row stride
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm5   \n\t" // ymm4 | ymm5
    "addq      %%rdi, %%rdx                      \n\t" // increment the pointer 8*cs_c
    "                                            \n\t" // The data layout of 6 x 16 block is as follows
    "                                            \n\t" 
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm6   \n\t" // ymm6 | ymm7
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm7   \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm8   \n\t" // ymm8 | ymm9
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm9   \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm10  \n\t" // ymm10| ymm11
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm11  \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm12  \n\t" // ymm12| ymm13
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm13  \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231ps      (%%rcx), %%ymm3,  %%ymm14  \n\t" // ymm14| ymm15
    "vfmadd231ps      (%%rdx), %%ymm3,  %%ymm15  \n\t" // ymm4 upto ymm15 have now been mutiplied by alpha
    "                                            \n\t" // TRSM computation starts
    "                                            \n\t"
    "movq         %6, %%rax                      \n\t" // load address of a11
    "vbroadcastss    (%%rax), %%ymm0             \n\t" // load a11 and duplicate
    "vmulps           %%ymm0,  %%ymm4, %%ymm4    \n\t" // ymm4  *= (1/alpha00)
    "vmulps           %%ymm0,  %%ymm5, %%ymm5    \n\t" // ymm5  *= (1/alpha00)
    "                                            \n\t"
    "movq         %7, %%rcx                      \n\t" // load address of c11
    "movq                $4, %%rdi               \n\t" // initialize float size as 4
    "                                            \n\t"
    "vmovups          %%ymm4, (%%rcx)            \n\t" // store in cll the final result of ymm4 & ymm5
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm5, (%%rcx)            \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t" //iteration 1
    "vbroadcastss     1 * 4(%%rax), %%ymm0       \n\t" // ymm0 = alpha10
    "vbroadcastss     7 * 4(%%rax), %%ymm1       \n\t" // ymm1 = (1/alpha11)
    "vmulps           %%ymm0,  %%ymm4, %%ymm2    \n\t" // ymm2 = alpha10 * alpha00* (beta00 .. beta07)
    "vmulps           %%ymm0,  %%ymm5, %%ymm3    \n\t" // ymm3 = alpha10 * alpha00 * (beta08 ... beta015)
    "                                            \n\t"
    "vsubps           %%ymm2,  %%ymm6, %%ymm6    \n\t" // ymm6 =  [beta10 ... beta17] - ymm2
    "vsubps           %%ymm3,  %%ymm7, %%ymm7    \n\t" // ymm7 = [beta18 ... beta1,15] - ymm3
    "                                            \n\t"
    "vmulps           %%ymm6,  %%ymm1, %%ymm6    \n\t" // ymm6 = 1/alpha11 * ymm6
    "vmulps           %%ymm7,  %%ymm1, %%ymm7    \n\t" // ymm7 = 1/alpha11 * ymm7
    "                                            \n\t"
    "vmovups          %%ymm6, (%%rcx)            \n\t" // store in cll the final result of ymm6 & ymm7 
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm7, (%%rcx)            \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t" //iteration 2
    "vbroadcastss     2 * 4(%%rax), %%ymm0       \n\t" // ymm0 = alpha20
    "vbroadcastss     8 * 4(%%rax), %%ymm1       \n\t" // ymm1 = alpha21
    "vmulps           %%ymm0,  %%ymm4, %%ymm2    \n\t" // ymm2 = alpha20 * [alpha00* (beta00 .. beta07)]
    "vmulps           %%ymm0,  %%ymm5, %%ymm3    \n\t" // ymm3 = alpha20 * [alpha00* (beta08 .. beta015)]
    "vbroadcastss     14 * 4(%%rax), %%ymm0      \n\t" // ymm0 = 1/alpha22
    "vfmadd231ps      %%ymm1, %%ymm6,  %%ymm2    \n\t" // ymm2 += alpha21 * ymm6
    "vfmadd231ps      %%ymm1, %%ymm7,  %%ymm3    \n\t" // ymm3 += alpha21 * ymm7
    "                                            \n\t"
    "vsubps           %%ymm2,  %%ymm8, %%ymm8    \n\t" // ymm8 -= ymm2
    "vsubps           %%ymm3,  %%ymm9, %%ymm9    \n\t" // ymm9 -= ymm3
    "                                            \n\t"
    "vmulps           %%ymm8,  %%ymm0, %%ymm8    \n\t" // ymm8 *= 1/alpha22
    "vmulps           %%ymm9,  %%ymm0, %%ymm9    \n\t" // ymm9 *= 1/alpha22
    "                                            \n\t"
    "vmovups          %%ymm8, (%%rcx)            \n\t" // store in cll the final result of ymm8 & ymm9
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm9, (%%rcx)            \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t" //iteration 3
    "vbroadcastss     3 * 4(%%rax), %%ymm0       \n\t" // ymm0 = alpha30
    "vbroadcastss     9 * 4(%%rax), %%ymm1       \n\t" // ymm1 = alpha31
    "vmulps           %%ymm0,  %%ymm4, %%ymm2    \n\t" // ymm2 = alpha30 * ymm4
    "vmulps           %%ymm0,  %%ymm5, %%ymm3    \n\t" // ymm3 = alpha30 * ymm5
    "vbroadcastss     15 * 4(%%rax), %%ymm0      \n\t" // ymm0 = alpha32
    "vfmadd231ps      %%ymm1, %%ymm6,  %%ymm2    \n\t" // ymm2 += alpha31 * ymm6
    "vfmadd231ps      %%ymm1, %%ymm7,  %%ymm3    \n\t" // ymm3 += alpha31 * ymm7
    "vbroadcastss     21 * 4(%%rax), %%ymm1      \n\t" // ymm1 = 1/alpha33
    "vfmadd231ps      %%ymm0, %%ymm8,  %%ymm2    \n\t" // ymm2 += alpha32 * ymm8
    "vfmadd231ps      %%ymm0, %%ymm9,  %%ymm3    \n\t" // ymm3 += alpha32 * ymm9
    "                                            \n\t"
    "vsubps           %%ymm2,  %%ymm10, %%ymm10  \n\t" // ymm10 -= ymm2
    "vsubps           %%ymm3,  %%ymm11, %%ymm11  \n\t" // ymm11 -= ymm3
    "                                            \n\t"
    "vmulps           %%ymm10,  %%ymm1, %%ymm10  \n\t" // ymm10 *= 1/alpha33
    "vmulps           %%ymm11,  %%ymm1, %%ymm11  \n\t" // ymm11 *= 1/alpha33
    "                                            \n\t"
    "vmovups          %%ymm10, (%%rcx)           \n\t" // store in cll the final result of ymm10 & ymm11
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm11, (%%rcx)           \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t" //iteration 4
    "vbroadcastss     4 * 4(%%rax), %%ymm0       \n\t" // ymm0 = alpha40
    "vbroadcastss     10 * 4(%%rax), %%ymm1      \n\t" // ymm1 = alpha41
    "vmulps           %%ymm0,  %%ymm4, %%ymm2    \n\t" // ymm2 = alpha40 * ymm4
    "vmulps           %%ymm0,  %%ymm5, %%ymm3    \n\t" // ymm3 = alpha40 * ymm5
    "vbroadcastss     16 * 4(%%rax), %%ymm0      \n\t" // ymm0 = alpha42
    "vfmadd231ps      %%ymm1, %%ymm6,  %%ymm2    \n\t" // ymm2 += alpha41 * ymm6
    "vfmadd231ps      %%ymm1, %%ymm7,  %%ymm3    \n\t" // ymm3 += alpha41 * ymm7
    "vbroadcastss     22 * 4(%%rax), %%ymm1      \n\t" // ymm1 = alpha43
    "vfmadd231ps      %%ymm0, %%ymm8,  %%ymm2    \n\t" // ymm2 += alpha42 * ymm8
    "vfmadd231ps      %%ymm0, %%ymm9,  %%ymm3    \n\t" // ymm3 += alpha42 * ymm9
    "vbroadcastss     28 * 4(%%rax), %%ymm0      \n\t" // ymm4 = 1/alpha44
    "vfmadd231ps      %%ymm1, %%ymm10,  %%ymm2   \n\t" // ymm2 += alpha43 * ymm10
    "vfmadd231ps      %%ymm1, %%ymm11,  %%ymm3   \n\t" // ymm3 += alpha43 * ymm11
    "                                            \n\t"
    "vsubps           %%ymm2,  %%ymm12, %%ymm12  \n\t" // ymm12 -= ymm2
    "vsubps           %%ymm3,  %%ymm13, %%ymm13  \n\t" // ymm13 -= ymm3
    "                                            \n\t"
    "vmulps           %%ymm12,  %%ymm0, %%ymm12  \n\t" // ymm12 *= 1/alpha44
    "vmulps           %%ymm13,  %%ymm0, %%ymm13  \n\t" // ymm13 *= 1/alpha44
    "                                            \n\t"
    "vmovups          %%ymm12, (%%rcx)           \n\t" // store in cll the final result of ymm12 & ymm13
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm13, (%%rcx)           \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t" //iteration 5
    "vbroadcastss     5 * 4(%%rax), %%ymm0       \n\t" // ymm0 = alpha50
    "vbroadcastss     11 * 4(%%rax), %%ymm1      \n\t" // ymm1 = alpha51
    "vmulps           %%ymm0,  %%ymm4, %%ymm2    \n\t" // ymm2 = alpha50 * ymm4
    "vmulps           %%ymm0,  %%ymm5, %%ymm3    \n\t" // ymm3 = alpha50 * ymm5
    "vbroadcastss     17 * 4(%%rax), %%ymm0      \n\t" // ymm0 = alpha52
    "vfmadd231ps      %%ymm1, %%ymm6,  %%ymm2    \n\t" // ymm2 += alpha51 * ymm6
    "vfmadd231ps      %%ymm1, %%ymm7,  %%ymm3    \n\t" // ymm3 += alpha51 * ymm7
    "vbroadcastss     23 * 4(%%rax), %%ymm1      \n\t" // ymm1 = alpha53
    "vfmadd231ps      %%ymm0, %%ymm8,  %%ymm2    \n\t" // ymm2 += alpha52 * ymm8
    "vfmadd231ps      %%ymm0, %%ymm9,  %%ymm3    \n\t" // ymm3 += alpha52 * ymm9
    "vbroadcastss     29 * 4(%%rax), %%ymm0      \n\t" // ymm0 = alpha54
    "vfmadd231ps      %%ymm1, %%ymm10,  %%ymm2   \n\t" // ymm2 += alpha53 * ymm10
    "vfmadd231ps      %%ymm1, %%ymm11,  %%ymm3   \n\t" // ymm3 += alpha53 * ymm11
    "vbroadcastss     35 * 4(%%rax), %%ymm1      \n\t" // ymm1 = alpha55
    "vfmadd231ps      %%ymm0, %%ymm12,  %%ymm2   \n\t" // ymm2 += alpha54 * ymm12
    "vfmadd231ps      %%ymm0, %%ymm13,  %%ymm3   \n\t" // ymm3 += alpha54 * ymm13
    "                                            \n\t"
    "vsubps           %%ymm2,  %%ymm14, %%ymm14  \n\t" // ymm14 -= ymm2
    "vsubps           %%ymm3,  %%ymm15, %%ymm15  \n\t" // ymm15 -= ymm3
    "                                            \n\t"
    "vmulps           %%ymm14,  %%ymm1, %%ymm14  \n\t" // ymm14 *= 1/alpha55
    "vmulps           %%ymm15,  %%ymm1, %%ymm15  \n\t" // ymm15 *= 1/alpha55
    "                                            \n\t"
    "vmovups          %%ymm14, (%%rcx)           \n\t" // store in cll the final result of ymm14 & ymm15
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t" // increment:c11 +=  (8 * sizeof(float));
    "vmovups          %%ymm15, (%%rcx)           \n\t"
    "leaq   (%%rcx,%%rdi,8), %%rcx               \n\t"
    "                                            \n\t"
    "movq                %8, %%rcx               \n\t"
    "movq                %10, %%rsi              \n\t" // load cs_c
    "leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = cs_c * sizeof(float)
    "                                            \n\t"
    "movq                %9, %%rdi               \n\t" // load rs_c
    "leaq        (,%%rdi,4), %%rdi               \n\t" // rs_c *= sizeof(float)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,8), %%rdx               \n\t" // load address of c +  8*cs_c;
    "                                            \n\t"
    "leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
    "leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
    "leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
    "cmpq       $4, %%rsi                        \n\t" // set ZF if (4*cs_c) == 4.
    "jz      .SROWSTORED                         \n\t" // jump to row storage case
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm4,  %%ymm0           \n\t" // store the result in c11
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm6,  %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm8,  %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm10, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm12, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm14, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%rdx, %%rcx                      \n\t" // rcx = c + 8*cs_c
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm5,  %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm7,  %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm9,  %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm11, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm13, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm15, %%ymm0           \n\t"
    SGEMM_OUTPUT_GS_BETA_NZ
    "                                            \n\t"
    "jmp    .SDONE                               \n\t"
    ".SROWSTORED:                                \n\t"// row storage case
    "                                            \n\t"
    "vmovups          %%ymm4,  (%%rcx)           \n\t"// using vector copy to populate the results
    "addq      %%rdi, %%rcx                      \n\t"// ymm4 to ymm15 results are populated
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
    "vmovups          %%ymm15, (%%rdx)           \n\t"
    ".SDONE:                                     \n\t"
    "                                            \n\t"

    : // output operands (none)
    : // input operands
      "m" (k_iter), // 0
      "m" (k_left), // 1
      "m" (a10),      // 2
      "m" (b01),      // 3
      "m" (beta),  // 4
      "m" (alpha),   // 5
      "m" (a11),   // 6
      "m" (b11),      // 7
      "m" (c11),   // 8
      "m" (rs_c),   // 9
      "m" (cs_c)
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





