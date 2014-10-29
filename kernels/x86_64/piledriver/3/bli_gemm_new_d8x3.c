/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

/* NOTE: The micro-kernels in this file were partially inspired by portions
   of code found in OpenBLAS 0.2.12 (http://www.openblas.net/). -FGVZ */

#include "blis.h"

void bli_sgemm_new_16x3(
                        dim_t              k,
                        float* restrict    alpha,
                        float* restrict    a,
                        float* restrict    b,
                        float* restrict    beta,
                        float* restrict    c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 8;
	dim_t   k_left = k % 8;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	"movq               %10, %%r14               \n\t" // load address of a_next.
	"                                            \n\t"
	"prefetcht0         128(%%rbx)               \n\t" // prefetch b
	"prefetcht0      64+128(%%rbx)               \n\t" // prefetch b
	"prefetcht0     128+128(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"addq            $32 * 4,  %%rax             \n\t"
	"addq            $12 * 4,  %%rbx             \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,4), %%rdi               \n\t" // cs_c *= sizeof(float)
	"leaq   (%%rcx,%%rdi,1), %%r10               \n\t" // load address of c + 1*cs_c;
	"leaq   (%%rcx,%%rdi,2), %%r11               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"vbroadcastss      -12 * 4(%%rbx),  %%xmm1   \n\t"
	"vbroadcastss      -11 * 4(%%rbx),  %%xmm2   \n\t"
	"vbroadcastss      -10 * 4(%%rbx),  %%xmm3   \n\t"
	"                                            \n\t"
	"vxorps    %%xmm4,  %%xmm4,  %%xmm4          \n\t"
	"vxorps    %%xmm5,  %%xmm5,  %%xmm5          \n\t"
	"vxorps    %%xmm6,  %%xmm6,  %%xmm6          \n\t"
	"vxorps    %%xmm7,  %%xmm7,  %%xmm7          \n\t"
	"vxorps    %%xmm8,  %%xmm8,  %%xmm8          \n\t"
	"vxorps    %%xmm9,  %%xmm9,  %%xmm9          \n\t"
	"vxorps    %%xmm10, %%xmm10, %%xmm10         \n\t"
	"vxorps    %%xmm11, %%xmm11, %%xmm11         \n\t"
	"vxorps    %%xmm12, %%xmm12, %%xmm12         \n\t"
	"vxorps    %%xmm13, %%xmm13, %%xmm13         \n\t"
	"vxorps    %%xmm14, %%xmm14, %%xmm14         \n\t"
	"vxorps    %%xmm15, %%xmm15, %%xmm15         \n\t"
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
	"je     .SCONSIDKLEFT                        \n\t" // if i == 0, jump to k_left code.
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0      16+192(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps           -32 * 4(%%rax),  %%xmm0   \n\t"
	"prefetcht0            384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps           -28 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps           -24 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps           -20 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss       -9 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss       -8 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovaps           -16 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss       -7 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0         64+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps           -12 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps            -8 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            -4 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss       -6 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss       -5 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovaps             0 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss       -4 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0        128+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps             4 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps             8 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            12 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss       -3 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss       -2 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovaps            16 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss       -1 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0        192+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps            20 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps            24 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            28 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss        0 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss        1 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq        $4 * 16 * 4,  %%rax             \n\t" // a += 4*16 (unroll x mr)
	"                                            \n\t"
	"                                            \n\t" // iteration 4
	"vmovaps           -32 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss        2 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0            384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps           -28 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps           -24 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps           -20 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss        3 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss        4 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"prefetcht0      80+192(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"                                            \n\t" // iteration 5
	"vmovaps           -16 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss        5 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0         64+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps           -12 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps            -8 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            -4 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss        6 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss        7 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 6
	"vmovaps             0 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss        8 * 4(%%rbx),  %%xmm3   \n\t"
	"prefetcht0        128+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps             4 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps             8 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            12 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss        9 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss       10 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 7
	"vmovaps            16 * 4(%%rax),  %%xmm0   \n\t"
	"vbroadcastss       11 * 4(%%rbx),  %%xmm3   \n\t"
	"addq        $8 *  3 * 4,  %%rbx             \n\t" // a += 4*3  (unroll x nr)
	"prefetcht0        192+384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps            20 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps            24 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps            28 * 4(%%rax),  %%xmm0   \n\t"
	"addq        $4 * 16 * 4,  %%rax             \n\t" // a += 4*16 (unroll x mr)
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss      -12 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss      -11 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"vbroadcastss      -10 * 4(%%rbx),  %%xmm3   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .SLOOPKITER                          \n\t" // jump to beginning of loop.
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
	"                                            \n\t"
	"je     .SPOSTACCUM                          \n\t" // if i == 0, we're done.
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0      16+192(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps           -32 * 4(%%rax),  %%xmm0   \n\t"
	"prefetcht0            384(%%rax)            \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps           -28 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps           -24 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps           -20 * 4(%%rax),  %%xmm0   \n\t"
	"vfmadd231ps      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vbroadcastss       -9 * 4(%%rbx),  %%xmm1   \n\t"
	"vfmadd231ps      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vbroadcastss       -8 * 4(%%rbx),  %%xmm2   \n\t"
	"vfmadd231ps      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"vbroadcastss       -7 * 4(%%rbx),  %%xmm3   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq        $1 * 16 * 4,  %%rax             \n\t" // a += 4*16 (unroll x mr)
	"addq        $1 *  3 * 4,  %%rbx             \n\t" // a += 4*3  (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .SLOOPKLEFT                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetchw    0 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetchw    0 * 8(%%r10)                   \n\t" // prefetch c + 1*cs_c
	"prefetchw    0 * 8(%%r11)                   \n\t" // prefetch c + 2*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // xmm4:   xmm5:   xmm6: 
	"                                            \n\t" // ( ab00  ( ab01  ( ab02
	"                                            \n\t" //   ab10    ab11    ab12  
	"                                            \n\t" //   ab20    ab21    ab22
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )
	"                                            \n\t"
	"                                            \n\t" // xmm7:   xmm8:   xmm9: 
	"                                            \n\t" // ( ab40  ( ab41  ( ab42
	"                                            \n\t" //   ab50    ab51    ab52  
	"                                            \n\t" //   ab60    ab61    ab62
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )
	"                                            \n\t"
	"                                            \n\t" // xmm10:  xmm11:  xmm12:
	"                                            \n\t" // ( ab80  ( ab01  ( ab02
	"                                            \n\t" //   ab90    ab11    ab12  
	"                                            \n\t" //   abA0    abA1    abA2
	"                                            \n\t" //   abB0 )  abB1 )  abB2 )
	"                                            \n\t"
	"                                            \n\t" // xmm13:  xmm14:  xmm15:
	"                                            \n\t" // ( abC0  ( abC1  ( abC2
	"                                            \n\t" //   abD0    abD1    abD2  
	"                                            \n\t" //   abE0    abE1    abE2
	"                                            \n\t" //   abF0 )  abF1 )  abF2 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastss    (%%rax), %%xmm0             \n\t" // load alpha and duplicate
	"vbroadcastss    (%%rbx), %%xmm2             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulps           %%xmm0,  %%xmm4,  %%xmm4   \n\t" // scale by alpha
	"vmulps           %%xmm0,  %%xmm5,  %%xmm5   \n\t"
	"vmulps           %%xmm0,  %%xmm6,  %%xmm6   \n\t"
	"vmulps           %%xmm0,  %%xmm7,  %%xmm7   \n\t"
	"vmulps           %%xmm0,  %%xmm8,  %%xmm8   \n\t"
	"vmulps           %%xmm0,  %%xmm9,  %%xmm9   \n\t"
	"vmulps           %%xmm0,  %%xmm10, %%xmm10  \n\t"
	"vmulps           %%xmm0,  %%xmm11, %%xmm11  \n\t"
	"vmulps           %%xmm0,  %%xmm12, %%xmm12  \n\t"
	"vmulps           %%xmm0,  %%xmm13, %%xmm13  \n\t"
	"vmulps           %%xmm0,  %%xmm14, %%xmm14  \n\t"
	"vmulps           %%xmm0,  %%xmm15, %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r14)               \n\t" // prefetch a_next
	"prefetcht0          64(%%r14)               \n\t" // prefetch a_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = rs_c * sizeof(float)
	"                                            \n\t"
	//"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  4*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $4, %%rsi                        \n\t" // set ZF if (4*rs_c) == 4.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (4*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"prefetcht0            (%%r15)               \n\t" // prefetch b_next
	"prefetcht0          64(%%r15)               \n\t" // prefetch b_next
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorps    %%xmm0,  %%xmm0,  %%xmm0          \n\t" // set xmm0 to zero.
	"vucomiss  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
	"je      .SBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .SCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t" // load c00:c30
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm4,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c00:c30
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t" // load c40:c70
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm7,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c40:c70
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t" // load c80:cB0
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm10, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c80:cB0
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t" // load cC0:cF0
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm13, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store cC0:cF0
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r10),        %%xmm0,  %%xmm0  \n\t" // load c01:c31
	"vmovhps    (%%r10,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r10,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r10,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm5,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c01:c31
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r10),        %%xmm0,  %%xmm0  \n\t" // load c41:c71
	"vmovhps    (%%r10,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r10,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r10,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm8,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c41:c71
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r10),        %%xmm0,  %%xmm0  \n\t" // load c81:cB1
	"vmovhps    (%%r10,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r10,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r10,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm11, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c81:cB1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r10),        %%xmm0,  %%xmm0  \n\t" // load cC1:cF1
	"vmovhps    (%%r10,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r10,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r10,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm14, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store cC1:cF1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r11),        %%xmm0,  %%xmm0  \n\t" // load c02:c32
	"vmovhps    (%%r11,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r11,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r11,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c02:c32
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r11),        %%xmm0,  %%xmm0  \n\t" // load c42:c72
	"vmovhps    (%%r11,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r11,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r11,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm9,  %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c42:c72
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r11),        %%xmm0,  %%xmm0  \n\t" // load c82:cB2
	"vmovhps    (%%r11,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r11,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r11,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm12, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c82:cB2
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps    (%%r11),        %%xmm0,  %%xmm0  \n\t" // load cC2:cF2
	"vmovhps    (%%r11,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%r11,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%r11,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm2,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm15, %%xmm0,  %%xmm0  \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store cC2:cF1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231ps  0 * 16(%%rcx), %%xmm2, %%xmm4  \n\t"
	"vfmadd231ps  1 * 16(%%rcx), %%xmm2, %%xmm7  \n\t"
	"vfmadd231ps  2 * 16(%%rcx), %%xmm2, %%xmm10 \n\t"
	"vfmadd231ps  3 * 16(%%rcx), %%xmm2, %%xmm13 \n\t"
	"                                            \n\t"
	"vmovups          %%xmm4,  0 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm7,  1 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm10, 2 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm13, 3 * 16(%%rcx)     \n\t"
	"                                            \n\t"
	"vfmadd231ps  0 * 16(%%r10), %%xmm2, %%xmm5  \n\t"
	"vfmadd231ps  1 * 16(%%r10), %%xmm2, %%xmm8  \n\t"
	"vfmadd231ps  2 * 16(%%r10), %%xmm2, %%xmm11 \n\t"
	"vfmadd231ps  3 * 16(%%r10), %%xmm2, %%xmm14 \n\t"
	"                                            \n\t"
	"vmovups          %%xmm5,  0 * 16(%%r10)     \n\t"
	"vmovups          %%xmm8,  1 * 16(%%r10)     \n\t"
	"vmovups          %%xmm11, 2 * 16(%%r10)     \n\t"
	"vmovups          %%xmm14, 3 * 16(%%r10)     \n\t"
	"                                            \n\t"
	"vfmadd231ps  0 * 16(%%r11), %%xmm2, %%xmm6  \n\t"
	"vfmadd231ps  1 * 16(%%r11), %%xmm2, %%xmm9  \n\t"
	"vfmadd231ps  2 * 16(%%r11), %%xmm2, %%xmm12 \n\t"
	"vfmadd231ps  3 * 16(%%r11), %%xmm2, %%xmm15 \n\t"
	"                                            \n\t"
	"vmovups          %%xmm6,  0 * 16(%%r11)     \n\t"
	"vmovups          %%xmm9,  1 * 16(%%r11)     \n\t"
	"vmovups          %%xmm12, 2 * 16(%%r11)     \n\t"
	"vmovups          %%xmm15, 3 * 16(%%r11)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .SCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm4,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c00:c30
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm7,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c40:c70
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm10, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store c80:cB0
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm13, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t" // store cC0:cF0
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r13)     \n\t"
	"leaq      (%%rcx,%%rsi,4), %%rcx            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm5,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c01:c31
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm8,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c41:c71
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm11, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store c81:cB1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm14, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10)           \n\t" // store cC1:cF1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r10,%%r13)     \n\t"
	"leaq      (%%r10,%%rsi,4), %%r10            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm6,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c02:c32
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm9,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c42:c72
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm12, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store c82:cB2
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%xmm15, %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11)           \n\t" // store cC2:cF1
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%r11,%%r13)     \n\t"
	"leaq      (%%r11,%%rsi,4), %%r11            \n\t" // c += 4*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups          %%xmm4,  0 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm7,  1 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm10, 2 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm13, 3 * 16(%%rcx)     \n\t"
	"                                            \n\t"
	"vmovups          %%xmm5,  0 * 16(%%r10)     \n\t"
	"vmovups          %%xmm8,  1 * 16(%%r10)     \n\t"
	"vmovups          %%xmm11, 2 * 16(%%r10)     \n\t"
	"vmovups          %%xmm14, 3 * 16(%%r10)     \n\t"
	"                                            \n\t"
	"vmovups          %%xmm6,  0 * 16(%%r11)     \n\t"
	"vmovups          %%xmm9,  1 * 16(%%r11)     \n\t"
	"vmovups          %%xmm12, 2 * 16(%%r11)     \n\t"
	"vmovups          %%xmm15, 3 * 16(%%r11)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SDONE:                                     \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c),   // 8
	  "m" (b_next), // 9
	  "m" (a_next)  // 10
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

void bli_dgemm_new_8x3(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double*   restrict beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 8;
	dim_t   k_left = k % 8;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	"movq               %10, %%r14               \n\t" // load address of a_next.
	"                                            \n\t"
	"prefetcht0         128(%%rbx)               \n\t" // prefetch b
	"prefetcht0      64+128(%%rbx)               \n\t" // prefetch b
	"prefetcht0     128+128(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"addq            $16 * 8,  %%rax             \n\t"
	"addq            $12 * 8,  %%rbx             \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(double)
	"leaq   (%%rcx,%%rdi,1), %%r10               \n\t" // load address of c + 1*cs_c;
	"leaq   (%%rcx,%%rdi,2), %%r11               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"vmovddup -12 * 8(%%rbx),  %%xmm1            \n\t"
	"vmovddup -11 * 8(%%rbx),  %%xmm2            \n\t"
	"vmovddup -10 * 8(%%rbx),  %%xmm3            \n\t"
	"                                            \n\t"
	"vxorpd    %%xmm4,  %%xmm4,  %%xmm4          \n\t"
	"vxorpd    %%xmm5,  %%xmm5,  %%xmm5          \n\t"
	"vxorpd    %%xmm6,  %%xmm6,  %%xmm6          \n\t"
	"vxorpd    %%xmm7,  %%xmm7,  %%xmm7          \n\t"
	"vxorpd    %%xmm8,  %%xmm8,  %%xmm8          \n\t"
	"vxorpd    %%xmm9,  %%xmm9,  %%xmm9          \n\t"
	"vxorpd    %%xmm10, %%xmm10, %%xmm10         \n\t"
	"vxorpd    %%xmm11, %%xmm11, %%xmm11         \n\t"
	"vxorpd    %%xmm12, %%xmm12, %%xmm12         \n\t"
	"vxorpd    %%xmm13, %%xmm13, %%xmm13         \n\t"
	"vxorpd    %%xmm14, %%xmm14, %%xmm14         \n\t"
	"vxorpd    %%xmm15, %%xmm15, %%xmm15         \n\t"
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
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to k_left code.
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0     -32+256(%%rbx)               \n\t" // prefetch b
	"prefetcht0      32+256(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps  -8 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0         384(%%rax)               \n\t" // prefetch a
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps  -7 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps  -6 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps  -5 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup  -9 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup  -8 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovaps  -4 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0      64+384(%%rax)               \n\t" // prefetch a
	"vmovddup  -7 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps  -3 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps  -2 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps  -1 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup  -6 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup  -5 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovaps   0 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0     128+384(%%rax)               \n\t" // prefetch a
	"vmovddup  -4 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps   1 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps   2 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps   3 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup  -3 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup  -2 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovaps   4 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0     192+384(%%rax)               \n\t" // prefetch a
	"vmovddup  -1 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps   5 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps   6 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps   7 * 16(%%rax),  %%xmm0            \n\t"
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup   0 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup   1 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 4
	"vmovaps  -8 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0         384(%%rax)               \n\t" // prefetch a
	"vmovddup   2 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps  -7 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps  -6 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps  -5 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup   3 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup   4 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"prefetcht0      96+256(%%rbx)               \n\t" // prefetch b
	"                                            \n\t"
	"                                            \n\t" // iteration 5
	"vmovaps  -4 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0      64+384(%%rax)               \n\t" // prefetch a
	"vmovddup   5 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps  -3 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps  -2 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps  -1 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup   6 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup   7 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 6
	"vmovaps   0 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0     128+384(%%rax)               \n\t" // prefetch a
	"vmovddup   8 * 8(%%rbx),  %%xmm3            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps   1 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps   2 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps   3 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup   9 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup  10 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 7
	"vmovaps   4 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0     192+384(%%rax)               \n\t" // prefetch a
	"vmovddup  11 * 8(%%rbx),  %%xmm3            \n\t"
	"addq         $8 * 3 * 8,  %%rbx             \n\t" // b += 8*3 (unroll x nr)
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps   5 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps   6 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps   7 * 16(%%rax),  %%xmm0            \n\t"
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup -12 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup -11 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"vmovddup -10 * 8(%%rbx),  %%xmm3            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .DLOOPKITER                          \n\t" // jump to beginning of loop.
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
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done.
	"                                            \n\t" // else, we prepare to
	"                                            \n\t" // enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t"
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done.
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps  -8 * 16(%%rax),  %%xmm0            \n\t"
	"prefetcht0         512(%%rax)               \n\t" // prefetch a
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm5   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm6   \n\t"
	"vmovaps  -7 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm7   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm8   \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm9   \n\t"
	"vmovaps  -6 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm10  \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm11  \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm12  \n\t"
	"vmovaps  -5 * 16(%%rax),  %%xmm0            \n\t"
	"vfmadd231pd      %%xmm1,  %%xmm0,  %%xmm13  \n\t"
	"vmovddup  -9 * 8(%%rbx),  %%xmm1            \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm14  \n\t"
	"vmovddup  -8 * 8(%%rbx),  %%xmm2            \n\t"
	"vfmadd231pd      %%xmm3,  %%xmm0,  %%xmm15  \n\t"
	"vmovddup  -7 * 8(%%rbx),  %%xmm3            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq         $1 * 8 * 8,  %%rax             \n\t" // a += 1*8 (1 x mr)
	"addq         $1 * 3 * 8,  %%rbx             \n\t" // b += 1*3 (1 x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .DLOOPKLEFT                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"prefetchw    0 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetchw    0 * 8(%%r10)                   \n\t" // prefetch c + 1*cs_c
	"prefetchw    0 * 8(%%r11)                   \n\t" // prefetch c + 2*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // xmm4:   xmm5:   xmm6:   
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  
	"                                            \n\t" //   ab10 )  ab11 )  ab12 )
	"                                            \n\t" //
	"                                            \n\t" // xmm7:   xmm8:   xmm9:   
	"                                            \n\t" // ( ab20  ( ab21  ( ab22  
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )
	"                                            \n\t" //
	"                                            \n\t" // xmm10:  xmm11:  xmm12:  
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  
	"                                            \n\t" //   ab50 )  ab51 )  ab52 )
	"                                            \n\t" //
	"                                            \n\t" // xmm13:  xmm14:  xmm15:  
	"                                            \n\t" // ( ab60  ( ab61  ( ab62  
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vmovddup        (%%rax), %%xmm0             \n\t" // load alpha and duplicate
	"vmovddup        (%%rbx), %%xmm2             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulpd           %%xmm0,  %%xmm4,  %%xmm4   \n\t" // scale by alpha
	"vmulpd           %%xmm0,  %%xmm5,  %%xmm5   \n\t"
	"vmulpd           %%xmm0,  %%xmm6,  %%xmm6   \n\t"
	"vmulpd           %%xmm0,  %%xmm7,  %%xmm7   \n\t"
	"vmulpd           %%xmm0,  %%xmm8,  %%xmm8   \n\t"
	"vmulpd           %%xmm0,  %%xmm9,  %%xmm9   \n\t"
	"vmulpd           %%xmm0,  %%xmm10, %%xmm10  \n\t"
	"vmulpd           %%xmm0,  %%xmm11, %%xmm11  \n\t"
	"vmulpd           %%xmm0,  %%xmm12, %%xmm12  \n\t"
	"vmulpd           %%xmm0,  %%xmm13, %%xmm13  \n\t"
	"vmulpd           %%xmm0,  %%xmm14, %%xmm14  \n\t"
	"vmulpd           %%xmm0,  %%xmm15, %%xmm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r14)               \n\t" // prefetch a_next
	"prefetcht0          64(%%r14)               \n\t" // prefetch a_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  8*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"prefetcht0            (%%r15)               \n\t" // prefetch b_next
	"prefetcht0          64(%%r15)               \n\t" // prefetch b_next
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%xmm0,  %%xmm0,  %%xmm0          \n\t" // set xmm0 to zero.
	"vucomisd  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
	"je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"je      .DGENSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // xmm4:   xmm5:   xmm6:   
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  
	"                                            \n\t" //   ab10 )  ab11 )  ab12 )
	"                                            \n\t" //
	"                                            \n\t" // xmm7:   xmm8:   xmm9:   
	"                                            \n\t" // ( ab20  ( ab21  ( ab22  
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )
	"                                            \n\t" //
	"                                            \n\t" // xmm10:  xmm11:  xmm12:  
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  
	"                                            \n\t" //   ab50 )  ab51 )  ab52 )
	"                                            \n\t" //
	"                                            \n\t" // xmm13:  xmm14:  xmm15:  
	"                                            \n\t" // ( ab60  ( ab61  ( ab62  
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231pd  0 * 16(%%rcx), %%xmm2, %%xmm4  \n\t"
	"vfmadd231pd  1 * 16(%%rcx), %%xmm2, %%xmm7  \n\t"
	"vfmadd231pd  2 * 16(%%rcx), %%xmm2, %%xmm10 \n\t"
	"vfmadd231pd  3 * 16(%%rcx), %%xmm2, %%xmm13 \n\t"
	"                                            \n\t"
	"vfmadd231pd  0 * 16(%%r10), %%xmm2, %%xmm5  \n\t"
	"vfmadd231pd  1 * 16(%%r10), %%xmm2, %%xmm8  \n\t"
	"vfmadd231pd  2 * 16(%%r10), %%xmm2, %%xmm11 \n\t"
	"vfmadd231pd  3 * 16(%%r10), %%xmm2, %%xmm14 \n\t"
	"                                            \n\t"
	"vfmadd231pd  0 * 16(%%r11), %%xmm2, %%xmm6  \n\t"
	"vfmadd231pd  1 * 16(%%r11), %%xmm2, %%xmm9  \n\t"
	"vfmadd231pd  2 * 16(%%r11), %%xmm2, %%xmm12 \n\t"
	"vfmadd231pd  3 * 16(%%r11), %%xmm2, %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups          %%xmm4,  0 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm7,  1 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm10, 2 * 16(%%rcx)     \n\t"
	"vmovups          %%xmm13, 3 * 16(%%rcx)     \n\t"
	"                                            \n\t"
	"vmovups          %%xmm5,  0 * 16(%%r10)     \n\t"
	"vmovups          %%xmm8,  1 * 16(%%r10)     \n\t"
	"vmovups          %%xmm11, 2 * 16(%%r10)     \n\t"
	"vmovups          %%xmm14, 3 * 16(%%r10)     \n\t"
	"                                            \n\t"
	"vmovups          %%xmm6,  0 * 16(%%r11)     \n\t"
	"vmovups          %%xmm9,  1 * 16(%%r11)     \n\t"
	"vmovups          %%xmm12, 2 * 16(%%r11)     \n\t"
	"vmovups          %%xmm15, 3 * 16(%%r11)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
/*
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load c00:c10
	"vmovupd    (%%rcx,%%r12), %%xmm1            \n\t" // load c20:c30
	"vfmadd231pd      %%xmm2,  %%xmm0,  %%xmm4   \n\t"
	"vfmadd231pd      %%xmm2,  %%xmm1,  %%xmm7   \n\t"
	"vmovupd          %%xmm4,  (%%rcx)           \n\t" // store c00:c10
	"vmovupd          %%xmm7,  (%%rcx,%%r12)     \n\t" // store c20:c30
	"addq      %%rdi, %%rcx                      \n\t"
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load c40:c50
	"vmovupd    (%%rdx,%%r12), %%xmm1            \n\t" // load c60:c70
	"vfmadd213pd      %%xmm10, %%xmm2,  %%xmm0   \n\t"
	"vfmadd213pd      %%xmm13, %%xmm2,  %%xmm1   \n\t"
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store c40:c50
	"vmovupd          %%xmm1,  (%%rdx,%%r12)     \n\t" // store c60:c70
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load c01:c11
	"vmovupd    (%%rcx,%%r12), %%xmm1            \n\t" // load c21:c31
	"vfmadd213pd      %%xmm5,  %%xmm2,  %%xmm0   \n\t"
	"vfmadd213pd      %%xmm8,  %%xmm2,  %%xmm1   \n\t"
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store c01:c11
	"vmovupd          %%xmm1,  (%%rcx,%%r12)     \n\t" // store c21:c31
	"addq      %%rdi, %%rcx                      \n\t"
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load c41:c51
	"vmovupd    (%%rdx,%%r12), %%xmm1            \n\t" // load c61:c71
	"vfmadd213pd      %%xmm11, %%xmm2,  %%xmm0   \n\t"
	"vfmadd213pd      %%xmm14, %%xmm2,  %%xmm1   \n\t"
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store c41:c51
	"vmovupd          %%xmm1,  (%%rdx,%%r12)     \n\t" // store c61:c71
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load c02:c12
	"vmovupd    (%%rcx,%%r12), %%xmm1            \n\t" // load c22:c32
	"vfmadd213pd      %%xmm6,  %%xmm2,  %%xmm0   \n\t"
	"vfmadd213pd      %%xmm9,  %%xmm2,  %%xmm1   \n\t"
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store c02:c12
	"vmovupd          %%xmm1,  (%%rcx,%%r12)     \n\t" // store c22:c32
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load c42:c52
	"vmovupd    (%%rdx,%%r12), %%xmm1            \n\t" // load c62:c72
	"vfmadd213pd      %%xmm12, %%xmm2,  %%xmm0   \n\t"
	"vfmadd213pd      %%xmm15, %%xmm2,  %%xmm1   \n\t"
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store c42:c52
	"vmovupd          %%xmm1,  (%%rdx,%%r12)     \n\t" // store c62:c72
*/
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c00:c10
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm4,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store c00:c10
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c20:c30
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm7,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // store c20:c30
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c40:c50
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm10, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store c40:c50
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c60:c70
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm13, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // store c60:c70
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c01:c11
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm5,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store c01:c11
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c21:c31
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm8,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // store c21:c31
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c41:c51
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm11, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store c41:c51
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c61:c71
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm14, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // store c61:c71
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c02:c12
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm6,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store c02:c12
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c22:c32
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm9,  %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // store c22:c32
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c42:c52
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm12, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store c42:c52
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c62:c72
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t"
	"vaddpd           %%xmm15, %%xmm0,  %%xmm0   \n\t"
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // store c62:c72
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .DCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlpd          %%xmm4,  (%%rcx)           \n\t"
	"vmovhpd          %%xmm4,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm7,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm7,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovlpd          %%xmm10, (%%rdx)           \n\t"
	"vmovhpd          %%xmm10, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm13, (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm13, (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovlpd          %%xmm5,  (%%rcx)           \n\t"
	"vmovhpd          %%xmm5,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm8,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm8,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovlpd          %%xmm11, (%%rdx)           \n\t"
	"vmovhpd          %%xmm11, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm14, (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm14, (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovlpd          %%xmm6,  (%%rcx)           \n\t"
	"vmovhpd          %%xmm6,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm9,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm9,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovlpd          %%xmm12, (%%rdx)           \n\t"
	"vmovhpd          %%xmm12, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm15, (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm15, (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovupd          %%xmm4,  (%%rcx)           \n\t"
	"vmovupd          %%xmm7,  (%%rcx,%%r12)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%xmm10, (%%rdx)           \n\t"
	"vmovupd          %%xmm13, (%%rdx,%%r12)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%xmm5,  (%%rcx)           \n\t"
	"vmovupd          %%xmm8,  (%%rcx,%%r12)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%xmm11, (%%rdx)           \n\t"
	"vmovupd          %%xmm14, (%%rdx,%%r12)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"vmovupd          %%xmm6,  (%%rcx)           \n\t"
	"vmovupd          %%xmm9,  (%%rcx,%%r12)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%xmm12, (%%rdx)           \n\t"
	"vmovupd          %%xmm15, (%%rdx,%%r12)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c),   // 8
	  "m" (b_next), // 9
	  "m" (a_next)  // 10
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

void bli_cgemm_new_4x2(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a,
                        scomplex* restrict b,
                        scomplex* restrict beta,
                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 8;
	dim_t   k_left = k % 8;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	"movq               %10, %%r14               \n\t" // load address of a_next.
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(scomplex)
	"leaq   (%%rcx,%%rdi,1), %%r10               \n\t" // load address of c + 1*cs_c;
	"                                            \n\t"
	"addq            $32 * 4,  %%rax             \n\t"
	"addq            $16 * 4,  %%rbx             \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vxorps    %%xmm8,  %%xmm8,  %%xmm8          \n\t"
	"vxorps    %%xmm9,  %%xmm9,  %%xmm9          \n\t"
	"vxorps    %%xmm10, %%xmm10, %%xmm10         \n\t"
	"vxorps    %%xmm11, %%xmm11, %%xmm11         \n\t"
	"vxorps    %%xmm12, %%xmm12, %%xmm12         \n\t"
	"vxorps    %%xmm13, %%xmm13, %%xmm13         \n\t"
	"vxorps    %%xmm14, %%xmm14, %%xmm14         \n\t"
	"vxorps    %%xmm15, %%xmm15, %%xmm15         \n\t"
	//"vzeroall                                    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .CCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".CLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"je     .CCONSIDKLEFT                        \n\t" // if i == 0, jump to k_left code.
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0         256(%%rbx)               \n\t"
	"prefetcht0         512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps            -32 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss       -16 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps            -28 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss       -15 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss       -14 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss       -13 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovaps            -24 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss       -12 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps            -20 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss       -11 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss       -10 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss        -9 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0      64+256(%%rbx)               \n\t"
	"prefetcht0      64+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovaps            -16 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss        -8 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps            -12 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss        -7 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss        -6 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss        -5 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovaps             -8 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss        -4 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps             -4 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss        -3 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss        -2 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss        -1 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0     128+256(%%rbx)               \n\t"
	"prefetcht0     128+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 4
	"vmovaps              0 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss         0 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps              4 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss         1 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss         2 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss         3 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 5
	"vmovaps              8 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss         4 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps             12 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss         5 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss         6 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss         7 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0     128+256(%%rbx)               \n\t"
	"prefetcht0     128+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 6
	"vmovaps             16 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss         8 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps             20 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss         9 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss        10 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss        11 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 7
	"vmovaps             24 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss        12 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps             28 * 4(%%rax),  %%xmm1  \n\t"
	"addq           $8 * 4 * 8, %%rax            \n\t" // a += 8*2 (unroll x mr)
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss        13 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss        14 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss        15 * 4(%%rbx),  %%xmm7  \n\t"
	"addq           $8 * 2 * 8, %%rbx            \n\t" // b += 8*2 (unroll x nr)
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .CLOOPKITER                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .CPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".CLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t"
	"je     .CPOSTACCUM                          \n\t" // if i == 0, we're done.
	"                                            \n\t"
	"prefetcht0         256(%%rbx)               \n\t"
	"prefetcht0         512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps            -32 * 4(%%rax),  %%xmm0  \n\t"
	"vbroadcastss       -16 * 4(%%rbx),  %%xmm4  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps            -28 * 4(%%rax),  %%xmm1  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vbroadcastss       -15 * 4(%%rbx),  %%xmm5  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vbroadcastss       -14 * 4(%%rbx),  %%xmm6  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vbroadcastss       -13 * 4(%%rbx),  %%xmm7  \n\t"
	"vfmadd231ps       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231ps       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq           $1 * 4 * 8, %%rax             \n\t" // a += 1*2 (1 x mr)
	"addq           $1 * 2 * 8, %%rbx             \n\t" // b += 1*2 (1 x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .CLOOPKLEFT                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetchw    0 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetchw    0 * 8(%%r10)                   \n\t" // prefetch c + 1*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vpermilps  $0xb1, %%xmm9,  %%xmm9           \n\t"
	"vpermilps  $0xb1, %%xmm11, %%xmm11          \n\t"
	"vpermilps  $0xb1, %%xmm13, %%xmm13          \n\t"
	"vpermilps  $0xb1, %%xmm15, %%xmm15          \n\t"
	"                                            \n\t"
	"vaddsubps         %%xmm9,  %%xmm8,  %%xmm8  \n\t"
	"vaddsubps         %%xmm11, %%xmm10, %%xmm10 \n\t"
	"vaddsubps         %%xmm13, %%xmm12, %%xmm12 \n\t"
	"vaddsubps         %%xmm15, %%xmm14, %%xmm14 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // xmm8:   xmm10:
	"                                            \n\t" // ( ab00  ( ab01
	"                                            \n\t" //   ab10    ab11
	"                                            \n\t" //   ab20    ab21
	"                                            \n\t" //   ab30 )  ab31 )
	"                                            \n\t"
	"                                            \n\t" // xmm12:  xmm14:
	"                                            \n\t" // ( ab40  ( ab41
	"                                            \n\t" //   ab50    ab51
	"                                            \n\t" //   ab60    ab61
	"                                            \n\t" //   ab70 )  ab71 )
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r14)               \n\t" // prefetch a_next
	"prefetcht0          64(%%r14)               \n\t" // prefetch a_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // scale by alpha
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"vbroadcastss    (%%rax), %%xmm0             \n\t" // load alpha_r and duplicate
	"vbroadcastss   4(%%rax), %%xmm1             \n\t" // load alpha_i and duplicate
	"                                            \n\t"
	"vpermilps  $0xb1, %%xmm8,  %%xmm9           \n\t"
	"vpermilps  $0xb1, %%xmm10, %%xmm11          \n\t"
	"vpermilps  $0xb1, %%xmm12, %%xmm13          \n\t"
	"vpermilps  $0xb1, %%xmm14, %%xmm15          \n\t"
	"                                            \n\t"
	"vmulps            %%xmm8,  %%xmm0,  %%xmm8  \n\t"
	"vmulps            %%xmm10, %%xmm0,  %%xmm10 \n\t"
	"vmulps            %%xmm12, %%xmm0,  %%xmm12 \n\t"
	"vmulps            %%xmm14, %%xmm0,  %%xmm14 \n\t"
	"                                            \n\t"
	"vmulps            %%xmm9,  %%xmm1,  %%xmm9  \n\t"
	"vmulps            %%xmm11, %%xmm1,  %%xmm11 \n\t"
	"vmulps            %%xmm13, %%xmm1,  %%xmm13 \n\t"
	"vmulps            %%xmm15, %%xmm1,  %%xmm15 \n\t"
	"                                            \n\t"
	"vaddsubps         %%xmm9,  %%xmm8,  %%xmm8  \n\t"
	"vaddsubps         %%xmm11, %%xmm10, %%xmm10 \n\t"
	"vaddsubps         %%xmm13, %%xmm12, %%xmm12 \n\t"
	"vaddsubps         %%xmm15, %%xmm14, %%xmm14 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastss    (%%rbx), %%xmm6             \n\t" // load beta_r and duplicate
	"vbroadcastss   4(%%rbx), %%xmm7             \n\t" // load beta_i and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(scomplex)
	"                                            \n\t"
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r15)               \n\t" // prefetch b_next
	"prefetcht0          64(%%r15)               \n\t" // prefetch b_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  8*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorps    %%xmm0,  %%xmm0,  %%xmm0          \n\t" // set xmm0 to zero.
	"vucomiss  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_r == 0.
	"sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
	"vucomiss  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_i == 0.
	"sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
	"andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
	"jne      .CBETAZERO                         \n\t" // if ZF = 0, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .CCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps     (%%rcx),       %%xmm0,  %%xmm0  \n\t" // load c00:c10
	"vmovhps     (%%rcx,%%rsi), %%xmm0,  %%xmm0  \n\t"
	"vmovlps     (%%rcx,%%r12), %%xmm2,  %%xmm2  \n\t" // load c20:c30
	"vmovhps     (%%rcx,%%r13), %%xmm2,  %%xmm2  \n\t"
	"vpermilps  $0xb1, %%xmm0,  %%xmm1           \n\t"
	"vpermilps  $0xb1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubps         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm8,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps           %%xmm0,  (%%rcx)          \n\t" // store c00:c10
	"vmovhps           %%xmm0,  (%%rcx,%%rsi)    \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulps            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubps         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddps            %%xmm12, %%xmm2,  %%xmm2  \n\t"
	"vmovlps           %%xmm2,  (%%rcx,%%r12)    \n\t" // store c20:c30
	"vmovhps           %%xmm2,  (%%rcx,%%r13)    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps     (%%r10),       %%xmm0,  %%xmm0  \n\t" // load c01:c11
	"vmovhps     (%%r10,%%rsi), %%xmm0,  %%xmm0  \n\t"
	"vmovlps     (%%r10,%%r12), %%xmm2,  %%xmm2  \n\t" // load c21:c31
	"vmovhps     (%%r10,%%r13), %%xmm2,  %%xmm2  \n\t"
	"vpermilps  $0xb1, %%xmm0,  %%xmm1           \n\t"
	"vpermilps  $0xb1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubps         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm10, %%xmm0,  %%xmm0  \n\t"
	"vmovlps           %%xmm0,  (%%r10)          \n\t" // store c01:c11
	"vmovhps           %%xmm0,  (%%r10,%%rsi)    \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulps            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubps         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddps            %%xmm14, %%xmm2,  %%xmm2  \n\t"
	"vmovlps           %%xmm2,  (%%r10,%%r12)    \n\t" // store c21:c31
	"vmovhps           %%xmm2,  (%%r10,%%r13)    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%rcx),       %%xmm0           \n\t" // load c00:c10
	"vmovups   16(%%rcx),       %%xmm2           \n\t" // load c20:c30
	"vpermilps  $0xb1, %%xmm0,  %%xmm1           \n\t"
	"vpermilps  $0xb1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubps         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm8,  %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,   (%%rcx)         \n\t" // store c00:c10
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulps            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubps         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddps            %%xmm12, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2, 16(%%rcx)         \n\t" // store c20:c30
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%r10),       %%xmm0           \n\t" // load c01:c11
	"vmovups   16(%%r10),       %%xmm2           \n\t" // load c21:c31
	"vpermilps  $0xb1, %%xmm0,  %%xmm1           \n\t"
	"vpermilps  $0xb1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulps            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubps         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddps            %%xmm10, %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,   (%%r10)         \n\t" // store c01:c11
	"                                            \n\t"
	"vmulps            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulps            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubps         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddps            %%xmm14, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2, 16(%%r10)         \n\t" // store c21:c31
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .CCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovlps           %%xmm8,  (%%rcx)          \n\t" // store c00:c10
	"vmovhps           %%xmm8,  (%%rcx,%%rsi)    \n\t"
	"                                            \n\t"
	"vmovlps           %%xmm12, (%%rcx,%%r12)    \n\t" // store c20:c30
	"vmovhps           %%xmm12, (%%rcx,%%r13)    \n\t"
	"                                            \n\t"
	"vmovlps           %%xmm10, (%%r10)          \n\t" // store c01:c11
	"vmovhps           %%xmm10, (%%r10,%%rsi)    \n\t"
	"                                            \n\t"
	"vmovlps           %%xmm14, (%%r10,%%r12)    \n\t" // store c21:c31
	"vmovhps           %%xmm14, (%%r10,%%r13)    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups           %%xmm8,    (%%rcx)        \n\t" // store c00:c10
	"vmovups           %%xmm12, 16(%%rcx)        \n\t" // store c20:c30
	"                                            \n\t"
	"vmovups           %%xmm10,   (%%r10)        \n\t" // store c01:c11
	"vmovups           %%xmm14, 16(%%r10)        \n\t" // store c21:c31
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CDONE:                                     \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c),   // 8
	  "m" (b_next), // 9
	  "m" (a_next)  // 10
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

void bli_zgemm_new_2x2(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a,
                        dcomplex* restrict b,
                        dcomplex* restrict beta,
                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 8;
	dim_t   k_left = k % 8;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	"movq               %10, %%r14               \n\t" // load address of a_next.
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(dcomplex)
	"leaq        (,%%rdi,2), %%rdi               \n\t"
	"leaq   (%%rcx,%%rdi,1), %%r10               \n\t" // load address of c + 1*cs_c;
	"                                            \n\t"
	"addq            $16 * 8,  %%rax             \n\t"
	"addq            $16 * 8,  %%rbx             \n\t"
	"                                            \n\t"
	"vxorpd    %%xmm8,  %%xmm8,  %%xmm8          \n\t"
	"vxorpd    %%xmm9,  %%xmm9,  %%xmm9          \n\t"
	"vxorpd    %%xmm10, %%xmm10, %%xmm10         \n\t"
	"vxorpd    %%xmm11, %%xmm11, %%xmm11         \n\t"
	"vxorpd    %%xmm12, %%xmm12, %%xmm12         \n\t"
	"vxorpd    %%xmm13, %%xmm13, %%xmm13         \n\t"
	"vxorpd    %%xmm14, %%xmm14, %%xmm14         \n\t"
	"vxorpd    %%xmm15, %%xmm15, %%xmm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .ZCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".ZLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"je     .ZCONSIDKLEFT                        \n\t" // if i == 0, jump to k_left code.
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0         256(%%rbx)               \n\t"
	"                                            \n\t"
	"prefetcht0         512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps   -16 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup  -16 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps   -14 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup  -15 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup  -14 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup  -13 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps   -12 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup  -12 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps   -10 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup  -11 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup  -10 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup   -9 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps    -8 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup   -8 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0      64+256(%%rbx)               \n\t"
	"                                            \n\t"
	"prefetcht0      64+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps    -6 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup   -7 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup   -6 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup   -5 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps    -4 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup   -4 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps    -2 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup   -3 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup   -2 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup   -1 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps     0 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup    0 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0     128+256(%%rbx)               \n\t"
	"                                            \n\t"
	"prefetcht0     128+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 4
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps     2 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup    1 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup    2 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup    3 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps     4 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup    4 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 5
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps     6 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup    5 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup    6 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup    7 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps     8 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup    8 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"prefetcht0     128+256(%%rbx)               \n\t"
	"                                            \n\t"
	"prefetcht0     128+512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 6
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps    10 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup    9 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup   10 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup   11 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vmovaps    12 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup   12 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 7
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps    14 * 8(%%rax),  %%xmm1           \n\t"
	"addq         $8 * 2 * 16, %%rax             \n\t" // a += 8*2 (unroll x mr)
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup   13 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup   14 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup   15 * 8(%%rbx),  %%xmm7           \n\t"
	"addq         $8 * 2 * 16, %%rbx             \n\t" // b += 8*2 (unroll x nr)
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .ZLOOPKITER                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .ZPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".ZLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t"
	"je     .ZPOSTACCUM                          \n\t" // if i == 0, we're done.
	"                                            \n\t"
	"prefetcht0         256(%%rbx)               \n\t"
	"                                            \n\t"
	"prefetcht0         512(%%rax)               \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovaps   -16 * 8(%%rax),  %%xmm0           \n\t"
	"vmovddup  -16 * 8(%%rbx),  %%xmm4           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm4,  %%xmm8  \n\t"
	"vmovaps   -14 * 8(%%rax),  %%xmm1           \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm4,  %%xmm12 \n\t"
	"vmovddup  -15 * 8(%%rbx),  %%xmm5           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm5,  %%xmm9  \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm5,  %%xmm13 \n\t"
	"vmovddup  -14 * 8(%%rbx),  %%xmm6           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm6,  %%xmm10 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm6,  %%xmm14 \n\t"
	"vmovddup  -13 * 8(%%rbx),  %%xmm7           \n\t"
	"vfmadd231pd       %%xmm0,  %%xmm7,  %%xmm11 \n\t"
	"vfmadd231pd       %%xmm1,  %%xmm7,  %%xmm15 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq         $1 * 2 * 16, %%rax             \n\t" // a += 1*2 (1 x mr)
	"addq         $1 * 2 * 16, %%rbx             \n\t" // b += 1*2 (1 x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jmp    .ZLOOPKLEFT                          \n\t" // jump to beginning of loop.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetchw    0 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetchw    0 * 8(%%r10)                   \n\t" // prefetch c + 1*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vpermilpd   $0x1, %%xmm9,  %%xmm9           \n\t"
	"vpermilpd   $0x1, %%xmm11, %%xmm11          \n\t"
	"vpermilpd   $0x1, %%xmm13, %%xmm13          \n\t"
	"vpermilpd   $0x1, %%xmm15, %%xmm15          \n\t"
	"                                            \n\t"
	"vaddsubpd         %%xmm9,  %%xmm8,  %%xmm8  \n\t"
	"vaddsubpd         %%xmm11, %%xmm10, %%xmm10 \n\t"
	"vaddsubpd         %%xmm13, %%xmm12, %%xmm12 \n\t"
	"vaddsubpd         %%xmm15, %%xmm14, %%xmm14 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // xmm8:   xmm10:
	"                                            \n\t" // ( ab00  ( ab01
	"                                            \n\t" //   ab10 )  ab11 )
	"                                            \n\t"
	"                                            \n\t" // xmm12:  xmm14:
	"                                            \n\t" // ( ab20  ( ab21
	"                                            \n\t" //   ab30 )  ab31 )
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r14)               \n\t" // prefetch a_next
	"prefetcht0          64(%%r14)               \n\t" // prefetch a_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // scale by alpha
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"vmovddup        (%%rax), %%xmm0             \n\t" // load alpha_r and duplicate
	"vmovddup       8(%%rax), %%xmm1             \n\t" // load alpha_i and duplicate
	"                                            \n\t"
	"vpermilpd   $0x1, %%xmm8,  %%xmm9           \n\t"
	"vpermilpd   $0x1, %%xmm10, %%xmm11          \n\t"
	"vpermilpd   $0x1, %%xmm12, %%xmm13          \n\t"
	"vpermilpd   $0x1, %%xmm14, %%xmm15          \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm8,  %%xmm0,  %%xmm8  \n\t"
	"vmulpd            %%xmm10, %%xmm0,  %%xmm10 \n\t"
	"vmulpd            %%xmm12, %%xmm0,  %%xmm12 \n\t"
	"vmulpd            %%xmm14, %%xmm0,  %%xmm14 \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm9,  %%xmm1,  %%xmm9  \n\t"
	"vmulpd            %%xmm11, %%xmm1,  %%xmm11 \n\t"
	"vmulpd            %%xmm13, %%xmm1,  %%xmm13 \n\t"
	"vmulpd            %%xmm15, %%xmm1,  %%xmm15 \n\t"
	"                                            \n\t"
	"vaddsubpd         %%xmm9,  %%xmm8,  %%xmm8  \n\t"
	"vaddsubpd         %%xmm11, %%xmm10, %%xmm10 \n\t"
	"vaddsubpd         %%xmm13, %%xmm12, %%xmm12 \n\t"
	"vaddsubpd         %%xmm15, %%xmm14, %%xmm14 \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vmovddup        (%%rbx), %%xmm6             \n\t" // load beta_r and duplicate
	"vmovddup       8(%%rbx), %%xmm7             \n\t" // load beta_i and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(dcomplex)
	"leaq        (,%%rsi,2), %%rsi               \n\t"
	//"leaq   (%%rcx,%%rsi,2), %%rdx               \n\t" // load address of c + 2*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0            (%%r15)               \n\t" // prefetch b_next
	"prefetcht0          64(%%r15)               \n\t" // prefetch b_next
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" // 16*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq      $16, %%rsi                        \n\t" // set ZF if (16*rs_c) == 16.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (16*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%xmm0,  %%xmm0,  %%xmm0          \n\t" // set xmm0 to zero.
	"vucomisd  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_r == 0.
	"sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
	"vucomisd  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_i == 0.
	"sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
	"andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
	"jne      .ZBETAZERO                         \n\t" // if ZF = 0, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .ZCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%rcx),       %%xmm0           \n\t" // load c00
	"vmovups     (%%rcx,%%rsi), %%xmm2           \n\t" // load c10
	"vpermilpd   $0x1, %%xmm0,  %%xmm1           \n\t"
	"vpermilpd   $0x1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulpd            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubpd         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddpd            %%xmm8,  %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,  (%%rcx)          \n\t" // store c00
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulpd            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubpd         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddpd            %%xmm12, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2,  (%%rcx,%%rsi)    \n\t" // store c10
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%r10),       %%xmm0           \n\t" // load c01
	"vmovups     (%%r10,%%rsi), %%xmm2           \n\t" // load c11
	"vpermilpd   $0x1, %%xmm0,  %%xmm1           \n\t"
	"vpermilpd   $0x1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulpd            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubpd         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddpd            %%xmm10, %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,  (%%r10)          \n\t" // store c01
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulpd            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubpd         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddpd            %%xmm14, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2,  (%%r10,%%rsi)    \n\t" // store c11
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%rcx),       %%xmm0           \n\t" // load c00
	"vmovups   16(%%rcx),       %%xmm2           \n\t" // load c10
	"vpermilpd   $0x1, %%xmm0,  %%xmm1           \n\t"
	"vpermilpd   $0x1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulpd            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubpd         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddpd            %%xmm8,  %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,   (%%rcx)         \n\t" // store c00
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulpd            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubpd         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddpd            %%xmm12, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2, 16(%%rcx)         \n\t" // store c10
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups     (%%r10),       %%xmm0           \n\t" // load c01
	"vmovups   16(%%r10),       %%xmm2           \n\t" // load c11
	"vpermilpd   $0x1, %%xmm0,  %%xmm1           \n\t"
	"vpermilpd   $0x1, %%xmm2,  %%xmm3           \n\t"
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm0,  %%xmm0  \n\t"
	"vmulpd            %%xmm7,  %%xmm1,  %%xmm1  \n\t"
	"vaddsubpd         %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vaddpd            %%xmm10, %%xmm0,  %%xmm0  \n\t"
	"vmovups           %%xmm0,   (%%r10)         \n\t" // store c01
	"                                            \n\t"
	"vmulpd            %%xmm6,  %%xmm2,  %%xmm2  \n\t"
	"vmulpd            %%xmm7,  %%xmm3,  %%xmm3  \n\t"
	"vaddsubpd         %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vaddpd            %%xmm14, %%xmm2,  %%xmm2  \n\t"
	"vmovups           %%xmm2, 16(%%r10)         \n\t" // store c11
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .ZCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups           %%xmm8,  (%%rcx)          \n\t" // store c00
	"vmovups           %%xmm12, (%%rcx,%%rsi)    \n\t" // store c10
	"                                            \n\t"
	"vmovups           %%xmm10, (%%r10)          \n\t" // store c01
	"vmovups           %%xmm14, (%%r10,%%rsi)    \n\t" // store c11
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups           %%xmm8,    (%%rcx)        \n\t" // store c00
	"vmovups           %%xmm12, 16(%%rcx)        \n\t" // store c10
	"                                            \n\t"
	"vmovups           %%xmm10,   (%%r10)        \n\t" // store c01
	"vmovups           %%xmm14, 16(%%r10)        \n\t" // store c11
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZDONE:                                     \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c),   // 8
	  "m" (b_next), // 9
	  "m" (a_next)  // 10
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

