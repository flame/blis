/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

void bli_sgemm_opt_d4x4(
                         dim_t     k,
                         float*    alpha,
                         float*    a,
                         float*    b,
                         float*    beta,
                         float*    c, inc_t rs_c, inc_t cs_c
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_dgemm_opt_d4x4(
                         dim_t     k,
                         double*   alpha,
                         double*   a,
                         double*   b,
                         double*   beta,
                         double*   c, inc_t rs_c, inc_t cs_c
                       )
{
	dim_t   k_iter;
	dim_t   k_left;

	k_iter  = k / 4;
	k_left  = k % 4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movq          %2, %%rax         \n\t" // load address of a.
		"movq          %3, %%rbx         \n\t" // load address of b.
		"                                \n\t"
		"addq     $8 * 16, %%rax         \n\t" // increment pointers to allow byte
		"addq     $8 * 16, %%rbx         \n\t" // offsets in the unrolled iterations.
		"                                \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t" // of a and b.
		"movaps  -8 * 16(%%rbx), %%xmm2  \n\t"
		"                                \n\t"
		"xorpd     %%xmm3,  %%xmm3       \n\t"
		"xorpd     %%xmm4,  %%xmm4       \n\t"
		"xorpd     %%xmm5,  %%xmm5       \n\t"
		"xorpd     %%xmm6,  %%xmm6       \n\t"
		"                                \n\t"
		"xorpd     %%xmm8,  %%xmm8       \n\t"
		"movaps    %%xmm8,  %%xmm9       \n\t"
		"movaps    %%xmm8, %%xmm10       \n\t"
		"movaps    %%xmm8, %%xmm11       \n\t"
		"movaps    %%xmm8, %%xmm12       \n\t"
		"movaps    %%xmm8, %%xmm13       \n\t"
		"movaps    %%xmm8, %%xmm14       \n\t"
		"movaps    %%xmm8, %%xmm15       \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq      %0, %%rsi             \n\t" // i = k_iter;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .CONSIDERKLEFT           \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKITER:                     \n\t" // MAIN LOOP
		"                                \n\t"
		"prefetcht0 1264(%%rax)          \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 1
		"movaps  -5 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -4 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -4 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -3 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"prefetcht0 1328(%%rax)          \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 2
		"movaps  -3 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -2 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -2 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -1 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 3
		"movaps  -1 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addq   $4 * 4 * 8, %%rax        \n\t" // a += 4*4 (unroll x mr)
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps   0 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addq   $4 * 4 * 8, %%rbx        \n\t" // b += 4*4 (unroll x nr)
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .LOOPKITER               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".CONSIDERKLEFT:                 \n\t"
		"                                \n\t"
		"movq      %1, %%rsi             \n\t" // i = k_left;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .POSTACCUM               \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKLEFT:                     \n\t" // EDGE LOOP
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addq   $4 * 4 * 8, %%rax        \n\t" // a += 4 (1 x mr)
		"addq   $4 * 4 * 8, %%rbx        \n\t" // b += 4 (1 x nr)
		"                                \n\t"
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .LOOPKLEFT               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".POSTACCUM:                     \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq    %4, %%rax               \n\t" // load address of alpha
		"movq    %5, %%rbx               \n\t" // load address of beta 
		"movddup (%%rax), %%xmm6         \n\t" // load alpha and duplicate
		"movddup (%%rbx), %%xmm7         \n\t" // load beta and duplicate
		"                                \n\t"
		"                                \n\t"
		"movq    %6, %%rcx               \n\t" // load address of c
		"                                \n\t"
		"movq    %7, %%rsi               \n\t" // load rs_c
		"movq    %8, %%rdi               \n\t" // load cs_c
		"salq    $3, %%rsi               \n\t" // rs_c *= sizeof(double)
		"salq    $3, %%rdi               \n\t" // cs_c *= sizeof(double)
		"                                \n\t"
		"leaq   (%%rcx,%%rsi,2), %%rdx   \n\t" // load address of c + 2*rs_c;
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
		"                                \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
		"                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
		"                                \n\t" //
		"                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
		"                                \n\t" // ( ab21  ( ab20  ( ab23  ( ab22
		"                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
		"movaps   %%xmm8,  %%xmm0        \n\t"
		"movsd    %%xmm9,  %%xmm8        \n\t"
		"movsd    %%xmm0,  %%xmm9        \n\t"
		"                                \n\t"
		"movaps  %%xmm10,  %%xmm0        \n\t"
		"movsd   %%xmm11, %%xmm10        \n\t"
		"movsd    %%xmm0, %%xmm11        \n\t"
		"                                \n\t"
		"movaps  %%xmm12,  %%xmm0        \n\t"
		"movsd   %%xmm13, %%xmm12        \n\t"
		"movsd    %%xmm0, %%xmm13        \n\t"
		"                                \n\t"
		"movaps  %%xmm14,  %%xmm0        \n\t"
		"movsd   %%xmm15, %%xmm14        \n\t"
		"movsd    %%xmm0, %%xmm15        \n\t"
		"                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
		"                                \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
		"                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
		"                                \n\t" //
		"                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
		"                                \n\t" // ( ab20  ( ab21  ( ab22  ( ab23
		"                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c00 and c10,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm8         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm8,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c20 and c30,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm12        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm12,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c01 and c11,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm9         \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd   %%xmm9,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c21 and c31,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm13        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm13,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c02 and c12,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm10        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm10,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c22 and c32,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm14        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm14,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%rcx),       %%xmm0   \n\t" // load c03 and c13,
		"movhpd  (%%rcx,%%rsi), %%xmm0   \n\t"
		"mulpd   %%xmm6,  %%xmm11        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm0         \n\t" // scale by beta,
		"addpd  %%xmm11,  %%xmm0         \n\t" // add the gemm result,
		"movlpd  %%xmm0,  (%%rcx)        \n\t" // and store back to memory.
		"movhpd  %%xmm0,  (%%rcx,%%rsi)  \n\t"
		"addq     %%rdi, %%rcx           \n\t"
		"                                \n\t"
		"movlpd  (%%rdx),       %%xmm1   \n\t" // load c23 and c33,
		"movhpd  (%%rdx,%%rsi), %%xmm1   \n\t"
		"mulpd   %%xmm6,  %%xmm15        \n\t" // scale by alpha,
		"mulpd   %%xmm7,  %%xmm1         \n\t" // scale by beta,
		"addpd  %%xmm15,  %%xmm1         \n\t" // add the gemm result,
		"movlpd  %%xmm1,  (%%rdx)        \n\t" // and store back to memory.
		"movhpd  %%xmm1,  (%%rdx,%%rsi)  \n\t"
		"addq     %%rdi, %%rdx           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter),
		  "m" (k_left),
		  "m" (a),
		  "m" (b),
		  "m" (alpha),
		  "m" (beta),
		  "m" (c),
		  "m" (rs_c),
		  "m" (cs_c)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);

}

void bli_cgemm_opt_d4x4(
                         dim_t     k,
                         scomplex* alpha,
                         scomplex* a,
                         scomplex* b,
                         scomplex* beta,
                         scomplex* c, inc_t rs_c, inc_t cs_c
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zgemm_opt_d4x4(
                         dim_t     k,
                         dcomplex* alpha,
                         dcomplex* a,
                         dcomplex* b,
                         dcomplex* beta,
                         dcomplex* c, inc_t rs_c, inc_t cs_c
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

